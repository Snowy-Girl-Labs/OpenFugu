#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: TRINITY (arXiv:2512.04695). Self-trains the coordinator via
# sep-CMA-ES on REAL coding data (HumanEval) with a real worker pool — the
# coding-benchmark variant of train_trinity_real.py. Original code.
"""
train_trinity_coding.py — TRINITY self-training on HumanEval coding benchmark.

Same sep-CMA-ES loop as train_trinity_real.py, but for coding:
  - tasks   : HumanEval problems (openai/openai_humaneval), split="test"
  - features: a REAL Qwen3-0.6B penultimate hidden state of the prompt
  - workers : a real pool via litellm (Novita), differing models
  - reward  : real code execution (verifiable signal via Python subprocess execution)

The coordinator (bias-free linear head over the hidden state) learns which
worker to send each coding problem to, to maximize solved rate.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, tempfile, time
import numpy as np

HIDDEN = 1024
HIDDEN_POS = -2


def extract_completion(text: str) -> str:
    """Extract candidate completion from fenced block or use raw response as-is."""
    m = re.search(r"```python\n(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\n(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


class Backbone:
    """Real Qwen3-0.6B -> penultimate hidden state of a question (the router feature)."""
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self._cache = {}

    def feature(self, question: str) -> np.ndarray:
        if question in self._cache:
            return self._cache[question]
        torch = self.torch
        ids = self.tok(f"user: {question}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model.model(**ids).last_hidden_state[0, HIDDEN_POS, :]
        v = h.float().cpu().numpy()
        self._cache[question] = v
        return v


def route(head_vec, feat, n_workers):
    """Bias-free linear head -> worker id (argmax), faithful to mini.py."""
    W = head_vec.reshape(n_workers, HIDDEN)
    return int(np.argmax(W @ feat))


def main():
    ap = argparse.ArgumentParser(description="TRINITY self-train on HumanEval (minimal).")
    ap.add_argument("--model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--slot-models", required=True, help="csv of litellm worker ids (the pool)")
    ap.add_argument("--n-train", type=int, default=40, help="HumanEval questions (kept small/cheap)")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="trinity_humaneval.npy")
    args = ap.parse_args()

    import cma, litellm
    from datasets import load_dataset

    workers = args.slot_models.split(",")
    n_workers = len(workers)
    api_key = os.environ.get("FUGU_API_KEY") or os.environ.get("NOVITA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("FUGU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    ds = load_dataset("openai/openai_humaneval", split="test")
    tasks = list(ds)[:args.n_train]
    print(f"[real-train] {len(tasks)} HumanEval tasks, {n_workers} workers: {workers}", flush=True)

    bb = Backbone(args.model)
    feats = [bb.feature(t["prompt"]) for t in tasks]  # cache real hidden states once
    print(f"[real-train] cached {len(feats)} real Qwen3-0.6B features (dim {feats[0].shape[0]})", flush=True)

    # worker call cache: (worker_id, task_id) -> solved? so CMA candidates reuse answers
    solve_cache: dict = {}

    def worker_solves(wid, task):
        task_id = task["task_id"]
        key = (wid, task_id)
        if key in solve_cache:
            return solve_cache[key]

        prompt = task["prompt"]
        test = task["test"]
        entry_point = task["entry_point"]

        worker_prompt = prompt + "\nComplete this Python function. Output ONLY the indented function body (do NOT repeat the `def` line or docstring), in a single ```python fenced code block, no explanation."

        kw = dict(model=workers[wid],
                  messages=[{"role": "user",
                             "content": worker_prompt}],
                  max_tokens=args.max_tokens, temperature=0.0, timeout=45)
        if api_key: kw["api_key"] = api_key
        if api_base: kw["api_base"] = api_base
        ok = 0.0
        for attempt in range(5):
            try:
                out = litellm.completion(**kw).choices[0].message.content or ""
                candidate = extract_completion(out)

                # Grading (the reward)
                full_code = prompt + "\n" + candidate + "\n" + test + f"\ncheck({entry_point})\n"
                tmp_file = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8")
                tmp_path = tmp_file.name
                try:
                    tmp_file.write(full_code)
                    tmp_file.close()
                    res = subprocess.run([sys.executable, tmp_path], capture_output=True, timeout=10)
                    ok = 1.0 if res.returncode == 0 else 0.0
                except subprocess.TimeoutExpired:
                    ok = 0.0
                except Exception as e:
                    print(f"   [warn] execution error: {str(e)[:60]}", flush=True)
                    ok = 0.0
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception:
                        pass
                break
            except litellm.RateLimitError:
                wait = 2 ** attempt
                print(f"   [rate-limit] worker {wid} 429, retry {attempt+1}/5 in {wait}s", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"   [warn] worker {wid} call failed: {str(e)[:60]}", flush=True)
                break
        solve_cache[key] = ok
        print(f"   [call] worker {wid} task done -> {'OK' if ok else 'miss'} (cache={len(solve_cache)})", flush=True)
        return ok

    def fitness(head_vec):
        tot = 0.0
        for task, feat in zip(tasks, feats):
            wid = route(head_vec, feat, n_workers)
            tot += worker_solves(wid, task)
        return tot / len(tasks)

    # baseline: each worker alone + random (uses the same cache -> cheap)
    rng = np.random.default_rng(args.seed)
    per_worker = []
    for w in range(n_workers):
        per_worker.append(np.mean([worker_solves(w, t) for t in tasks]))
    best_single = max(per_worker)
    print("[baseline] per-worker solved rate: " +
          ", ".join(f"{workers[w].split('/')[-1]}={per_worker[w]:.2f}" for w in range(n_workers)), flush=True)

    # sep-CMA-ES over the head
    dim = n_workers * HIDDEN
    es = cma.CMAEvolutionStrategy(np.zeros(dim), args.sigma0,
                                  {"seed": args.seed, "verbose": -9, "CMA_diagonal": True})
    best_vec, best_fit = None, -1.0
    for it in range(args.iters):
        cands = es.ask()
        fits = [fitness(c) for c in cands]
        es.tell(cands, [-f for f in fits])
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best_fit, best_vec = fits[i], cands[i].copy()
        print(f"[iter {it}] best_solved={best_fit:.3f}  "
              f"(best single worker {best_single:.3f})  cache={len(solve_cache)}", flush=True)

    np.save(args.out, best_vec)
    print(f"\n[result] coordinator solved {best_fit:.3f} vs best single worker {best_single:.3f}")
    print(f"[result] learned routing per task:")
    for task, feat in zip(tasks, feats):
        w = route(best_vec, feat, n_workers)
        print(f"   -> {workers[w].split('/')[-1]:24s} | {task['prompt'][:50]}")
    if best_fit >= best_single:
        print("PASS — sep-CMA-ES self-trained TRINITY on REAL HumanEval, "
              "coordinator >= best single worker")


if __name__ == "__main__":
    main()
