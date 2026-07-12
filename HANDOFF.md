# HANDOFF: TRINITY materialized-weights loader

Repo: /Users/russ/Projects/snowy2/OpenFugu (venv .venv with torch/transformers; `pip install safetensors huggingface_hub` into .venv if missing). Do NOT git commit.

Problem: openfugu/mini.py loads a 19,456-dim CMA-ES vector (env FUGU_VECTOR, model_iter_60.npy) and reconstructs 9 SVF-adapted matrices + linear router head. That .npy no longer exists upstream. HF dataset `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b` now ships MATERIALIZED weights:
- checkpoints/0001_embedder.token_embedding.kernel.safetensors ... 0009_language_modeling_head.output.kernel.safetensors (embedder; decoder.blocks.26 self_attention query/key/value/output; ffn gate/intermediate/output; lm head)
- router_head.safetensors (tensor `trinity.router_head.linear.weight`, [10,1024])
- manifest.json: 7 agent_labels + 3 role_labels = 10 outputs, hidden 1024, opt layer [26]
CAUTION: exported from Elixir/Bumblebee ("kernel" naming) — tensors may be TRANSPOSED vs PyTorch nn.Linear convention; verify by shape against the Qwen3-0.6B state dict and transpose where needed.

Steps:
1. Read openfugu/mini.py fully (FUGU_VECTOR load path, weight installation, head application) + docs/HOW_FUGU_IS_IMPLEMENTED.md (SVF + head sections).
2. Download the 11 files via huggingface_hub.hf_hub_download(repo_type="dataset") into artifacts/materialized/ (~650MB).
3. New file openfugu/materialized.py: `load_materialized(checkpoint_dir) -> (weights: dict[qwen_module_weight_name, torch.Tensor], head: torch.Tensor)`; map Bumblebee names → HF Qwen3 module paths as used by mini.py; handle dtype + transposition.
4. Minimal hook in mini.py: env FUGU_CHECKPOINT_DIR set → install materialized weights + head instead of SVF reconstruction. Smallest possible diff.
5. Verify: FUGU_MODEL=/Users/russ/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca FUGU_CHECKPOINT_DIR=$PWD/artifacts/materialized FUGU_FIXTURE=$PWD/artifacts/qwen_router_prompt_eval_cases.json .venv/bin/python -m openfugu.mini --self-test (check exact flag in mini.py). Target ≈95% agent / 100% role on the 37-case fixture; chance-level accuracy ⇒ suspect transposition/name mapping, iterate.

Report: self-test numbers + mapping/transposition decisions.
