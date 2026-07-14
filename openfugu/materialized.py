import os
import json
import torch
from safetensors.torch import load_file

def load_materialized(checkpoint_dir: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Loads materialized weights and router head from the specified checkpoint directory.
    Maps Bumblebee names -> HF Qwen3 PyTorch module paths.
    Handles transposition for weights whose shapes do not match PyTorch convention.
    """
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # 1. Load the 9 selected tensors
    weights = {}
    selected_tensors = manifest.get("selected_tensors", [])
    
    expected_shapes = {
        "model.embed_tokens.weight": (151936, 1024),
        "model.layers.26.self_attn.q_proj.weight": (2048, 1024),
        "model.layers.26.self_attn.k_proj.weight": (1024, 1024),
        "model.layers.26.self_attn.v_proj.weight": (1024, 1024),
        "model.layers.26.self_attn.o_proj.weight": (1024, 2048),
        "model.layers.26.mlp.gate_proj.weight": (3072, 1024),
        "model.layers.26.mlp.up_proj.weight": (3072, 1024),
        "model.layers.26.mlp.down_proj.weight": (1024, 3072),
        "lm_head.weight": (151936, 1024),
    }

    for entry in selected_tensors:
        chk_path = entry["checkpoint_path"]
        abs_chk_path = os.path.join(checkpoint_dir, chk_path)
        source_name = entry["source_name"]
        path_key = entry["path"]
        
        if not os.path.exists(abs_chk_path):
            raise FileNotFoundError(f"Checkpoint file not found: {abs_chk_path}")
            
        tensors_dict = load_file(abs_chk_path)
        if path_key not in tensors_dict:
            # fallback to checking if any key in tensors_dict contains the name
            matching_keys = [k for k in tensors_dict if path_key in k]
            if matching_keys:
                tensor = tensors_dict[matching_keys[0]]
            else:
                raise KeyError(
                    f"Tensor key {path_key!r} for {source_name!r} not found in "
                    f"{abs_chk_path} (available keys: {sorted(tensors_dict)})"
                )
        else:
            tensor = tensors_dict[path_key]

        if source_name not in expected_shapes:
            raise ValueError(f"No expected shape registered for tensor {source_name!r}")
        expected_shape = expected_shapes[source_name]
        if source_name in [
            "model.layers.26.self_attn.k_proj.weight",
            "model.layers.26.self_attn.v_proj.weight",
        ]:
            tensor = tensor.T.contiguous()
        elif tensor.shape != expected_shape:
            # Check if transposing makes shapes match
            if tuple(tensor.shape[::-1]) == expected_shape:
                tensor = tensor.T.contiguous()
            else:
                raise ValueError(
                    f"Tensor {source_name} shape {tensor.shape} does not match expected shape {expected_shape}"
                )
                
        weights[source_name] = tensor

    # 2. Load the router head
    router_head_path = os.path.join(checkpoint_dir, "router_head.safetensors")
    if not os.path.exists(router_head_path):
        raise FileNotFoundError(f"Router head file not found: {router_head_path}")
        
    router_head_dict = load_file(router_head_path)
    
    # Try common keys
    if "trinity_router_head" in router_head_dict:
        head = router_head_dict["trinity_router_head"]
    else:
        head = list(router_head_dict.values())[0]

    # Expected shape is (10, 1024)
    if head.shape != (10, 1024):
        if tuple(head.shape[::-1]) == (10, 1024):
            head = head.T.contiguous()
        else:
            raise ValueError(f"Router head shape {head.shape} is not (10, 1024)")

    return weights, head
