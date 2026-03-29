import torch
import os

def check_keys():
    model_path = "trained_models/outputs/plan_fast_compare/hybrid/best_model.pth"
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    keys = list(state_dict.keys())
    print("Found", len(keys), "keys.")
    print("Top-level keys sample:")
    for k in keys[:20]:
        print(k)
    print("...")
    for k in keys[-20:]:
        print(k)

if __name__ == "__main__":
    check_keys()
