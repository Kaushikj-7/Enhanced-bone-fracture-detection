import torch
from models.baselines import CNNOnlyModel, ViTOnlyModel
from models.hybrid_model import HybridModel
from models.micro_hybrid import MicroHybridModel

def check_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    # Try to see keys to guess architecture
    keys = list(checkpoint.keys())
    print(f"Total keys: {len(keys)}")
    
    # Heuristic for identification
    if any("cnn_branch.backbone" in k for k in keys) and any("vit_branch.vit" in k for k in keys):
        print("Detected: HybridModel")
    elif any("cnn_branch" in k for k in keys) and not any("vit_branch" in k for k in keys):
        print("Detected: CNNOnlyModel")
    elif any("vit_branch" in k for k in keys) and not any("cnn_branch" in k for k in keys):
        print("Detected: ViTOnlyModel")
    elif any("enhancer" in k for k in keys) and any("cnn_branch" in k for k in keys):
        print("Detected: MicroHybridModel")
    else:
        # Check specific layers
        if any("backbone" in k for k in keys) and len(keys) > 50:
            print("Detected: Likely CNN Branch or Hybrid")
        else:
            print("Could not definitively identify architecture.")
            print("First 10 keys:", keys[:10])

if __name__ == "__main__":
    check_model("outputs/feasibility_test/best_model.pth")
