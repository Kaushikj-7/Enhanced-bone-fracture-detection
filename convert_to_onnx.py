import torch
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.hybrid_model import HybridModel

def convert():
    model_path = "trained_models/outputs/plan_fast_compare/hybrid/best_model.pth"
    output_onnx = "vit_model.onnx"
    
    # Initialize model with the same parameters as used in training
    # Based on run_full_pipeline.py defaults
    model = HybridModel(
        cnn_backbone="resnet18",
        vit_model="vit_base_patch16_224",
        pretrained=False, # We'll load weights
        use_lora=True
    )
    
    # Load weights
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")
    # Sometimes weights are wrapped in 'model_state_dict' or similar
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting model to {output_onnx}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Export complete.")

if __name__ == "__main__":
    convert()
