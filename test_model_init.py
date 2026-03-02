import torch
from models.hybrid_model import HybridModel

def test_model():
    print("Initializing HybridModel...")
    model = HybridModel(cnn_backbone="resnet18", vit_model="vit_base_patch16_224", num_classes=1, pretrained=False)
    model.eval()
    
    # Create dummy input [Batch, Channels, Height, Width]
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Performing forward pass...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        assert output.shape == (2, 1), f"Unexpected output shape: {output.shape}"
        print("Model verification PASSED.")
    except Exception as e:
        print(f"Model verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
