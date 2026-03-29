import torch
from PIL import Image
from models.hybrid_model import HybridModel
from utils.preprocessing import get_transforms
import os

from utils.gpu_preprocessing import GPUFracturePreprocessor

def run_inference(image_path, model_path="outputs/production_run/hybrid/best_model.pth", simple_pre=False):
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the model architecture (Bigger Model: ResNet18 + ViT Base)
    model = HybridModel(
        cnn_backbone="resnet18", 
        vit_model="vit_base_patch16_224", 
        pretrained=False
    )
    
    # 3. Load weights (with partial loading for the new enhancer block)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    # Partial loading for new block compatibility
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # 4. Preprocessors
    # CPU: Basic Sanitization + Crop + Resize (from get_transforms)
    # GPU: Final Feature Isolation (Wavelet/Frangi/Norm)
    transform = get_transforms(mode="val")
    gpu_pre = None
    if device.type == "cuda":
        gpu_pre = GPUFracturePreprocessor(device=device, simple_pre=simple_pre).to(device)

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 5. Predict
        with torch.no_grad():
            if gpu_pre is not None:
                input_tensor = gpu_pre(input_tensor)
            
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = "FRACTURE" if probability > 0.5 else "NORMAL"

        print(f"\nResults for: {os.path.basename(image_path)}")
        print(f"Prediction: {prediction}")
        print(f"Confidence (Probability): {probability:.4f}")
        
        return prediction, probability
    except Exception as e:
        print(f"Error during inference for {image_path}: {e}")
        return None, 0.0

if __name__ == "__main__":
    # Update this with a path to one of your local MURA images
    sample_img = "data/MURA-v1.1/valid/XR_ELBOW/patient11186/study1_positive/image1.png"
    if os.path.exists(sample_img):
        run_inference(sample_img)
    else:
        print(f"Sample image not found at {sample_img}. Please update the script with a valid path.")
