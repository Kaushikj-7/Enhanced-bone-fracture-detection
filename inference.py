import torch
from PIL import Image
from models.hybrid_model import HybridModel
from utils.preprocessing import get_transforms
import os

from utils.gpu_preprocessing import GPUFracturePreprocessor

from utils.gradcam import GradCAM, overlay_heatmap
import cv2
import numpy as np

def run_inference(image_path, model_path="outputs/production_run/hybrid/best_model.pth", output_heatmap="outputs/latest_inference_heatmap.png", simple_pre=False):
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the model architecture (ResNet18 + ViT Base + Interpretability Fusion)
    model = HybridModel(
        cnn_backbone="resnet18", 
        vit_model="vit_base_patch16_224", 
        pretrained=False
    )
    
    # 3. Load weights (Partial loading for new architecture compatibility)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} tensors. (Fusion/Head may be reset)")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # 4. Preprocessors
    transform = get_transforms(mode="val")
    gpu_pre = None
    if device.type == "cuda":
        gpu_pre = GPUFracturePreprocessor(device=device, simple_pre=simple_pre).to(device)

    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 5. Predict & Generate Heatmap
        # We enable gradients specifically for Grad-CAM
        input_tensor.requires_grad = True
        
        # Initialize GradCAM targeting the new fusion layer
        grad_cam = GradCAM(model=model, target_layer=model.get_last_conv_layer())
        
        if gpu_pre is not None:
            # We don't want to track gradients through the preprocessor for the heatmap
            # as we want the heatmap relative to the preprocessed features
            with torch.no_grad():
                processed_input = gpu_pre(input_tensor)
        else:
            processed_input = input_tensor

        # Forward pass
        logits = model(processed_input)
        probability = torch.sigmoid(logits).item()
        prediction = "FRACTURE" if probability > 0.5 else "NORMAL"

        # 6. Conditional Heatmap Generation (Compute Optimization)
        # We only run the expensive Grad-CAM backward pass if a fracture is predicted
        if probability > 0.5:
            print("Fracture detected. Generating diagnostic heatmap...")
            heatmap = grad_cam.generate_cam(processed_input)
            
            # Save Overlay
            img_np = np.array(image.resize((224, 224)))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            overlay = overlay_heatmap(img_bgr, heatmap)
            cv2.imwrite(output_heatmap, overlay)
            heatmap_status = f"Saved to {output_heatmap}"
        else:
            heatmap_status = "Skipped (Normal prediction)"

        print(f"\n--- Inference Report ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {probability:.4f}")
        print(f"Heatmap: {heatmap_status}")
        
        return prediction, probability
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

if __name__ == "__main__":
    # Update this with a path to one of your local MURA images
    sample_img = "data/MURA-v1.1/valid/XR_ELBOW/patient11186/study1_positive/image1.png"
    if os.path.exists(sample_img):
        run_inference(sample_img)
    else:
        print(f"Sample image not found at {sample_img}. Please update the script with a valid path.")
