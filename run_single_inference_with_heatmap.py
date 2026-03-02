import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from models.hybrid_model import HybridModel
from utils.preprocessing import get_transforms
from utils.gradcam import GradCAM, overlay_heatmap

def run_single_gradcam(image_path, model_path="outputs/feasibility_test/best_model.pth", output_path="outputs/single_inference_heatmap.png"):
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load model
    # Architecture must match the feasibility test (ResNet18 + ViT-Tiny)
    model = HybridModel(cnn_backbone="resnet18", vit_model="vit_tiny_patch16_224", pretrained=False)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Preprocess image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    transform = get_transforms(mode="val")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 4. Setup Grad-CAM
    # Target the Attention layer to see the final FUSED decision focus
    target_layer = model.attention
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # 5. Generate Heatmap
    print(f"Processing image: {image_path}")
    try:
        heatmap = grad_cam.generate_cam(input_tensor, target_class=0)
        
        # Denormalize image for visualization
        img_np = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Save original temporarily for overlay function
        temp_orig = "temp_inference_orig.png"
        plt.imsave(temp_orig, img_np)
        
        # Overlay heatmap
        overlay = overlay_heatmap(temp_orig, heatmap)
        
        # Save final result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        print(f"Success! Prediction heatmap saved to: {output_path}")

        # Clean up
        if os.path.exists(temp_orig):
            os.remove(temp_orig)

        # Print confidence
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            print(f"Fracture Probability: {prob:.4f}")
            print(f"Classification: {'FRACTURE' if prob > 0.5 else 'NORMAL'}")

    except Exception as e:
        print(f"Failed to generate heatmap: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    img_path = r"C:\Users\Kaushik j\OneDrive\Documents\ipcv_project\data\MURA-v1.1\train\XR_ELBOW\patient00196\study1_positive\image3.png"
    run_single_gradcam(img_path)
