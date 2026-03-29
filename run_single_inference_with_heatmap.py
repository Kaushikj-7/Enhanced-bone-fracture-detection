import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from models.hybrid_model import HybridModel
from models.micro_hybrid import MicroHybridModel
from utils.preprocessing import get_transforms
from utils.gradcam import GradCAM, overlay_heatmap


def run_single_gradcam(
    image_path,
    model_type="micro",
    model_path=None,
    output_path="outputs/single_inference_heatmap.png",
):
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load model
    if model_type == "micro":
        model = MicroHybridModel(pretrained=True)
        if model_path is None:
            # For testing without a trained checkpoint, we'll use pretrained weights
            print("Using pretrained MicroHybrid weights for structural verification.")
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = HybridModel(
            cnn_backbone="resnet18", vit_model="vit_tiny_patch16_224", pretrained=False
        )
        if model_path:
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
    # Dynamically resolve target layer
    if hasattr(model, "get_last_conv_layer"):
        target_layer = model.get_last_conv_layer()
    else:
        # Fallback for standard HybridModel
        target_layer = model.cnn_branch.backbone[7][-1]

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

        # IMPORTANT: img_np is RGB. OpenCV works with BGR.
        # Convert to BGR for overlay and final save
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Overlay heatmap
        # Pass the BGR image directly to overlay_heatmap
        overlay = overlay_heatmap(img_bgr, heatmap)

        # Save final result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        print(f"Success! Prediction heatmap saved to: {output_path}")

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
    img_path = r"data\MURA-v1.1\valid\XR_WRIST\patient11185\study1_positive\image1.png"
    model_path = r"outputs\micro_35min_run\micro\best_model.pth"
    run_single_gradcam(
        img_path,
        model_type="micro",
        model_path=model_path,
        output_path="outputs/visual_test_inference.png",
    )
