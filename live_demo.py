import torch
import cv2
import sys
import os
import numpy as np
from models.micro_hybrid import MicroHybridModel
from src.dataset import get_transforms
from PIL import Image
from utils.gradcam import GradCAM, overlay_heatmap

def run_live_inference(image_name):
    # Set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"outputs\micro_35min_run\micro\best_model.pth"
    
    # Load model
    model = MicroHybridModel(num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing
    transform = get_transforms(split="test")
    
    # GradCAM
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)

    # Input image
    img_path = os.path.join('presentation_demo/source', image_name)
    if not os.path.exists(img_path):
        print(f"Image {image_name} not found!")
        return

    # 1. Prepare image
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 2. Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    is_fracture = prob > 0.5
    label = "FRACTURE DETECTED" if is_fracture else "NORMAL / NO FRACTURE"
    confidence = prob if is_fracture else (1-prob)
    color = (0, 0, 255) if is_fracture else (0, 255, 0) # BGR: Red for fracture, Green for normal

    # 3. GradCAM
    cam = grad_cam.generate_cam(input_tensor)
    
    # Load and resize original for display
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, (400, 400)) # Larger for judges
    
    heatmap_overlay = overlay_heatmap(orig_img, cam)

    # Combine side-by-side
    combined = np.hstack((orig_img, heatmap_overlay))

    # Add big labels
    cv2.rectangle(combined, (10, 10), (790, 80), (0, 0, 0), -1)
    cv2.putText(combined, f"STATUS: {label}", (20, 45), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    cv2.putText(combined, f"CONFIDENCE: {confidence*100:.1f}%", (20, 75), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # 4. Save for record
    out_name = f"live_demo_output_{image_name}"
    cv2.imwrite(out_name, combined)
    
    print("\n" + "="*40)
    print(f" IMAGE: {image_name}")
    print(f" RESULT: {label}")
    print(f" CONFIDENCE: {confidence*100:.1f}%")
    print(f" Output saved to: {out_name}")
    print("="*40)
    
    # Optional: try to show it (will fail if no display)
    # cv2.imshow("Live Model Inference", combined)
    # cv2.waitKey(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python live_demo.py <image_name>")
    else:
        run_live_inference(sys.argv[1])
