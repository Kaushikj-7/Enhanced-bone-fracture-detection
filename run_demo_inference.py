import torch
import os
import cv2
import pandas as pd
import numpy as np
from models.micro_hybrid import MicroHybridModel
from src.dataset import get_transforms
from PIL import Image
from tqdm import tqdm
from utils.gradcam import GradCAM, overlay_heatmap

def run_demo_inference(model_path, demo_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for demo inference.")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Load Model
    model = MicroHybridModel(num_classes=1, pretrained=False)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Preprocessing
    # Use the test/val transforms which include our Advanced Preprocessing
    transform = get_transforms(split="test")

    # Initialize GradCAM
    # MicroHybridModel has a helper for this
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)

    # I/O Directories
    images_dir = os.path.join(demo_dir, 'images')
    results_dir = os.path.join(demo_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    ground_truth_df = pd.read_csv(os.path.join(demo_dir, 'ground_truth.csv'))
    
    demo_results = []

    print(f"Processing {len(ground_truth_df)} images for demo...")

    for _, row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df)):
        img_name = row['filename']
        true_label = row['label']
        img_path = os.path.join(images_dir, img_name)

        # 1. Prediction
        img_pil = Image.open(img_path).convert("RGB")
        # Apply transform
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()

        pred_idx = 1 if prob > 0.5 else 0
        pred_label = "fracture" if pred_idx == 1 else "normal"
        status = "CORRECT" if pred_label == true_label else "INCORRECT"

        # 2. GradCAM
        # For GradCAM we need to do a forward pass with gradients enabled
        # Reset hooks or just use a new instance if needed, but GradCAM class handles it
        cam = grad_cam.generate_cam(input_tensor)
        
        # Load original image for overlay
        # We need to resize it to 224x224 as the model sees that
        orig_img = cv2.imread(img_path)
        orig_img_resized = cv2.resize(orig_img, (224, 224))
        
        heatmap_overlay = overlay_heatmap(orig_img_resized, cam)

        # Add text overlay
        color = (0, 255, 0) if status == "CORRECT" else (0, 0, 255)
        cv2.putText(heatmap_overlay, f"Pred: {pred_label.upper()} ({prob:.2f})", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(heatmap_overlay, f"True: {true_label.upper()}", (5, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(heatmap_overlay, status, (5, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save result
        res_name = f"res_{img_name}"
        cv2.imwrite(os.path.join(results_dir, res_name), heatmap_overlay)

        demo_results.append({
            'filename': img_name,
            'true_label': true_label,
            'pred_label': pred_label,
            'probability': prob,
            'status': status,
            'result_image': res_name
        })

    # Save summary
    results_df = pd.DataFrame(demo_results)
    results_df.to_csv(os.path.join(demo_dir, 'demo_results.csv'), index=False)
    
    accuracy = (results_df['status'] == 'CORRECT').mean() * 100
    print(f"\nDemo Inference Complete! Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {results_dir}")
    print(f"Summary report: {os.path.join(demo_dir, 'demo_results.csv')}")

if __name__ == "__main__":
    MODEL_PATH = r"outputs\micro_35min_run\micro\best_model.pth"
    DEMO_DIR = "demo_set"
    run_demo_inference(MODEL_PATH, DEMO_DIR, DEMO_DIR)
