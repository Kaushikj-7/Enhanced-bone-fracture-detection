import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders
from models.hybrid_model import HybridModel
from training.train import train_pipeline
from utils.gradcam import GradCAM, overlay_heatmap

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Feasibility + Heatmaps on {device}")

    # Use a small batch for CPU speed
    batch_size = 4
    data_dir = "data"
    output_dir = "outputs/feasibility_test"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    dataloaders, _ = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    
    # Mock small loaders
    class SubLoader:
        def __init__(self, loader, steps=5):
            self.loader = loader
            self.steps = steps
            self.dataset = loader.dataset
        def __iter__(self):
            count = 0
            for batch in self.loader:
                yield batch
                count += 1
                if count >= self.steps: break
        def __len__(self):
            return self.steps

    train_loader = SubLoader(dataloaders['train'], steps=5)
    val_loader = SubLoader(dataloaders['val'], steps=2)

    # Model
    model = HybridModel(cnn_backbone="resnet18", vit_model="vit_tiny_patch16_224", pretrained=True).to(device)

    config = {
        "training": {
            "learning_rate": 0.001,
            "num_epochs": 1,
            "patience": 2
        }
    }

    print("Running 1 Sample Epoch...")
    model, history = train_pipeline(model, train_loader, val_loader, config, device, output_dir)
    
    # --- HEATMAP GENERATION ---
    print("\nGenerating Sample Heatmap (Grad-CAM)...")
    model.eval()
    
    # Get one image from validation
    inputs, labels = next(iter(dataloaders['val']))
    input_tensor = inputs[0:1].to(device) # Single image [1, 3, 224, 224]
    input_tensor.requires_grad = True
    
    # Target layer for ResNet18 based Sequential backbone
    try:
        target_layer = model.cnn_branch.backbone[7][-1]
        grad_cam = GradCAM(model=model, target_layer=target_layer)
        
        heatmap = grad_cam.generate_cam(input_tensor, target_class=0)
        print(f"Heatmap stats: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")

        # Denormalize image for visualization
        img_np = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Save original and heatmap overlay
        plt.imsave(os.path.join(output_dir, "original_sample.png"), img_np)

        # Save temp image for overlay function (cv2 needs file on disk or specific format)
        temp_path = os.path.join(output_dir, "temp_cam_input.png")
        plt.imsave(temp_path, img_np)

        # Overlay
        import cv2
        cam_overlay = overlay_heatmap(temp_path, heatmap)

        # Save final result
        save_path = os.path.join(output_dir, "sample_heatmap.png")
        cv2.imwrite(save_path, cam_overlay)
        print(f"Success! Heatmap saved to: {save_path}")

        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Feasibility Result ---")
    print(f"Final Train Acc: {history['train_acc'][-1]:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()
