import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_dataloaders
from models.baselines import CNNOnlyModel, ViTOnlyModel
from models.hybrid_model import HybridModel

# Try to import GradCAM from utils, handle if missing
try:
    from utils.gradcam import GradCAM, overlay_heatmap

    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: utils.gradcam not found. Skipping GradCAM.")


def _build_model(name: str, cnn_backbone: str, vit_backbone: str, pretrained: bool):
    if name == "cnn":
        return CNNOnlyModel(cnn_backbone=cnn_backbone, pretrained=pretrained)
    if name == "vit":
        return ViTOnlyModel(vit_model=vit_backbone, pretrained=pretrained)
    if name == "hybrid":
        return HybridModel(
            cnn_backbone=cnn_backbone,
            vit_model=vit_backbone,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown experiment: {name}")


def load_checkpoint(model, path):
    if not os.path.exists(path):
        print(f"Warning: Checkpoint not found at {path}")
        return model
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def evaluate_model(model, loader, device, output_dir, exp_name):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Evaluating {exp_name}...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Eval {exp_name}"):
            inputs = inputs.to(device)
            # Forward
            outputs = model(inputs)
            # Softmax
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Save raw predictions
    df = pd.DataFrame({"label": all_labels, "prob": all_preds})
    df.to_csv(os.path.join(output_dir, f"{exp_name}_predictions.csv"), index=False)

    # Check if we have enough distinct labels for ROC
    if len(np.unique(all_labels)) < 2:
        print(f"Skipping ROC for {exp_name}: only one class present in val set.")
        return all_preds, all_labels

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {exp_name} (AUC={roc_auc:.2f})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{exp_name}_roc.png"))
    plt.close()

    # Confusion Matrix (Threshold 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds_binary)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {exp_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_dir, f"{exp_name}_cm.png"))
    plt.close()

    # Classification Report
    cr = classification_report(
        all_labels, preds_binary, output_dict=True, zero_division=0
    )
    with open(os.path.join(output_dir, f"{exp_name}_metrics.json"), "w") as f:
        json.dump(cr, f, indent=4)

    return all_preds, all_labels


def get_target_layer(model, exp_name):
    # Determine the target layer for GradCAM based on model type
    try:
        if exp_name == "cnn":
            # ResNet18 typical last layer in our Sequential backbone is index 7
            return model.cnn_branch.backbone[7][-1]
        elif exp_name == "hybrid":
            # HybridModel uses 'cnn_branch' with same ResNet structure
            return model.cnn_branch.backbone[7][-1]
        elif exp_name == "vit":
            # ViT GradCAM is hard with standard hooks, skipping for now
            return None
    except Exception as e:
        print(f"Could not resolve target layer for {exp_name}: {e}")
        return None
    return None


def run_gradcam(model, loader, device, output_dir, exp_name, num_images=5):
    if not GRADCAM_AVAILABLE:
        return

    target_layer = get_target_layer(model, exp_name)
    if target_layer is None:
        print(f"Skipping GradCAM for {exp_name} (no suitable target layer found).")
        return

    print(f"Generating GradCAM for {exp_name}...")
    cam_dir = os.path.join(output_dir, f"{exp_name}_gradcam")
    os.makedirs(cam_dir, exist_ok=True)

    # Initialize GradCAM
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    model.eval()

    # Process one batch, pick first N images
    try:
        count = 0
        for inputs, labels in loader:
            if count >= num_images:
                break

            inputs = inputs.to(device)
            # Process single images to avoid dimension mismatch in utils.gradcam
            for i in range(inputs.size(0)):
                if count >= num_images:
                    break

                input_tensor = inputs[i].unsqueeze(0)  # [1, C, H, W]
                # Enable grad for input/model just for CAM generation
                # But typically GradCAM needs gradients during backward

                # utils.gradcam.generate_cam does forward+backward
                # Ensure gradients are enabled
                with torch.set_grad_enabled(True):
                    # For custom GradCAM, input must require grad?
                    input_tensor.requires_grad = True
                    heatmap = grad_cam.generate_cam(
                        input_tensor, target_class=1
                    )  # Target 'positive' class

                # Prepare image for overlay
                img_np = inputs[i].cpu().detach().permute(1, 2, 0).numpy()
                # Denormalize roughly for visualization (assuming ImageNet norms)
                # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                # Save original temp file for overlay function
                orig_path = os.path.join(cam_dir, f"temp_orig_{count}.png")
                plt.imsave(orig_path, img_np)

                # Use overlay_heatmap
                import cv2

                overlay = overlay_heatmap(orig_path, heatmap)

                # Save final visualization together
                save_path = os.path.join(cam_dir, f"cam_{count}_label_{labels[i]}.png")
                cv2.imwrite(save_path, overlay)

                # cleanup temp
                if os.path.exists(orig_path):
                    os.remove(orig_path)

                count += 1

    except Exception as e:
        print(f"Error generating GradCAM for {exp_name}: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiments", default="cnn,vit,hybrid")
    parser.add_argument("--cnn_backbone", default="resnet18")
    parser.add_argument("--vit_backbone", default="vit_base_patch16_224")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gradcam", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = args.experiments.split(",")

    # Load Validation Data
    dataloaders, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if "val" not in dataloaders:
        print("No validation set found. Cannot finalize artifacts.")
        return

    val_loader = dataloaders["val"]

    for exp in experiments:
        exp_dir = os.path.join(args.output_dir, exp)
        model_path = os.path.join(exp_dir, "best_model.pth")

        # Look for model
        if not os.path.exists(model_path):
            # Try last_model.pth
            model_path = os.path.join(exp_dir, "last_model.pth")

        if not os.path.exists(model_path):
            print(f"Skipping {exp}: No model file found.")
            continue

        print(f"\nProcessing {exp} artifacts using {model_path}...")
        model = _build_model(
            exp, args.cnn_backbone, args.vit_backbone, pretrained=False
        )
        model = load_checkpoint(model, model_path)
        model = model.to(device)

        # Evaluation
        evaluate_model(
            model, val_loader, device, exp_dir, exp
        )  # Pass exp_dir not args.output_dir so files go INTO exp folder

        # GradCAM
        if args.num_gradcam > 0:
            run_gradcam(
                model,
                val_loader,
                device,
                args.output_dir,
                exp,
                num_images=args.num_gradcam,
            )  # args.output_dir here creates exp_gradcam folder?
            # Actually, let's put gradcam inside the exp folder too or root output?
            # Current logic: `cam_dir = os.path.join(output_dir, f'{exp_name}_gradcam')`
            # This puts `cnn_gradcam`, `vit_gradcam` in `outputs/`. That's fine.

    print("Done finalizing artifacts.")


if __name__ == "__main__":
    main()
