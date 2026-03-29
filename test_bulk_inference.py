import torch
import os
import cv2
import matplotlib.pyplot as plt
from models.micro_hybrid import MicroHybridModel
from utils.preprocessing import get_transforms
from PIL import Image
from tqdm import tqdm
from utils.gradcam import GradCAM, overlay_heatmap


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for bulk inference.")

    # Model paths
    model_path = r"outputs\micro_35min_run\micro\best_model.pth"
    if not os.path.exists(model_path):
        print("Model not found. Please train it first.")
        return

    # Load Model
    model = MicroHybridModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing
    transform = get_transforms(mode="val")

    # Initialize GradCAM
    # Targeting the last convolutional layer output before GAP
    target_layer = (
        model.cnn_branch[-1]
        if hasattr(model, "cnn_branch")
        else model.get_last_conv_layer()
    )
    grad_cam = GradCAM(model, target_layer)

    # I/O Directories
    input_dir = r"test_outputs\presentation_images"
    output_dir = r"test_outputs\presentation_results"
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    print(f"Found {len(images)} images to process...")

    correct = 0

    for img_name in tqdm(images, desc="Generating Visualizations"):
        img_path = os.path.join(input_dir, img_name)

        # 1. Prediction
        img_pil = Image.open(img_path).convert("RGB")
        tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            prob = torch.sigmoid(out).item()

        pred_is_fracture = prob > 0.5
        true_is_fracture = img_name.startswith("pos_")

        if pred_is_fracture == true_is_fracture:
            correct += 1
            status = "CORRECT"
        else:
            status = "INCORRECT"

        pred_label = "FRACTURE" if pred_is_fracture else "NORMAL"
        true_label = "FRACTURE" if true_is_fracture else "NORMAL"

        # 2. GradCAM Generation
        try:
            cam = grad_cam.generate(tensor)
            original_img = cv2.imread(img_path)
            original_img = cv2.resize(original_img, (224, 224))

            heatmap_overlay = overlay_heatmap(original_img, cam)

            # Add text overlay of the results onto the image
            color = (0, 255, 0) if status == "CORRECT" else (0, 0, 255)
            cv2.putText(
                heatmap_overlay,
                f"Pred: {pred_label} ({prob:.2f})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                heatmap_overlay,
                f"True: {true_label}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Save Output
            out_path = os.path.join(output_dir, f"result_{img_name}")
            cv2.imwrite(out_path, heatmap_overlay)

        except Exception as e:
            print(f"Failed GradCAM for {img_name}: {e}")

    print("\n" + "=" * 40)
    print(f"Bulk Inference Complete!")
    print(
        f"Accuracy on 100 Sample set: {correct}/{len(images)} ({correct / len(images) * 100:.1f}%)"
    )
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
