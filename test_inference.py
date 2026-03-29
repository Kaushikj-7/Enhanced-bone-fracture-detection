import torch
import os
from models.micro_hybrid import MicroHybridModel
from utils.preprocessing import get_transforms
from PIL import Image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "outputs/micro_35min_run/micro/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return

    print("Loading Micro Hybrid Model...")
    model = MicroHybridModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transforms(mode="val")
    path_groups = [
        "data/MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png",
        "data/MURA-v1.1/valid/XR_WRIST/patient11186/study1_positive/image1.png",
        "data/MURA-v1.1/valid/XR_WRIST/patient11199/study2_negative/image1.png",
        "data/MURA-v1.1/valid/XR_WRIST/patient11212/study2_negative/image1.png",
    ]

    print("\n--- INFERENCE RESULTS ---")
    for p in path_groups:
        # replace posix slashes with os.sep for windows
        p_os = os.path.join(*p.split("/"))
        if os.path.exists(p_os):
            image = Image.open(p_os).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                prob = torch.sigmoid(out).item()

            true_label = (
                "FRACTURE (Positive)" if "positive" in p else "NORMAL (Negative)"
            )
            pred_label = "FRACTURE" if prob > 0.5 else "NORMAL"

            print(f"\nImage: {os.path.basename(p)}")
            print(f"Ground Truth : {true_label}")
            print(f"Prediction   : {pred_label} (Confidence: {prob:.4f})")

            # Indicate correct / incorrect
            is_correct = pred_label in true_label
            print(f"Status       : {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        else:
            print(f"File not found: {p_os}")


if __name__ == "__main__":
    main()
