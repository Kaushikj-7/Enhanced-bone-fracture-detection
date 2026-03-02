import torch
from PIL import Image
from models.hybrid_model import HybridModel
from utils.preprocessing import get_transforms
import os

def run_inference(image_path, model_path="outputs/feasibility_test/best_model.pth"):
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the model architecture (Must match the one used during training)
    # Using vit_tiny as specified in our optimized production plan
    model = HybridModel(cnn_backbone="resnet18", vit_model="vit_tiny_patch16_224", pretrained=False)
    
    # 3. Load weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Preprocess the image
    transform = get_transforms(mode="val")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 5. Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = "FRACTURE" if probability > 0.5 else "NORMAL"

    print(f"
Results for: {os.path.basename(image_path)}")
    print(f"Prediction: {prediction}")
    print(f"Confidence (Probability): {probability:.4f}")
    
    return prediction, probability

if __name__ == "__main__":
    # Update this with a path to one of your local MURA images
    sample_img = "data/MURA-v1.1/valid/XR_ELBOW/patient11186/study1_positive/image1.png"
    if os.path.exists(sample_img):
        run_inference(sample_img)
    else:
        print(f"Sample image not found at {sample_img}. Please update the script with a valid path.")
