from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.micro_hybrid import MicroHybridModel
from utils.preprocessing import get_transforms

app = FastAPI()

# Model configuration (constraint-friendly Cloud Run default)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "trained_models/outputs/micro_pipeline_run/micro/best_model.pth",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading model from {MODEL_PATH}...")
    model = MicroHybridModel(num_classes=1, pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: Model weights not found at {MODEL_PATH}")
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Preprocess
    transform = get_transforms(mode="val")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
    
    prediction = "FRACTURE" if prob > 0.5 else "NORMAL"
    
    return {
        "prediction": prediction,
        "probability": prob,
        "status": "success"
    }

@app.get("/")
def read_root():
    return {"message": "Bone Fracture Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
