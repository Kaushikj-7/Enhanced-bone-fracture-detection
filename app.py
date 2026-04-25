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
# Path mirrors existing training artifact layout in this repository.
DEFAULT_MODEL_PATH = "trained_models/outputs/micro_pipeline_run/micro/best_model.pth"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
STRICT_MODEL_LOAD = os.getenv("STRICT_MODEL_LOAD", "false").lower() in (
    "1",
    "true",
    "yes",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading model from {MODEL_PATH}...")
    model = MicroHybridModel(num_classes=1, pretrained=False)

    active_model_path = MODEL_PATH
    custom_path_missing = MODEL_PATH != DEFAULT_MODEL_PATH and not os.path.exists(MODEL_PATH)
    fallback_available = os.path.exists(DEFAULT_MODEL_PATH)
    if custom_path_missing and fallback_available:
        print(
            f"Warning: MODEL_PATH not found ({MODEL_PATH}). Falling back to default weights at {DEFAULT_MODEL_PATH}."
        )
        active_model_path = DEFAULT_MODEL_PATH

    if os.path.exists(active_model_path):
        checkpoint = torch.load(active_model_path, map_location=DEVICE)
        # strict mode is configurable; default is compatibility mode for historical checkpoints.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            load_result = model.load_state_dict(
                checkpoint["model_state_dict"], strict=STRICT_MODEL_LOAD
            )
        else:
            load_result = model.load_state_dict(checkpoint, strict=STRICT_MODEL_LOAD)

        missing_keys = getattr(load_result, "missing_keys", [])
        unexpected_keys = getattr(load_result, "unexpected_keys", [])
        if missing_keys or unexpected_keys:
            missing_preview = missing_keys[:5]
            unexpected_preview = unexpected_keys[:5]
            print(
                "Warning: checkpoint loaded with key mismatches "
                f"(missing={len(missing_keys)}, "
                f"unexpected={len(unexpected_keys)}). "
                f"Samples missing={missing_preview}, unexpected={unexpected_preview}"
            )
    else:
        if MODEL_PATH != DEFAULT_MODEL_PATH:
            print(
                f"Warning: MODEL_PATH env var points to a missing file: {MODEL_PATH}. "
                "No fallback checkpoint found; service will run with randomly initialized weights."
            )
        else:
            print(
                f"Warning: Default model weights not found at {MODEL_PATH}. "
                "Service will run with randomly initialized weights."
            )
    
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
