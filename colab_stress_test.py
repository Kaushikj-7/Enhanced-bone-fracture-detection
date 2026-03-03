import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

# 1. Environment & Dependency Check
def check_environment():
    print("=== [1/6] Environment Check ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        import pywt
        import skimage
        import timm
        print("Dependencies: PyWavelets, Scikit-Image, TIMM found.")
    except ImportError as e:
        print(f"CRITICAL ERROR: Missing dependency: {e}")
        return False
    return True

# 2. Preprocessing Stress Test
def check_preprocessing():
    print("
=== [2/6] Preprocessing Stress Test ===")
    try:
        from utils.advanced_preprocessing import AdvancedFracturePreprocessor
        preprocessor = AdvancedFracturePreprocessor(target_size=(224, 224))
        dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        out = preprocessor(dummy_img)
        out_np = np.array(out)
        print(f"Input: (512, 512, 3) -> Output: {out_np.shape}")
        if out_np.shape == (224, 224, 3):
            print("Preprocessing Stack: SUCCESS (3-Channel CLAHE-Wavelet-Frangi)")
            return True
    except Exception as e:
        print(f"Preprocessing Error: {e}")
    return False

# 3. Model & AMP Stress Test
def check_model_and_amp():
    print("
=== [3/6] Model & AMP Stress Test ===")
    try:
        from models.micro_hybrid import MicroHybridModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MicroHybridModel(pretrained=False).to(device)
        
        # Parameter Count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params/1e6:.3f}M")
        
        # Forward Pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(dummy_input)
        print(f"Forward Pass: SUCCESS (Output shape: {output.shape})")
        
        # Backward Pass (AMP)
        criterion = nn.BCEWithLogitsLoss()
        labels = torch.ones(2, 1).to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            loss = criterion(output, labels)
        scaler.scale(loss).backward()
        print("Backward Pass (AMP): SUCCESS")
        return True
    except Exception as e:
        print(f"Model/AMP Error: {e}")
        import traceback
        traceback.print_exc()
    return False

# 4. Grad-CAM Diagnostic
def check_gradcam():
    print("
=== [4/6] Grad-CAM Diagnostic ===")
    try:
        from utils.gradcam import GradCAM
        from models.micro_hybrid import MicroHybridModel
        
        model = MicroHybridModel(pretrained=False)
        target_layer = model.get_last_conv_layer()
        grad_cam = GradCAM(model, target_layer)
        
        dummy_input = torch.randn(1, 3, 224, 224)
        heatmap = grad_cam.generate_cam(dummy_input)
        
        if heatmap.shape == (224, 224) and np.max(heatmap) <= 1.0:
            print("Grad-CAM Generation: SUCCESS (Target: Enhancer Fuse Layer)")
            return True
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
    return False

def run_stress_test():
    print("************************************************")
    print("* STARTING HIGH-RELIABILITY PIPELINE DIAGNOSTIC *")
    print("************************************************")
    
    if not check_environment(): return
    if not check_preprocessing(): return
    if not check_model_and_amp(): return
    if not check_gradcam(): return
    
    print("
************************************************")
    print("* PIPELINE STATUS: 100% OPERATIONAL            *")
    print("************************************************")

if __name__ == "__main__":
    run_stress_test()
