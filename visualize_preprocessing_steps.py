import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.advanced_preprocessing import AdvancedFracturePreprocessor
import os

def visualize_steps(image_path, output_path="outputs/preprocessing_check.png"):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Initialize Preprocessor
    preprocessor = AdvancedFracturePreprocessor(target_size=(224, 224))
    
    # 1. Original
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # 2. Sanitize & Balance (CLAHE) - Channel R
    balanced = preprocessor.sanitize_and_balance(img_np)
    
    # 3. Wavelet Boost (Fine Cracks) - Channel G
    wavelet = preprocessor.wavelet_detail_boost(balanced)
    
    # 4. Frangi Vesselness (Linear Ridges) - Channel B
    frangi_img = preprocessor.frangi_vesselness(balanced)
    
    # 5. Final Stack
    final_stack = preprocessor(img)
    final_np = np.array(final_stack)

    # Plotting
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 5, 1)
    plt.title("1. Original")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title("2. CLAHE (Balanced)")
    plt.imshow(balanced, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title("3. Wavelets (Crack Boost)")
    plt.imshow(wavelet, cmap='magma')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.title("4. Frangi (Linear Ridge)")
    plt.imshow(frangi_img, cmap='hot')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.title("5. Final 3-Channel Stack")
    plt.imshow(final_np)
    plt.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    # Test on a known positive elbow image
    img_path = r"data/MURA-v1.1/train/XR_ELBOW/patient00196/study1_positive/image3.png"
    visualize_steps(img_path)
