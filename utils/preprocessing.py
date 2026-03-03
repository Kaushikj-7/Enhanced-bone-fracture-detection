from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

try:
    from utils.advanced_preprocessing import AdvancedFracturePreprocessor
except ImportError:
    # If called from within utils/ or as a relative import fails, try to handle it.
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.advanced_preprocessing import AdvancedFracturePreprocessor

def get_transforms(mode="train", input_size=224):
    """
    Returns high-reliability image transformations (CLAHE + Wavelet + Frangi) for training and inference.
    """
    # Standard ImageNet normalization for our 3-channel feature stack
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if mode == "train":
        return transforms.Compose(
            [
                AdvancedFracturePreprocessor(target_size=(input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
                normalize,
            ]
        )
    else:
        # Validation/Test/Inference
        return transforms.Compose(
            [
                AdvancedFracturePreprocessor(target_size=(input_size, input_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


def preprocess_image(image_path, size=224):
    """
    Reads and preprocesses a single image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        transform = get_transforms(mode="val", input_size=size)
        return transform(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
