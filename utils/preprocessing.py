from torchvision import transforms
from PIL import Image


def get_transforms(mode="train", input_size=224):
    """
    Returns image transformations for training and inference.

    Training: Resize, Random Rotation, Horizontal Flip, Color Jitter
    Validation/Test: ResizeOnly
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
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
