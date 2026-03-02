import os
import shutil
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from PIL import Image

# We will use 'keremberke/bone-fracture-classification' from Hugging Face
# which is a commonly used subset/version compatible with this task.
HF_DATASET_ID = "keremberke/bone-fracture-classification"


def process_hf_dataset(target_root):
    print(f"Loading dataset {HF_DATASET_ID} from Hugging Face...")

    try:
        # Load dataset (it has train, validation, test splits usually)
        dataset = load_dataset(HF_DATASET_ID, name="full")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Dataset loaded. Keys:", dataset.keys())

    # Define target paths
    dirs = {
        "train": {
            "normal": os.path.join(target_root, "train", "normal"),
            "fracture": os.path.join(target_root, "train", "fracture"),
        },
        "validation": {
            "normal": os.path.join(target_root, "val", "normal"),
            "fracture": os.path.join(target_root, "val", "fracture"),
        },
        "test": {
            "normal": os.path.join(target_root, "test", "normal"),
            "fracture": os.path.join(target_root, "test", "fracture"),
        },
    }

    # Create directories
    for split_dirs in dirs.values():
        for d in split_dirs.values():
            os.makedirs(d, exist_ok=True)

    # Mappings
    # The dataset typically returns 'image' and 'labels' (or 'label')
    # Use features to determine label map
    # 0 -> fracture, 1 -> normal ?? Need to check features
    # Usually: 0: fracture, 1: normal OR vice versa.
    # Let's inspect features or assume standard.
    # For 'keremberke/bone-fracture-classification':
    # labels: 0 -> fracture, 1 -> normal (Common in object detection, but this is classification split)
    # Check features directly

    if "train" in dataset:
        features = dataset["train"].features
        if "labels" in features:
            label_names = features["labels"].names  # Object Detection?
        elif "label" in features:
            label_names = features["label"].names  # Classification

        print(f"Label names found: {label_names}")

    # Processing function
    def save_split(hf_split_name, target_split_name):
        if hf_split_name not in dataset:
            print(f"Split {hf_split_name} not found in dataset.")
            return

        print(f"Processing {hf_split_name} data...")
        split_data = dataset[hf_split_name]

        for i, item in enumerate(tqdm(split_data)):
            image = item["image"]
            label_idx = item["label"]

            # Determine class name from index
            class_name = label_names[label_idx]

            # Map class_name to our folder structure
            # Our folders: 'normal', 'fracture'
            if "fracture" in class_name.lower():
                dest_dir = dirs[target_split_name]["fracture"]
            elif "normal" in class_name.lower():
                dest_dir = dirs[target_split_name]["normal"]
            else:
                continue  # Skip unknown classes

            # Save image
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            save_path = os.path.join(dest_dir, f"{hf_split_name}_{i}.jpg")
            image.save(save_path)

    # Process splits
    # HF splits are usually 'train', 'validation', 'test'
    save_split("train", "train")
    save_split("validation", "validation")
    save_split("test", "test")

    print("Data processing complete.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    process_hf_dataset(data_dir)
