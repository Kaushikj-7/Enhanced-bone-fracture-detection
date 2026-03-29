import os
import shutil
import pandas as pd
import random
from tqdm import tqdm

def prepare_demo_set(root_dir, output_dir, n_positive=10, n_negative=10):
    valid_dir = os.path.join(root_dir, 'MURA-v1.1', 'valid')
    if not os.path.exists(valid_dir):
        # Check standard layout
        valid_dir = os.path.join(root_dir, 'valid')
        if not os.path.exists(valid_dir):
            # Try val
            valid_dir = os.path.join(root_dir, 'val')
            
    if not os.path.exists(valid_dir):
        print(f"Error: Could not find validation directory in {root_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    positive_paths = []
    negative_paths = []

    for root, dirs, files in os.walk(valid_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                if 'positive' in root.lower():
                    positive_paths.append(path)
                elif 'negative' in root.lower():
                    negative_paths.append(path)

    print(f"Found {len(positive_paths)} positive and {len(negative_paths)} negative images.")

    if len(positive_paths) < n_positive or len(negative_paths) < n_negative:
        print("Warning: Not enough images found for the requested count.")
        n_positive = min(n_positive, len(positive_paths))
        n_negative = min(n_negative, len(negative_paths))

    selected_positive = random.sample(positive_paths, n_positive)
    selected_negative = random.sample(negative_paths, n_negative)

    demo_data = []

    for i, path in enumerate(selected_positive):
        new_name = f"pos_{i:02d}.png"
        shutil.copy(path, os.path.join(images_dir, new_name))
        demo_data.append({'filename': new_name, 'label': 'fracture', 'label_idx': 1, 'original_path': path})

    for i, path in enumerate(selected_negative):
        new_name = f"neg_{i:02d}.png"
        shutil.copy(path, os.path.join(images_dir, new_name))
        demo_data.append({'filename': new_name, 'label': 'normal', 'label_idx': 0, 'original_path': path})

    df = pd.DataFrame(demo_data)
    df.to_csv(os.path.join(output_dir, 'ground_truth.csv'), index=False)
    print(f"Demo set prepared in {output_dir}")
    print(f"Created ground_truth.csv with {len(df)} entries.")

if __name__ == "__main__":
    prepare_demo_set('data', 'demo_set', n_positive=10, n_negative=10)
