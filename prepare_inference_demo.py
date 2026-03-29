import os
import shutil
import pandas as pd
import glob

def prepare_demo_set(source_dir, target_dir, num_per_class=100):
    positive_target = os.path.join(target_dir, "positive")
    negative_target = os.path.join(target_dir, "negative")
    
    os.makedirs(positive_target, exist_ok=True)
    os.makedirs(negative_target, exist_ok=True)
    
    # MURA paths for validation
    valid_dir = os.path.join(source_dir, "valid")
    
    all_images = glob.glob(os.path.join(valid_dir, "**/*.png"), recursive=True)
    
    pos_images = [img for img in all_images if "positive" in img]
    neg_images = [img for img in all_images if "negative" in img]
    
    print(f"Found {len(pos_images)} positive and {len(neg_images)} negative images in validation.")
    
    selected_pos = pos_images[:num_per_class]
    selected_neg = neg_images[:num_per_class]
    
    manifest = []
    
    def copy_list(images, target, label):
        for i, img_path in enumerate(images):
            filename = f"image_{i+1}.png"
            dest_path = os.path.join(target, filename)
            shutil.copy(img_path, dest_path)
            manifest.append({
                "original_path": img_path,
                "demo_path": os.path.join("inference_demo_set", label, filename),
                "label": label
            })
            
    copy_list(selected_pos, positive_target, "positive")
    copy_list(selected_neg, negative_target, "negative")
    
    pd.DataFrame(manifest).to_csv("inference_manifest.csv", index=False)
    print(f"Copied {len(selected_pos)} positive and {len(selected_neg)} negative images.")
    print("Manifest created: inference_manifest.csv")

if __name__ == "__main__":
    prepare_demo_set("data/MURA-v1.1", "inference_demo_set")
