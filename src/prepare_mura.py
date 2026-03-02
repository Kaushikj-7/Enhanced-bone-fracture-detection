import os
import shutil
import glob
from tqdm import tqdm
import subprocess
import sys

# MURA-v1.1 Dataset ID on Kaggle
DATASET_ID = "cjinny/mura-v11" 

def download_mura(download_path):
    print(f"Attempting to download {DATASET_ID} to {download_path}...")
    
    # Check for Kaggle credentials
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("Error: 'kaggle.json' not found in ~/.kaggle/")
        print("Please create an API token from your Kaggle account (Account -> Create New API Token),")
        print("and place the 'kaggle.json' file in C:\\Users\\<YourUser>\\.kaggle\\")
        return False
        
    try:
        # Use Kaggle CLI to download
        # We use subprocess to call 'kaggle' executable which should be in path or accessible via python -m
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET_ID, "--unzip", "-p", download_path]
        subprocess.check_call(cmd)
        print("Download successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def process_mura(source_root, target_root):
    # MURA structure: MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image1.png
    
    print("Processing MURA dataset structure...")
    
    # Define target paths
    target_train_norm = os.path.join(target_root, 'train', 'normal')
    target_train_frac = os.path.join(target_root, 'train', 'fracture')
    target_val_norm = os.path.join(target_root, 'val', 'normal')
    target_val_frac = os.path.join(target_root, 'val', 'fracture')
    target_test_norm = os.path.join(target_root, 'test', 'normal')
    target_test_frac = os.path.join(target_root, 'test', 'fracture')
    
    for p in [target_train_norm, target_train_frac, target_val_norm, target_val_frac, target_test_norm, target_test_frac]:
        os.makedirs(p, exist_ok=True)
        
    # Helper to process a split
    def process_split(split_name, is_train_split):
        mura_split_dir = os.path.join(source_root, "MURA-v1.1", split_name)
        if not os.path.exists(mura_split_dir):
            # Try without version number if extracted differently
            mura_split_dir = os.path.join(source_root, split_name)
            
        if not os.path.exists(mura_split_dir):
             print(f"Could not find MURA {split_name} directory in {source_root}")
             return

        # Walk through MURA directory
        # train/Part/Patient/Study/Image
        print(f"Scanning {split_name} data...")
        images = glob.glob(os.path.join(mura_split_dir, "*", "*", "*", "*.png"))
        
        print(f"Found {len(images)} images in {split_name} set.")
        
        counter = 0
        for img_path in tqdm(images, desc=f"Processing {split_name}"):
            # Determin label from path (studyX_positive or studyX_negative)
            parent_dir = os.path.dirname(img_path) 
            study_name = os.path.basename(parent_dir) # e.g., study1_negative
            
            is_fracture = "positive" in study_name
            is_normal = "negative" in study_name
            
            if not (is_fracture or is_normal):
                continue # Skip unclear items
            
            # Determine target folder
            if is_train_split:
                # MURA 'train' goes to our 'train'
                dest_dir = target_train_frac if is_fracture else target_train_norm
            else:
                # MURA 'valid' -> Split into Val and Test (50/50 split by patient/study preferably, but random for now)
                # Simple random split based on counter
                if counter % 2 == 0:
                    dest_dir = target_val_frac if is_fracture else target_val_norm
                else:
                    dest_dir = target_test_frac if is_fracture else target_test_norm
            
            # Destination filename: preserve uniqueness
            # Use part_patient_study_image.png
            parts = img_path.split(os.sep)
            # parts[-4] = BodyPart, [-3] = Patient, [-2] = Study, [-1] = Image
            try:
                unique_name = f"{parts[-4]}_{parts[-3]}_{parts[-2]}_{parts[-1]}"
            except IndexError:
                unique_name = os.path.basename(img_path)
                
            shutil.copy2(img_path, os.path.join(dest_dir, unique_name))
            counter += 1

    # Process "train" -> train
    process_split("train", True)
    
    # Process "valid" -> val + test
    process_split("valid", False)
    
    print("Processing complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    temp_dir = os.path.join(project_root, 'temp_mura_download')
    
    # 1. Download
    if download_mura(temp_dir):
        # 2. Process
        process_mura(temp_dir, data_dir)
        
        # 3. Cleanup ?
        # shutil.rmtree(temp_dir)
        print("Done. Data is ready in data/ folder.")
    else:
        print("Setup failed.")
