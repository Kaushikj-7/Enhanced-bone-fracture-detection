import os
import hashlib
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AutoCrop:
    """
    Crops black borders and applies CLAHE to enhance bone density/cracks.
    """
    def __init__(self, threshold=15, apply_clahe=True):
        self.threshold = threshold
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # 1. CLAHE Enhancement for cracks
        if self.apply_clahe:
            gray = self.clahe.apply(gray)
            # If original was RGB, we put enhanced gray back into channels or keep as RGB?
            # Pretrained models expect 3 channels.
            img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
        # 2. Find pixels above threshold
        mask = gray > self.threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            return Image.fromarray(img_np)
            
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = img_np[y0:y1, x0:x1]
        return Image.fromarray(cropped)

try:
    import kagglehub
except ImportError:
    kagglehub = None

kaggle = None

try:
    import pandas as pd
except ImportError:
    pd = None


class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, download=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): One of 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download (bool): If true, downloads the dataset from Kaggle.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if download:
            self._download_dataset()

        self.classes = ["normal", "fracture"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _download_dataset(self):
        if os.path.exists(os.path.join(self.root_dir, "train")) or os.path.exists(
            os.path.join(self.root_dir, "MURA-v1.1")
        ):
            print("Dataset already appears to exist. Skipping download.")
            return

        print("Downloading MURA dataset...")

        # Try using kagglehub first as it is the new standard
        if kagglehub is not None:
            try:
                import shutil

                print("Using kagglehub to download tommyngx/mura-v1...")
                path = kagglehub.dataset_download("tommyngx/mura-v1")
                print(f"Downloaded to {path}")

                # Copy to root_dir
                target_dir = os.path.join(self.root_dir, "MURA-v1.1")

                # If path contains MURA-v1.1 subdir, use that
                possible_subdir = os.path.join(path, "MURA-v1.1")
                if os.path.exists(possible_subdir):
                    source = possible_subdir
                else:
                    source = path

                if not os.path.exists(target_dir):
                    print(f"Copying from {source} to {target_dir}...")
                    shutil.copytree(source, target_dir)
                else:
                    print(f"Target directory {target_dir} already exists.")

                return
            except Exception as e:
                print(f"kagglehub download failed: {e}")

        # Fallback to kaggle API if kagglehub fails or isn't installed
        print("Downloading MURA dataset from Kaggle API directly...")
        try:
            # Lazy import: importing kaggle at module import time can fail in environments
            # without credentials, but we only need it in this fallback.
            import kaggle as kaggle_pkg

            # Authenticate
            kaggle_pkg.api.authenticate()

            # Download MURA dataset
            # Using 'stanfordml/mura-v11'
            dataset_name = "stanfordml/mura-v11"
            download_path = self.root_dir

            if not os.path.exists(download_path):
                os.makedirs(download_path)

            kaggle_pkg.api.dataset_download_files(
                dataset_name, path=download_path, unzip=True
            )
            print("Download and extraction complete.")

            # (Folder organization logic remains same, but opendatasets puts it in 'mura-v11' folder usually)

        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("Please ensure you have a valid kaggle.json in ~/.kaggle/")

            # Organize files if necessary (MURA comes with specific structure)
            # MURA structure usually: MURA-v1.1/train/...
            # We need to map it to our self.root_dir

            # Check for extracted folder
            extracted_folder = os.path.join(download_path, "MURA-v1.1")

            if os.path.exists(extracted_folder):
                # Move contents or Symlink?
                # For simplicity, we just leave it and let _make_dataset find it
                # BUT _make_dataset looks for 'train'/'val' directly under root_dir or MURA-v1.1?
                # Let's handle MURA path in _make_dataset or move folders here.

                # Helper to move folders up
                import shutil

                for sub in ["train", "valid"]:
                    src = os.path.join(extracted_folder, sub)
                    dst_name = "val" if sub == "valid" else "train"
                    dst = os.path.join(self.root_dir, dst_name)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.move(src, dst)
                        print(f"Moved {src} to {dst}")

        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("Please ensure you have a valid kaggle.json in ~/.kaggle/")

    def _make_dataset(self):
        instances = []
        # Check standard path
        directory = os.path.join(self.root_dir, self.split)

        # Check MURA path adjustment (if user didn't move but structure exists)
        # Often MURA unzips as MURA-v1.1/train
        if not os.path.exists(directory):
            mura_dir = os.path.join(
                self.root_dir,
                "MURA-v1.1",
                "valid" if self.split in {"val", "test"} else self.split,
            )
            if os.path.exists(mura_dir):
                directory = mura_dir

        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            return []

        # Strategy 1: Standard "normal" / "fracture" folders
        found_standard = False
        for target_class in self.classes:
            class_dir = os.path.join(directory, target_class)
            if os.path.isdir(class_dir):
                found_standard = True
                target = self.class_to_idx[target_class]
                for root, _, fnames in os.walk(class_dir, followlinks=True):
                    for fname in fnames:
                        path = os.path.join(root, fname)
                        if self._is_image_file(path):
                            item = (path, target)
                            instances.append(item)

        if found_standard and len(instances) > 0:
            return instances

        # Strategy 2: MURA-style recursively search for "_positive" and "_negative" folders
        # MURA structure: train/XR_ELBOW/patient00011/study1_negative/image1.png
        # Also supports other datasets where class is part of folder name
        print(
            f"Standard class folders not found in {directory}. Scanning for MURA-style structure..."
        )

        def _split_bucket(study_rel_path: str) -> str:
            # Deterministic 50/50 split across studies.
            digest = hashlib.md5(study_rel_path.encode("utf-8")).hexdigest()
            return "val" if (int(digest[:8], 16) % 2 == 0) else "test"

        # Fast path: use MURA CSV filelists if present.
        # This avoids expensive directory walking over the full tree.
        mura_root = os.path.join(self.root_dir, "MURA-v1.1")
        train_csv = os.path.join(mura_root, "train_image_paths.csv")
        valid_csv = os.path.join(mura_root, "valid_image_paths.csv")

        def _study_rel_from_rel_image_path(rel_image_path: str) -> str:
            rel_image_path = rel_image_path.replace("\\", "/")
            if "/train/" in rel_image_path:
                suffix = rel_image_path.split("/train/", 1)[1]
            elif "/valid/" in rel_image_path:
                suffix = rel_image_path.split("/valid/", 1)[1]
            else:
                suffix = rel_image_path
            # Remove filename => study folder
            return suffix.rsplit("/", 1)[0]

        # Only use CSVs when the dataset is actually the MURA layout.
        if (
            os.path.isdir(mura_root)
            and os.path.exists(train_csv)
            and os.path.exists(valid_csv)
        ):
            csv_path = train_csv if self.split == "train" else valid_csv
            if self.split in {"train", "val", "test"} and os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rel = line.strip().replace("\\", "/")
                        if not rel:
                            continue

                        # Map to absolute path under root_dir
                        abs_path = os.path.join(self.root_dir, rel.replace("/", os.sep))
                        if not self._is_image_file(abs_path):
                            continue

                        rel_l = rel.lower()
                        if "positive" in rel_l:
                            target = self.class_to_idx["fracture"]
                        elif "negative" in rel_l:
                            target = self.class_to_idx["normal"]
                        else:
                            continue

                        # If we're using valid_image_paths.csv, create a stable val/test split by study.
                        if csv_path == valid_csv and self.split in {"val", "test"}:
                            study_rel = _study_rel_from_rel_image_path(rel)
                            if _split_bucket(study_rel) != self.split:
                                continue

                        instances.append((abs_path, target))

                if len(instances) > 0:
                    return instances

        for root, dirs, fnames in os.walk(directory, followlinks=True):
            # Check if current folder indicates a class
            folder_name = os.path.basename(root).lower()
            target = None

            # MURA convention: "positive" = abnormal/fracture, "negative" = normal
            if "positive" in folder_name:
                target = self.class_to_idx["fracture"]
            elif "negative" in folder_name:
                target = self.class_to_idx["normal"]

            if target is not None:
                # If we're indexing MURA's 'valid' folder, create a stable val/test split.
                if os.path.basename(directory).lower() == "valid" and self.split in {
                    "val",
                    "test",
                }:
                    study_rel = os.path.relpath(root, directory).replace("\\", "/")
                    if _split_bucket(study_rel) != self.split:
                        continue
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if self._is_image_file(path):
                        instances.append((path, target))

        return instances

    def _is_image_file(self, filename):
        return filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target


from utils.advanced_preprocessing import AdvancedFracturePreprocessor

def get_transforms(split):
    # Standard ImageNet normalization figures for our 3-channel stack
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose(
            [
                AdvancedFracturePreprocessor(target_size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
                normalize,
            ]
        )
    else:
        # Validation/Test
        return transforms.Compose(
            [
                AdvancedFracturePreprocessor(target_size=(224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    # Only use num_workers=0 on Windows to avoid multiprocessing issues in simple scripts
    if os.name == "nt":
        num_workers = 0

    # Auto-download if needed (using 'train' split as trigger)
    download_flag = False
    if not os.path.exists(os.path.join(data_dir, "train")) and not os.path.exists(
        os.path.join(data_dir, "MURA-v1.1")
    ):
        # Check if empty or missing, might want to trigger download
        # We trigger download only on the first dataset init
        download_flag = True

    datasets = {}

    # Initialize train set first to handle potential download
    datasets["train"] = BoneFractureDataset(
        data_dir,
        split="train",
        transform=get_transforms("train"),
        download=download_flag,
    )

    # Init others without download flag (it's already done or checked)
    for x in ["val", "test"]:
        datasets[x] = BoneFractureDataset(
            data_dir, split=x, transform=get_transforms(x), download=False
        )

    dataloaders = {}
    for x in ["train", "val", "test"]:
        if len(datasets[x]) > 0:
            dataloaders[x] = DataLoader(
                datasets[x],
                batch_size=batch_size,
                shuffle=(x == "train"),
                num_workers=num_workers,
            )
        else:
            print(f"Warning: Dataset split '{x}' is empty.")
            dataloaders[x] = []

    return dataloaders, datasets
