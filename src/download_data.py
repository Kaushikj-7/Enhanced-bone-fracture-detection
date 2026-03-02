import os
import shutil


def download_mura_dataset(destination_dir):
    print(f"Attempting to download MURA dataset to {destination_dir}...")
    try:
        import kagglehub

        # Download latest version
        path = kagglehub.dataset_download("tommyngx/mura-v1")
        print("Path to downloaded dataset files:", path)

        # The downloaded path likely contains the dataset contents.
        # We want to ensure it ends up in destination_dir/MURA-v1.1 or similar so dataset.py finds it.

        target_dir = os.path.join(destination_dir, "MURA-v1.1")
        if os.path.exists(target_dir):
            print(f"Target directory {target_dir} already exists. Skipping copy.")
            return

        # Check if path contains MURA-v1.1 subdir
        possible_subdir = os.path.join(path, "MURA-v1.1")
        if os.path.exists(possible_subdir):
            source = possible_subdir
        else:
            source = path

        print(f"Copying from {source} to {target_dir}...")
        shutil.copytree(source, target_dir)
        print("Dataset setup complete.")

    except Exception as e:
        print(f"Error downloading dataset: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    download_mura_dataset(data_dir)
