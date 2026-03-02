import os
from src.dataset import get_dataloaders

def test_dataset():
    data_dir = "data"
    print(f"Testing dataset loading from {data_dir}...")
    
    # Check if data exists
    if not os.path.exists(os.path.join(data_dir, "MURA-v1.1")):
        print("MURA-v1.1 dataset not found in 'data/' directory.")
        return

    try:
        dataloaders, datasets = get_dataloaders(data_dir, batch_size=4, num_workers=0)
        
        for split in ["train", "val", "test"]:
            dataset = datasets.get(split)
            if dataset and len(dataset) > 0:
                print(f"{split.capitalize()} set size: {len(dataset)}")
                # Try to get one batch
                loader = dataloaders[split]
                inputs, labels = next(iter(loader))
                print(f"{split.capitalize()} batch shape: {inputs.shape}, Labels: {labels}")
            else:
                print(f"{split.capitalize()} set is empty or missing.")
                
        print("Dataset verification COMPLETED.")
    except Exception as e:
        print(f"Dataset verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
