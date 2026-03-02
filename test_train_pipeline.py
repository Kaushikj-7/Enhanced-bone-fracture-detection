import torch
import os
from src.dataset import get_dataloaders
from models.hybrid_model import HybridModel
from training.train import train_pipeline

class MockDataLoader:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = [None] * len(batch[0]) # Mocking dataset for len()

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1

def test_train():
    data_dir = "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Tiny dataset / Small batch
    dataloaders, _ = get_dataloaders(data_dir, batch_size=2, num_workers=0)
    
    # Mock config
    config = {
        "training": {
            "learning_rate": 0.001,
            "num_epochs": 1,
            "patience": 2
        }
    }
    
    save_dir = "test_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Initializing Model...")
    model = HybridModel(pretrained=False).to(device)
    
    # Mocking dataloaders to have only 1 batch for speed
    train_batch = next(iter(dataloaders["train"]))
    val_batch = next(iter(dataloaders["val"]))
    
    train_loader = MockDataLoader(train_batch)
    val_loader = MockDataLoader(val_batch)
    
    print("Starting tiny training pipeline...")
    try:
        model, history = train_pipeline(model, train_loader, val_loader, config, device, save_dir)
        print("Training pipeline PASSED.")
    except Exception as e:
        print(f"Training pipeline FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_train()
