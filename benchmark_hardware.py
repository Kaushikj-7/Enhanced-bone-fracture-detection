import torch
import time
import os
import json
from training.train import train_one_epoch, validate_one_epoch
from src.dataset import get_dataloaders
from models.hybrid_model import HybridModel
from models.micro_hybrid import MicroHybridModel
import psutil

def benchmark_config(config_name, model_type, batch_size, accumulation_steps, simple_pre=False):
    print(f"\n>>> BENCHMARKING: {config_name}")
    print(f"Model: {model_type} | Batch Size: {batch_size} | Accumulation: {accumulation_steps} | Simple Pre: {simple_pre}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a small subset for benchmarking
    # We use a dummy config for dataloader
    data_dir = "data" # Adjust if needed
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Skipping benchmark.")
        return None

    try:
        dataloaders, _ = get_dataloaders(data_dir, batch_size=batch_size, num_workers=2, simple_pre=simple_pre)
        train_loader = dataloaders['train']
    except Exception as e:
        print(f"Failed to load dataloader: {e}")
        return None

    if model_type == "hybrid":
        model = HybridModel(num_classes=1)
    else:
        model = MicroHybridModel(num_classes=1)
        
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Monitor
    from utils.monitoring import ResourceMonitor
    monitor = ResourceMonitor(interval_batches=1)
    
    start_time = time.time()
    max_batches = 10 # Just test 10 batches
    
    # We need a custom loop to limit batches
    model.train()
    batch_times = []
    
    try:
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= max_batches:
                break
            
            b_start = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            batch_times.append(time.time() - b_start)
            monitor.step()
            
            del inputs, labels, outputs, loss
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"!!! OOM DETECTED for {config_name}")
            return {"status": "OOM"}
        else:
            print(f"Error during benchmark: {e}")
            return {"status": "Error", "msg": str(e)}

    avg_time = sum(batch_times) / len(batch_times)
    fps = batch_size / avg_time
    
    vram_peak = 0
    if device.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        
    cpu_avg = psutil.cpu_percent()
    
    print(f"RESULT: {fps:.2f} images/sec | Peak VRAM: {vram_peak:.0f}MB | CPU: {cpu_avg}%")
    
    return {
        "status": "Success",
        "fps": fps,
        "vram_peak": vram_peak,
        "cpu_avg": cpu_avg,
        "avg_batch_time": avg_time
    }

if __name__ == "__main__":
    results = {}
    
    # Test configurations
    configs = [
        ("Micro-B8", "micro", 8, 1, True),
        ("Micro-B16", "micro", 16, 1, True),
        ("Hybrid-B8", "hybrid", 8, 1, False),
        ("Hybrid-B16", "hybrid", 16, 1, False),
        ("Hybrid-B4-Acc4", "hybrid", 4, 4, False),
    ]
    
    for name, m_type, b_size, acc, simple in configs:
        res = benchmark_config(name, m_type, b_size, acc, simple)
        if res:
            results[name] = res
            
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nBenchmarks complete. See benchmark_results.json")
