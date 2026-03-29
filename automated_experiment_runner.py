import os
import subprocess
import json
import time
import pandas as pd
import torch

# Configuration for 10 experiments
EXPERIMENTS = [
    {
        "name": "Exp01_Baseline",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4"]
    },
    {
        "name": "Exp02_FocalLoss",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--loss", "focal"]
    },
    {
        "name": "Exp03_SGD_Momentum",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--optimizer", "sgd"]
    },
    {
        "name": "Exp04_SimplePre_Fast",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "16", "--accumulation_steps", "2", "--simple_pre"]
    },
    {
        "name": "Exp05_MicroModel",
        "args": ["--experiments", "micro", "--epochs", "2", "--batch_size", "16", "--accumulation_steps", "1"]
    },
    {
        "name": "Exp06_HighLR_Decay",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--learning_rate", "0.005"]
    },
    {
        "name": "Exp07_FineTune_Unfrozen",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--fine_tune"]
    },
    {
        "name": "Exp08_Lite_ResNet34",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--cnn_backbone", "resnet34"]
    },
    {
        "name": "Exp09_NoPretrained_ColdStart",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--no_pretrained"]
    },
    {
        "name": "Exp10_Optimal_Combo",
        "args": ["--experiments", "hybrid", "--epochs", "2", "--batch_size", "8", "--accumulation_steps", "4", "--loss", "focal", "--optimizer", "sgd", "--learning_rate", "0.0005"]
    }
]

def run_experiment(exp):
    name = exp["name"]
    args = exp["args"]
    output_dir = f"outputs/experiments/{name}"
    
    print(f"\n\n{'='*20} RUNNING {name} {'='*20}")
    
    cmd = [
        "python", "main.py",
        "--output_dir", output_dir,
        "--data_dir", "data", # Assumes data is present
        "--max_train_batches", "50", # Limit for speed in 10-run iteration
        "--max_val_batches", "20",
        "--max_eval_batches", "10",
        "--num_gradcam", "5"
    ] + args
    
    start_time = time.time()
    try:
        # Run the full pipeline
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        # Extract metrics from output_dir
        # We look for training_history.json in each experiment folder
        metrics = {}
        for root, dirs, files in os.walk(output_dir):
            if "training_history.json" in files:
                with open(os.path.join(root, "training_history.json"), "r") as f:
                    history = json.load(f)
                    metrics["final_val_acc"] = history["val_acc"][-1] if history["val_acc"] else 0
                    metrics["final_val_loss"] = history["val_loss"][-1] if history["val_loss"] else 0
        
        # Check if heatmaps were generated
        heatmap_count = 0
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if "gradcam" in f.lower() and f.endswith((".png", ".jpg")):
                    heatmap_count += 1
        
        metrics["heatmap_count"] = heatmap_count
        metrics["duration_sec"] = duration
        metrics["status"] = "Success"
        
        # Log VRAM peak if possible
        if torch.cuda.is_available():
            metrics["vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"Error in {name}: {e.stderr}")
        return {"status": "Failed", "error": e.stderr}

if __name__ == "__main__":
    summary_results = []
    
    os.makedirs("outputs/experiments", exist_ok=True)
    
    for exp in EXPERIMENTS:
        res = run_experiment(exp)
        res["experiment"] = exp["name"]
        summary_results.append(res)
        
        # Save intermediate results
        pd.DataFrame(summary_results).to_csv("outputs/experiments/experiment_summary.csv", index=False)

    print("\n\nALL EXPERIMENTS COMPLETE.")
    print(pd.DataFrame(summary_results))
