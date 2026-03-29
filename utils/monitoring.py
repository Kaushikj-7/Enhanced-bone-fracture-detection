import psutil
import torch
import time

class ResourceMonitor:
    def __init__(self, interval_batches=10):
        self.interval = interval_batches
        self.batch_count = 0
        self.start_time = time.time()
        
    def step(self):
        self.batch_count += 1
        if self.batch_count % self.interval == 0:
            self.log_metrics()
            
    def log_metrics(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        gpu_info = ""
        if torch.cuda.is_available():
            # Get VRAM usage for the current device
            device = torch.cuda.current_device()
            vram_used = torch.cuda.memory_allocated(device) / (1024 ** 2) # MB
            vram_cached = torch.cuda.memory_reserved(device) / (1024 ** 2) # MB
            gpu_info = f" | GPU VRAM: {vram_used:.0f}MB used, {vram_cached:.0f}MB cached"
            
        elapsed = time.time() - self.start_time
        print(f"\n[BENCHMARK] Batch {self.batch_count} | CPU: {cpu_usage}% | RAM: {ram_usage}%{gpu_info} | Elapsed: {elapsed:.2f}s")
        
    def final_report(self):
        print("\n--- FINAL RESOURCE REPORT ---")
        self.log_metrics()
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
