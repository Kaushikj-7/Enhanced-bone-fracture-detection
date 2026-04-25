# Resource-Optimized Training Guidelines

The project has been optimized to run on low-resource hardware (e.g., 4GB VRAM, 8GB RAM). 

## Core Optimizations

1.  **Automatic Mixed Precision (AMP):** Enabled by default on CUDA. Reduces VRAM usage and speeds up training.
2.  **Gradient Accumulation:** Allows using small physical batch sizes (e.g., 4 or 8) while maintaining a larger effective batch size (e.g., 32). This is critical for preventing Out-of-Memory (OOM) errors.
3.  **VRAM-Aware Auto-Scaling:** The training pipeline automatically detects available VRAM and adjusts `accumulation_steps` if needed.
4.  **GPU Preprocessing:** Heavy mathematical filters (Wavelets, Ridge) are offloaded to the GPU using `GPUFracturePreprocessor`. This reduces CPU load and prevents system hangs.
5.  **Memory Management:** Explicit `del` of large tensors and `torch.cuda.empty_cache()` at epoch boundaries.

## Strict Guidelines for Experimentation

### 1. Hardware Benchmarking
Before starting a full run, always run the benchmark script to find the best configuration for your specific machine:
```bash
python benchmark_hardware.py
```

### 2. Batch Size Settings
| VRAM | Recommended Batch Size | Accumulation Steps |
| :--- | :--- | :--- |
| < 4GB | 4 | 8 |
| 4GB - 6GB | 8 | 4 |
| 6GB - 8GB | 16 | 2 |
| > 8GB | 32 | 1 |

### 3. Monitoring
Check the `[BENCHMARK]` logs in the console during training.
- **CPU > 90%:** Reduce `num_workers` to 0 or 1 in `config.json`.
- **RAM > 90%:** Close other applications. The preprocessor caches some data in memory.
- **VRAM Spike:** If you see `OOM`, reduce `batch_size` and increase `accumulation_steps`.

### 4. Logging
Results are logged to `outputs/` including a `training_history.json`. Benchmarking results are saved to `benchmark_results.json`.

## How to use "Lite" mode
If the `HybridModel` is still too slow, use the `micro` experiment flag:
```bash
python main.py --experiment micro
```
This uses a significantly smaller model architecture while keeping the same preprocessing pipeline.
