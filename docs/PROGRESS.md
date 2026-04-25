# Project Progress & Architectural Context

## 1. Current Strategic Focus: Micro-Hybrid (1M Parameters)
To optimize computational resources and ensure architectural integrity before scaling, we are currently training a **1M parameter Micro-Hybrid model**.

### Why Both Branches Use the Same X-ray
The hybrid model does **not** require separate data streams for CNN and ViT. Both branches receive the same image, but learn complementary signals due to architectural inductive bias.

- **CNN bias (locality)**: convolution kernels operate on small neighborhoods, so the branch specializes in crack edges, micro-discontinuities, and local texture shifts.
- **ViT bias (global relations)**: self-attention compares patch-to-patch relationships, so the branch captures long-range alignment and structural consistency.
- **Fusion advantage**: weak local and weak global cues can combine into a strong fracture decision boundary.

Conceptual flow:

Same X-ray -> CNN local features + ViT global context -> fused representation -> fracture probability

### **Architecture: `models/micro_hybrid.py`**
- **CNN Branch**: MobileNetV3-Small (Pretrained, ~0.9M params).
- **Enhancement Block**: `FractureEnhancementBlock` using multi-scale dilated convolutions to specifically capture hairline fractures and cracks.
- **ViT Branch**: `MicroTransformer` (2-layer, 4-head) for global anatomical context.
- **Transfer Learning**: The CNN backbone is frozen by default (`model.freeze_backbone()`) to focus training on crack-specific features.

## 2. Enterprise-Grade Preprocessing Pipeline (Phase 1-3 Alignment)
We have fully aligned the pipeline with the **Optimal High-Reliability Pipeline** (CLAHE + Wavelet Domain Filtering) to prevent "shortcut learning" from text tags and noise.

### **Sequential Steps implemented in `utils/advanced_preprocessing.py`:**
1.  **Phase 1: Deterministic Sanitization (The Guardrail)**: 
    -   Implemented robust boundary cropping using the largest anatomical contour.
    -   Added heuristic-based text/marker masking to ensure the Vision Transformer does not use burnt-in tags as classification weights.
2.  **Phase 2: Localized Illumination via CLAHE (The Equalizer)**:
    -   Applied localized contrast equalization with optimized clip limits to maximize trabecular visibility while suppressing background noise.
3.  **Phase 3: High-Frequency Isolation via Wavelet Transform (The Extractor)**:
    -   Upgraded to **Discrete Wavelet Transform (DWT)** with **Soft Thresholding (Shrinkage)** and **Homomorphic-style Non-linear Gain**.
    -   This aggressively amplifies microscopic hairline cracks while suppressing anatomical shape and noise.

### **Input to Model (3-Channel Feature Map):**
- **Channel R**: Structural (CLAHE-balanced image).
- **Channel G**: Frequency (Wavelet-boosted/shrunk detail coefficients).
- **Channel B**: Ridge (Frangi-filtered fracture lines).

## 3. Interpretability & Heatmaps
Fixed "Spatial Drift" in Grad-CAM by ensuring the target layer is a 2D convolutional feature map, not flattened transformer tokens.
- **Target Layer**: `model.enhancer.fuse` (The point where all crack-enhanced features are merged).
- **Visualization**: Updated `overlay_heatmap` to handle RGB/BGR correctly and support direct image array input.

## 4. Local Verification & Pipeline Integrity
- **Visualization Script**: Created `test_pipeline_visualize.py` to verify the 3-channel (CLAHE, Wavelet, Frangi) stack.
  - Successfully isolated trabecular bone (Phase 2) and high-frequency crack detail (Phase 3).
- **Mini-Training Check**: Executed `test_mini_train.py` on local CPU (4 samples, 1 epoch).
  - **Result**: Fully integrated loop (Preprocessing -> Micro-Hybrid Model -> Training -> Metric Logging) works without errors.
  - **Output**: Verified model weights saved correctly.

## 5. Optimization & Performance (RTX/T4 Ready)
- **Automatic Mixed Precision (AMP)**: Integrated `torch.cuda.amp` into `training/train.py` to leverage Tensor Cores on T4 GPUs, expected to reduce training time by 50-70%.
- **Bottleneck Projection**: Implemented a 1x1 convolution bottleneck (256-dim) in `HybridModel` to reduce the dimensionality of concatenated CNN+ViT features, lowering memory overhead for the Attention and Classifier heads.
- **Micro-Optimization**: Verified `MicroHybridModel` as the "Gold Standard" for low-compute environments (1M parameters).

## 6. Pipeline Execution Update (13 March 2026)
- Completed end-to-end fast-plan runs through the current pipeline:
  - `micro` -> outputs in `outputs/micro_pipeline_run` with split-aware (`val,test`) artifacts + final report.
  - `cnn, vit, hybrid` -> outputs in `outputs/plan_fast_compare` with split-aware (`val,test`) artifacts + final report.
- All generated checkpoints are additionally synchronized to `trained_models/` to keep trained weights separated from architecture code paths.
- Full long-run execution started:
  - `micro` (10 epochs, full loaders, split-aware artifacts/report target) -> active output path `outputs/micro_full_10ep`.

## 7. Input Contract Guarantee for Model Building
To enforce model-ready input consistency, the dataset pipeline now applies a strict preprocessing wrapper before tensor conversion.

Guaranteed conditions before model forward pass:
- Image mode is always `RGB`.
- Spatial size is always `(224, 224)`.
- Channel count is always `3`.
- Pixel content is sanitized to finite `uint8` range `[0, 255]` before `ToTensor()`.
- If advanced preprocessing fails on any sample, a deterministic safe fallback (RGB conversion + resize + numeric sanitization) is applied.

This guarantees that dataloader output shape remains `B x 3 x 224 x 224`, matching model input requirements for CNN/ViT/Hybrid branches.

## 8. Google Colab Integration
- **`hybrid_cnn_vit_colab.ipynb`**: Fully updated to support the 1M Micro-Hybrid model, advanced preprocessing (CLAHE/Wavelets/Frangi), and AMP-optimized training.
1.  **Colab Full Execution**: Run `hybrid_cnn_vit_colab.ipynb` to complete the 10-epoch training run using the new high-reliability pipeline.
2.  **Inspect Results**: Analyze `outputs/micro_colab` for heatmap accuracy on hairline fractures.
3.  **Scale to 17M Model**: Once validated, transition to `HybridModel` (ResNet18 + ViT-Tiny) by switching the `--experiments` flag in the pipeline.


### **GPU Acceleration Verified (14 March 2026)**
- **Hardware**: NVIDIA GeForce RTX 2050 (4GB VRAM).
- **Environment**: PyTorch 2.6.0+cu124 (CUDA 12.4).
- **Benefit**: Training time on the full MURA dataset is expected to be reduced significantly (est. 10-15x faster than CPU).

| 2026-03-14 | local | cnn,vit,hybrid,micro | 1 (fast plan) | val/test | roc,cm,gradcam,report | acc=0.5719 (vit) | verified all 4 models in a single combined run |

---
*Last Updated: Saturday, 14 March 2026 (Pipeline Verified on Local CPU)*

## 9. Shortcut Learning & Architecture Fixes (19 March 2026)
Successfully diagnosed and applied pipeline/architectural fixes to resolve Grad-CAM boundary shortcut learning ('R TKB' text mask focus) and frozen weight mismatches:
- **Phase 1: Preprocessing Flow**: Applied robust text masking and corrected augmentation to occur *before* intense multi-channel filtering (CLAHE, wavelets, frangi).
- **Phase 2: Class Imbalance**: Integrated WeightedRandomSampler directly into datasets alongside explicit numerical pos_weight passing to BCEWithLogitsLoss.
- **Phase 3: Backbone Dynamics**: Rebuilt MicroHybridModel to strictly utilize a parallel CNN-ViT stream with used concatenation. Adopted 3-stage unfreeze_stage for transfer-learning custom input.
- **Phase 4: Fracture Resolution**: Stepped down ViT patch_size from 16 to 8 allowing stronger gradient reception over hairline crack frequencies.
- **Next Step**: Conduct a bounded --max_train_batches mini-run on the GPU to lock F1, Precision, Recall, and AUC before launching the multi-hour final ResNet scale.
