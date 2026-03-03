# Project Progress & Architectural Context

## 1. Current Strategic Focus: Micro-Hybrid (1M Parameters)
To optimize computational resources and ensure architectural integrity before scaling, we are currently training a **1M parameter Micro-Hybrid model**.

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

## 6. Google Colab Integration
- **`hybrid_cnn_vit_colab.ipynb`**: Fully updated to support the 1M Micro-Hybrid model, advanced preprocessing (CLAHE/Wavelets/Frangi), and AMP-optimized training.
1.  **Colab Full Execution**: Run `hybrid_cnn_vit_colab.ipynb` to complete the 10-epoch training run using the new high-reliability pipeline.
2.  **Inspect Results**: Analyze `outputs/micro_colab` for heatmap accuracy on hairline fractures.
3.  **Scale to 17M Model**: Once validated, transition to `HybridModel` (ResNet18 + ViT-Tiny) by switching the `--experiments` flag in the pipeline.

---
*Last Updated: Tuesday, 3 March 2026 (Aligned with Research-Backed Pipeline)*
