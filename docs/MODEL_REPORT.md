# Technical Report: Micro-Hybrid Transformer-CNN for Bone Fracture Detection

## 1. Executive Summary
The **Micro-Hybrid Transformer-CNN** is a dual-stream deep learning architecture designed for high-precision medical imaging. It combines the local spatial feature extraction of Convolutional Neural Networks (CNN) with the global contextual reasoning of Vision Transformers (ViT).

## 2. Model Architecture
The architecture consists of two primary branches that fuse into a unified classification head:

### A. CNN Branch (Local Feature Extraction)
- **Base:** MobileNetV3-Small (Pre-trained on ImageNet).
- **Function:** Identifies low-level textures, edges, and fine-grained bone structures.
- **Enhancement:** A custom **Fracture Enhancement Block** using multi-scale dilated convolutions to amplify high-frequency details (like hairline cracks).

### B. Transformer Branch (Global Context)
- **Base:** Micro-Transformer (Custom lightweight ViT).
- **Optimization:** **LoRA (Low-Rank Adaptation)** applied to attention layers to reduce trainable parameters by 85%, preventing overfitting on small medical datasets.
- **Function:** Captures long-range dependencies across the entire X-ray to understand the structural integrity of the bone.

### C. Fusion & Classification
- **Integration:** Features from both branches are concatenated (Feature Fusion).
- **Head:** A 3-layer MLP with Dropout (0.3) for final binary classification (Fracture vs. Normal).

## 3. Architectural Diagram
```text
      [ INPUT X-RAY IMAGE (224x224) ]
                   |
         __________|__________
        |                     |
  [ CNN STREAM ]       [ ViT STREAM ]
  (MobileNetV3)       (Micro-Transformer)
        |                     |
 [ FRACTURE ENHANCER ]  [ LoRA ATTENTION ]
        |                     |
        |__________.__________|
                   |
         [ CONCATENATION LAYER ]
                   |
        [ DENSE CLASSIFIER HEAD ]
                   |
       [ FINAL DIAGNOSIS OUTPUT ]
```

## 4. Image Preprocessing Pipeline
To ensure clinical reliability, every image passes through a 4-stage pipeline:
1. **Adaptive Cropping:** Removes non-relevant black borders from X-ray plates.
2. **CLAHE Enhancement:** Contrast Limited Adaptive Histogram Equalization to highlight bone density.
3. **Lanczos Resampling:** High-quality resizing to 224x224 to prevent aliasing artifacts.
4. **Z-Score Normalization:** Standardizes pixel intensity for model stability.

## 5. Performance Metrics
Based on the `micro_35min_run` production logs:
- **Total Parameters:** ~5.4M (Highly efficient)
- **Validation Accuracy:** 71.0%
- **Inference Speed:** < 45ms per image (on GPU)
- **F1-Score (Normal):** 0.63
- **Recall (Normal):** 0.86 (High sensitivity for healthy bone detection)

---
*This model was developed for the IPCV project to demonstrate the synergy between classical CNNs and modern Transformers in medical diagnostics.*
