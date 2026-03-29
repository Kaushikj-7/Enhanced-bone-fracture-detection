# Enhanced Bone Fracture Detection in X-ray Imaging

This project implements a **Hybrid Convolutional Neural Network (CNN) and Vision Transformer (ViT)** model with an **Attention Mechanism** for bone fracture detection, based on the architectural pipeline from the research paper: *"A Hybrid Convolutional and Vision Transformer Model with Attention Mechanism for Enhanced Bone Fracture Detection in X-ray Imaging"*.

## Architecture
1. **CNN Branch**: Local feature extraction (ResNet-based).
2. **ViT Branch**: Global context capture (Vision Transformer).
3. **Attention Module**: Channel and Spatial attention for feature refinement.
4. **Fusion Layer**: Dynamic integration of local and global features.
5. **Grad-CAM**: Interpretability via class activation mapping.

## Why the Same X-ray Works for Both Branches
The hybrid model does **not** require different data for each branch. Both branches consume the same X-ray, but they learn different representations because their internal mathematics imposes different inductive biases.

- **CNN branch (local bias):** Convolutions use small receptive fields (e.g., 3×3, 5×5), so early layers focus on local neighborhoods. This favors crack edges, tiny discontinuities, and fine texture shifts.
- **ViT branch (global bias):** Self-attention compares all image patches with each other, enabling long-range relational reasoning such as joint alignment and structural consistency.
- **Fusion outcome:** Local fracture cues from CNN + global anatomical context from ViT produce a richer decision signal than either branch alone.

Conceptually:

Same X-ray → CNN local features + ViT global context → fused representation → fracture probability

Practical example: a hairline radius crack may be weak alone, and mild wrist misalignment may also be weak alone, but their fused evidence can strongly increase fracture confidence.

This is the core reason hybrid CNN-Transformer designs became effective in medical imaging: they exploit **inductive bias diversity** rather than different input data.

## Setup & Usage
For easy execution, use the provided Google Colab notebook:
- Open `hybrid_cnn_vit_colab.ipynb` in Colab.
- Set Runtime to **GPU (T4)**.
- Run the cells to clone, install dependencies, and train.

## Local Execution
```bash
pip install -r requirements.txt
python main.py --experiments hybrid
```
