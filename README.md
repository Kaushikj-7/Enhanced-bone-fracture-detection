# Enhanced Bone Fracture Detection in X-ray Imaging

This project implements a **Hybrid Convolutional Neural Network (CNN) and Vision Transformer (ViT)** model with an **Attention Mechanism** for bone fracture detection, based on the architectural pipeline from the research paper: *"A Hybrid Convolutional and Vision Transformer Model with Attention Mechanism for Enhanced Bone Fracture Detection in X-ray Imaging"*.

## Architecture
1. **CNN Branch**: Local feature extraction (ResNet-based).
2. **ViT Branch**: Global context capture (Vision Transformer).
3. **Attention Module**: Channel and Spatial attention for feature refinement.
4. **Fusion Layer**: Dynamic integration of local and global features.
5. **Grad-CAM**: Interpretability via class activation mapping.

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
