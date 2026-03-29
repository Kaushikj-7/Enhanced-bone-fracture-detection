Project: Hybrid CNN–ViT with Attention for Bone Fracture Detection

Source: Selvaraj et al., ICMSCI 2025 (IEEE)

1. Objective

Implement a hybrid deep learning model combining:

Convolutional Neural Network (CNN)

Vision Transformer (ViT)

Attention mechanism

Feature fusion

Binary classification head

Grad-CAM interpretability

Target task: Binary fracture detection (Fracture / No Fracture)

2. Dataset Requirements (As Mentioned in Paper)

Primary datasets:

MURA dataset

Bone X-ray dataset

Input format:

X-ray grayscale images

Labeled binary classes

Expected preprocessing:

Resize to 224×224

Normalize pixel values to [0, 1]

Apply augmentation

3. Data Preprocessing Pipeline
3.1 Image Standardization

Resize → (224, 224)

Convert to 3-channel if required (ViT compatibility)

Normalize → scale pixel values to [0,1]

3.2 Data Augmentation

Apply:

Random rotation

Horizontal flipping

Zoom transformation

Optional brightness/contrast jitter

Reason (paper justification):
To prevent overfitting due to limited medical data.

4. Model Architecture Specification
4.1 CNN Branch (Local Feature Extraction)

Purpose:
Extract local spatial features (edges, textures, patterns).

Structure:

Convolution layers

ReLU activations

Pooling layers

Final feature vector extraction

Output:
Local feature embedding vector

4.2 Vision Transformer Branch (Global Context Extraction)

Purpose:
Capture long-range dependencies across image regions.

Steps:

Split image into patches (e.g., 16×16)

Flatten patches

Linear embedding

Add positional encoding

Pass through Transformer encoder layers

Output:
Global feature embedding vector

4.2.1 Architectural Learning Rationale (Same Input, Different Representations)

Important design clarification:
The CNN branch and ViT branch should consume the same X-ray image. They still learn different information because of different architectural inductive biases.

CNN inductive bias:
- Convolution kernels (e.g., 3×3, 5×5) enforce local receptive fields.
- Early layers detect local structures: crack edges, micro-discontinuities, and fine texture changes.

ViT inductive bias:
- Patch tokenization + self-attention allows each patch to compare with all other patches.
- This supports global relational reasoning: long-range alignment, structural consistency, and context-aware anomaly detection.

Fusion interpretation:
Same X-ray
→ CNN local features
→ ViT global context
→ Fusion/Attention
→ stronger fracture probability estimate

Clinical example:
- CNN may detect a thin hairline discontinuity.
- ViT may detect subtle distal radius-carpal misalignment.
- Combined signal is stronger than either cue alone.

Key ML principle:
Hybrid performance gain comes from **inductive bias diversity**, not from using different input data per branch.

4.3 Attention Mechanism

Applied after CNN + ViT feature extraction.

Purpose:

Focus on fracture-relevant regions

Improve interpretability

Improve feature prioritization

Implementation:

Attention layer applied on fused features
OR

Channel/spatial attention module

4.4 Feature Fusion Strategy

From paper:
“Feature fusion of combined CNN, ViT and attention outputs”

Implementation:

Concatenate CNN features + ViT features

Pass through attention module

Global average pooling

Dense layer

4.5 Classification Head

Fully connected layer

Sigmoid activation

Binary output

Output:
Probability of fracture

5. Training Configuration (Paper-Specified)

Optimizer:
Adam

Learning Rate:
0.001

Loss Function:
Binary Cross-Entropy

Epochs:
20

Early stopping:
Based on validation loss

Batch size:
Choose 16 or 32 depending on GPU memory

6. Evaluation Metrics (As Used in Paper)

Compute:

Accuracy

Precision

Recall

F1 Score

AUC-ROC

Sensitivity

Specificity

Expected reference performance (from paper):

Accuracy: 91.3%
Precision: 89.1%
Recall: 92.4%
F1 Score: 90.7%
AUC-ROC: 94.8%

7. Interpretability Module

Implement Grad-CAM:

Generate class activation heatmaps

Overlay heatmap on original X-ray

Visualize fracture-focused regions

Purpose:
Improve clinical reliability

8. Experimental Comparison (Optional Extension)

Paper compares against:

CNN-only model

ViT-only model

SVM

CNN-SVM hybrid

MobileNetV3

CNN-LSTM hybrid

Implement at least:

CNN baseline

ViT baseline

Hybrid model

9. Directory Structure
project/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── cnn_branch.py
│   ├── vit_branch.py
│   ├── attention.py
│   ├── hybrid_model.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│
├── utils/
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── gradcam.py
│
├── configs/
│   ├── config.json
│
└── main.py
10. Implementation Order (Critical Execution Plan)

Step 1 → Implement preprocessing
Step 2 → Build CNN-only model
Step 3 → Build ViT-only model
Step 4 → Implement feature fusion
Step 5 → Add attention module
Step 6 → Add Grad-CAM
Step 7 → Train hybrid model
Step 8 → Evaluate and compare

Mental Model to Remember

When paper says:

“CNN captures local patterns”

Think:
Convolution → sliding kernel → spatial feature detector

When paper says:

“ViT captures global dependency”

Think:
Self-attention → Q, K, V → weighted global interaction

When paper says:

“Attention improves interpretability”

Think:
Weighted importance map over feature space

Architecture logic =

Local (CNN)

Global (ViT)

Focus (Attention)
= Robust medical classifier