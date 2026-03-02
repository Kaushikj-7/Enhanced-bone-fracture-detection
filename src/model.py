import torch
import torch.nn as nn
import warnings

# This file is maintained for backward compatibility.
# Please use the implementations in the `models/` directory.
from models.hybrid_model import HybridModel
from models.baselines import CNNOnlyModel, ViTOnlyModel

class HybridCNNViT(nn.Module):
    def __init__(self, mode="hybrid"):
        super(HybridCNNViT, self).__init__()
        warnings.warn("HybridCNNViT in src/model.py is deprecated. Use models/ instead.")
        self.mode = mode
        
        if mode == "hybrid":
            self.model = HybridModel(num_classes=1)
        elif mode == "cnn":
            self.model = CNNOnlyModel(num_classes=1)
        elif mode == "vit":
            self.model = ViTOnlyModel(num_classes=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        """Returns the last convolutional layer for GradCAM."""
        if self.mode in ["hybrid", "cnn"]:
            # Retrieve the last conv layer from the CNN branch
            for m in reversed(list(self.model.cnn_branch.modules())):
                if isinstance(m, nn.Conv2d):
                    return m
        return None
