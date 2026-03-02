import torch
import torch.nn as nn

from .cnn_branch import CNNBranch
from .vit_branch import ViTBranch


class CNNOnlyModel(nn.Module):
    def __init__(self, cnn_backbone="resnet18", num_classes=1, pretrained=True):
        super().__init__()
        self.cnn_branch = CNNBranch(backbone_name=cnn_backbone, pretrained=pretrained)

        # Freeze backbone for transfer learning
        for param in self.cnn_branch.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_branch.out_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.cnn_branch(x)
        # GAP due to spatial output from branch
        feat = feat.mean(dim=[2, 3])
        return self.classifier(feat)


class ViTOnlyModel(nn.Module):
    def __init__(
        self, vit_model="vit_base_patch16_224", num_classes=1, pretrained=True
    ):
        super().__init__()
        self.vit_branch = ViTBranch(model_name=vit_model, pretrained=pretrained)

        # Freeze backbone for transfer learning
        for param in self.vit_branch.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.vit_branch.out_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.vit_branch(x)
        # GAP due to spatial output from branch
        feat = feat.mean(dim=[2, 3])
        return self.classifier(feat)
