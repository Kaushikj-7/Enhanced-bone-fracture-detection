import torch
import torch.nn as nn
import torchvision.models as models


def _resolve_weights(model_fn_name: str, pretrained: bool):
    if not pretrained:
        return None

    # Torchvision >=0.13 uses explicit Weights enums.
    if model_fn_name == "resnet18":
        return getattr(models, "ResNet18_Weights").DEFAULT
    if model_fn_name == "resnet50":
        return getattr(models, "ResNet50_Weights").DEFAULT
    if model_fn_name == "mobilenet_v3_small":
        return getattr(models, "MobileNet_V3_Small_Weights").DEFAULT
    return None


class CNNBranch(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True):
        super(CNNBranch, self).__init__()

        # Load Pretrained Backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(
                weights=_resolve_weights("resnet18", pretrained)
            )
            self.out_features = self.backbone.fc.in_features
            # Remove the classification head
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(
                weights=_resolve_weights("resnet50", pretrained)
            )
            self.out_features = self.backbone.fc.in_features
            # Remove Head (FC) AND Pooling (AdvPool/AvgPool)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # Add support for other backbones if needed (e.g., MobileNetV3 as mentioned in paper comparisons)
        elif backbone_name == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(
                weights=_resolve_weights("mobilenet_v3_small", pretrained)
            )
            self.out_features = self.backbone.classifier[0].in_features
            # MobileNet structure is complex, usually features are in 'features' block
            # For simplicity let's stick to ResNet modifications or handle carefully
            self.backbone = self.backbone.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.backbone(x)
        # features: [B, C, H', W'] (e.g., [B, 512, 7, 7] for ResNet18)
        return features
