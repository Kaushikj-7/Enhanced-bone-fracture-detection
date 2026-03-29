import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .vit_branch import ViTBranch
from .attention import AttentionModule

class FractureEnhancementBlock(nn.Module):
    """
    Captures multi-scale local details (edges, cracks) using dilated convolutions.
    This fulfills the paper's requirement for the CNN to capture local texture shifts.
    """
    def __init__(self, in_channels):
        super(FractureEnhancementBlock, self).__init__()
        # Branch 1: Standard 1x1 convolution
        self.branch1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        # Branch 2: 3x3 Dilated (Rate 2) for hairline cracks
        self.branch2 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size=3, padding=2, dilation=2
        )
        # Branch 3: 3x3 Dilated (Rate 4) for larger discontinuities
        self.branch3 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size=3, padding=4, dilation=4
        )
        # Branch 4: Global Context
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
        )
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = nn.functional.interpolate(
            self.branch4(x), size=size, mode="bilinear", align_corners=False
        )
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.relu(self.bn(self.fuse(out) + x))
        return out

from .lora import inject_lora

class HybridModel(nn.Module):
    def __init__(
        self,
        cnn_backbone="resnet18",
        vit_model="vit_tiny_patch16_224",
        num_classes=1,
        pretrained=True,
        use_lora=True,
    ):
        super(HybridModel, self).__init__()

        # 1. CNN Branch (Intermediate Extraction)
        full_cnn = CNNBranch(backbone_name=cnn_backbone, pretrained=pretrained).backbone
        
        if cnn_backbone.startswith("resnet"):
            self.cnn_early = nn.Sequential(*list(full_cnn.children())[:6]) # Up to end of Layer 2 (28x28)
            self.cnn_late = nn.Sequential(*list(full_cnn.children())[6:])  # Layer 3 & 4
            cnn_mid_dim = 128 if cnn_backbone == "resnet18" else 512
            self.enhancer = FractureEnhancementBlock(cnn_mid_dim)
            self.cnn_out_dim = 512 if cnn_backbone == "resnet18" else 2048
        else:
            self.cnn_early = full_cnn
            self.cnn_late = nn.Identity()
            self.cnn_out_dim = CNNBranch(backbone_name=cnn_backbone, pretrained=pretrained).out_features
            self.enhancer = FractureEnhancementBlock(self.cnn_out_dim)

        # 2. ViT Branch
        self.vit_branch = ViTBranch(model_name=vit_model, pretrained=pretrained)
        
        # EFFICIENCY: Inject LoRA into ViT Attention and MLP blocks
        if use_lora:
            print(f"Injecting LoRA into {vit_model} branch for parameter efficiency.")
            inject_lora(
                self.vit_branch.vit, 
                r=8, 
                alpha=16.0, 
                target_modules=["qkv", "fc1", "fc2", "proj"]
            )

        # Bottleneck Projections
        self.bottleneck_dim = 256
        self.cnn_proj = nn.Conv2d(self.cnn_out_dim, self.bottleneck_dim, kernel_size=1)
        self.vit_proj = nn.Conv2d(self.vit_branch.out_features, self.bottleneck_dim, kernel_size=1)
        
        combined_dim = self.bottleneck_dim * 2
        self.attention = AttentionModule(input_dim=combined_dim)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        # Freeze backbones by default (Stage 0: Only Enhancer & Head)
        self.freeze_backbones()

    def freeze_backbones(self):
        # Even with LoRA, we freeze the original weights
        for param in self.cnn_early.parameters():
            param.requires_grad = False
        for param in self.cnn_late.parameters():
            param.requires_grad = False
        for param in self.vit_branch.parameters():
            param.requires_grad = False
        
        # UNFREEZE LoRA parameters in ViT (they are nn.Parameters created in LoRALinear)
        # We need to explicitly find them if vit_branch was frozen
        for name, param in self.vit_branch.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # Ensure Enhancer, Head and Attention are trainable
        for param in self.enhancer.parameters():
            param.requires_grad = True
        for param in self.attention.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.cnn_proj.parameters():
            param.requires_grad = True
        for param in self.vit_proj.parameters():
            param.requires_grad = True

    def unfreeze_stage(self, stage: int):
        """
        Stage 0: LoRA adapters + Enhancer + Head (Low compute)
        Stage 1: Unfreeze Late CNN (Higher compute)
        Stage 2: Unfreeze Early CNN + Original ViT weights (Full compute)
        """
        if stage >= 1:
            for param in self.cnn_late.parameters():
                param.requires_grad = True
            print("Unfrozen Late CNN stages.")
        if stage >= 2:
            for param in self.cnn_early.parameters():
                param.requires_grad = True
            for param in self.vit_branch.parameters():
                param.requires_grad = True
            print("Full Transfer Learning enabled.")

    def forward(self, x):
        # CNN Stream with Intermediate Enhancement
        feat_mid = self.cnn_early(x)       # [B, 128, 28, 28]
        feat_enh = self.enhancer(feat_mid) # Enhanced at 28x28 resolution
        cnn_feat = self.cnn_late(feat_enh) # [B, 512, 7, 7]
        
        # ViT Stream
        vit_feat = self.vit_branch(x)
        
        # Project and Fuse
        cnn_feat = self.cnn_proj(cnn_feat)
        vit_feat = self.vit_proj(vit_feat)
        
        # Match spatial dims
        vh, vw = vit_feat.shape[2], vit_feat.shape[3]
        cnn_feat_up = nn.functional.interpolate(cnn_feat, size=(vh, vw), mode="bilinear")
        
        fused = torch.cat((cnn_feat_up, vit_feat), dim=1)
        attended, _ = self.attention(fused)
        
        pooled = torch.mean(attended, dim=[2, 3])
        return self.classifier(pooled)

    def get_last_conv_layer(self):
        """
        Used for Grad-CAM.
        We target the attention module which is the final spatial feature map before pooling.
        This captures the combined local (CNN) and global (ViT) influence.
        """
        return self.attention
