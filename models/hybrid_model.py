import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .vit_branch import ViTBranch
from .attention import AttentionModule

class HybridModel(nn.Module):
    def __init__(
        self,
        cnn_backbone="resnet18",
        vit_model="vit_tiny_patch16_224",
        num_classes=1,
        pretrained=True,
    ):
        super(HybridModel, self).__init__()

        # 1. CNN Branch - Extracts local visual features
        self.cnn_branch = CNNBranch(backbone_name=cnn_backbone, pretrained=pretrained)

        # 2. ViT Branch - Extracts global contextual information
        self.vit_branch = ViTBranch(model_name=vit_model, pretrained=pretrained)

        # FREEZE BACKBONES (Feature Extraction Only) by default
        for param in self.cnn_branch.parameters():
            param.requires_grad = False
        for param in self.vit_branch.parameters():
            param.requires_grad = False

        # Dynamic dimension retrieval to support any backbone
        cnn_out_dim = self.cnn_branch.out_features
        vit_out_dim = self.vit_branch.out_features
        
        # Optimization: Bottleneck Projection
        # Reduce high-dimensional features to a shared bottleneck dimension
        # This reduces parameters in the attention and classification heads significantly.
        self.bottleneck_dim = 256
        self.cnn_proj = nn.Conv2d(cnn_out_dim, self.bottleneck_dim, kernel_size=1)
        self.vit_proj = nn.Conv2d(vit_out_dim, self.bottleneck_dim, kernel_size=1)
        
        combined_dim = self.bottleneck_dim * 2

        # 3. Attention Mechanism
        self.attention = AttentionModule(input_dim=combined_dim)

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def set_fine_tuning(self, cnn_unfreeze=True, vit_unfreeze=True):
        """
        Selectively unfreeze the last layers of backbones for 'Optimization'.
        """
        if cnn_unfreeze:
            params = list(self.cnn_branch.backbone.parameters())
            # Unfreeze the last block
            for param in params[-min(len(params), 10):]:
                param.requires_grad = True

        if vit_unfreeze:
            params = list(self.vit_branch.vit.parameters())
            # Unfreeze the last block
            for param in params[-min(len(params), 15):]:
                param.requires_grad = True

        print("Model set to Fine-tuning mode. Last layers unfrozen.")

    def forward(self, x):
        # Extract features
        cnn_feat = self.cnn_branch(x)  # [B, C_cnn, H_cnn, W_cnn]
        vit_feat = self.vit_branch(x)  # [B, C_vit, H_vit, W_vit]

        # Apply Bottleneck Projections (Reduces compute for later stages)
        cnn_feat = self.cnn_proj(cnn_feat)
        vit_feat = self.vit_proj(vit_feat)

        # Spatial Feature Fusion: dynamically match ViT spatial dimensions
        vit_h, vit_w = vit_feat.shape[2], vit_feat.shape[3]
        cnn_feat_up = nn.functional.interpolate(
            cnn_feat, size=(vit_h, vit_w), mode="bilinear", align_corners=False
        )

        # Concatenate along channel dimension
        fused = torch.cat((cnn_feat_up, vit_feat), dim=1)  # [B, bottleneck*2, H_vit, W_vit]

        # Apply Attention Module
        attended_feat, _ = self.attention(fused)

        # Global Average Pooling
        pooled_feat = torch.mean(attended_feat, dim=[2, 3])  # [B, combined_dim]

        # Final Classification
        logits = self.classifier(pooled_feat)

        return logits
