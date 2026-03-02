import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .vit_branch import ViTBranch
from .attention import AttentionModule

class HybridModel(nn.Module):
    def __init__(
        self,
        cnn_backbone="resnet18",
        vit_model="vit_base_patch16_224",
        num_classes=1,
        pretrained=True,
    ):
        super(HybridModel, self).__init__()

        # 4.1 CNN Branch - Extracts local visual features
        self.cnn_branch = CNNBranch(backbone_name=cnn_backbone, pretrained=pretrained)

        # 4.2 ViT Branch - Extracts global contextual information
        self.vit_branch = ViTBranch(model_name=vit_model, pretrained=pretrained)

        # FREEZE BACKBONES (Feature Extraction Only)
        for param in self.cnn_branch.parameters():
            param.requires_grad = False
        for param in self.vit_branch.parameters():
            param.requires_grad = False

        # Dynamic dimension retrieval to support any backbone
        cnn_out_dim = self.cnn_branch.out_features
        vit_out_dim = self.vit_branch.out_features
        combined_dim = cnn_out_dim + vit_out_dim

        # 4.3 Attention Mechanism
        self.attention = AttentionModule(input_dim=combined_dim)

        # 4.5 Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def set_fine_tuning(self, cnn_unfreeze=True, vit_unfreeze=True):
        """
        Selectively unfreeze the last layers of backbones for 'Optimization'.
        This allows the model to learn specific medical textures (fractures).
        """
        if cnn_unfreeze:
            # Unfreeze the last layer of ResNet (layer4)
            for name, param in self.cnn_branch.backbone.named_parameters():
                if "7" in name: # Index 7 is the last block in our Sequential ResNet
                    param.requires_grad = True
        
        if vit_unfreeze:
            # Unfreeze the last block of ViT
            # For timm models, typically named 'blocks.N'
            for name, param in self.vit_branch.vit.named_parameters():
                if "blocks.11" in name or "norm" in name: # Last block for ViT-Base, for Tiny check name
                    param.requires_grad = True
        
        print("Model set to Fine-tuning mode.")

    def forward(self, x):
        # Extract features
        cnn_feat = self.cnn_branch(x)  # [B, C_cnn, H_cnn, W_cnn]
        vit_feat = self.vit_branch(x)  # [B, C_vit, H_vit, W_vit]

        # Spatial Feature Fusion: dynamically match ViT spatial dimensions
        vit_h, vit_w = vit_feat.shape[2], vit_feat.shape[3]
        cnn_feat_up = nn.functional.interpolate(
            cnn_feat, size=(vit_h, vit_w), mode="bilinear", align_corners=False
        )

        # Concatenate along channel dimension
        fused = torch.cat((cnn_feat_up, vit_feat), dim=1)  # [B, combined_dim, H_vit, W_vit]

        # Apply Attention Module
        attended_feat, _ = self.attention(fused)

        # Global Average Pooling
        pooled_feat = torch.mean(attended_feat, dim=[2, 3])  # [B, combined_dim]

        # Final Classification
        logits = self.classifier(pooled_feat)  # [B, num_classes]

        return logits
