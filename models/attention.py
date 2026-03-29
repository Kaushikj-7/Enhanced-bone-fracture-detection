import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel-wise attention (SE block style)"""
    def __init__(self, input_dim, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_dim = max(1, input_dim // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x), y

class SpatialAttention(nn.Module):
    """Spatial attention (CBAM style)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y, y

class AttentionModule(nn.Module):
    """Combined Channel and Spatial Attention"""
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.channel_att = ChannelAttention(input_dim)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        if x.dim() == 4:
            # Apply channel attention first, then spatial attention
            x_ch, weights_ch = self.channel_att(x)
            x_sp, weights_sp = self.spatial_att(x_ch)
            # The original code expected returning (attended_x, weights)
            return x_sp, (weights_ch, weights_sp)
        else:
            # Fallback for vectors [B, C]
            # Just do a simple SE-style channel attention on vectors if passed
            reduced_dim = max(1, x.size(1) // 16)
            fc = nn.Sequential(
                nn.Linear(x.size(1), reduced_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced_dim, x.size(1), bias=False),
                nn.Sigmoid()
            ).to(x.device)
            weights = fc(x)
            return x * weights, weights

class FractureInterpretabilityFusion(nn.Module):
    """
    Advanced fusion layer designed specifically for Grad-CAM interpretability.
    Combines Local (CNN) and Global (ViT) features using a gated mechanism.
    """
    def __init__(self, cnn_dim, vit_dim, fusion_dim=256):
        super(FractureInterpretabilityFusion, self).__init__()
        # Alignment projections
        self.cnn_proj = nn.Conv2d(cnn_dim, fusion_dim, kernel_size=1)
        self.vit_proj = nn.Conv2d(vit_dim, fusion_dim, kernel_size=1)
        
        # Spatial Gate: Learns where to trust CNN vs ViT
        self.gate = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(),
            nn.Conv2d(fusion_dim, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Final output projection (This is the layer targeted by Grad-CAM)
        self.output_conv = nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(fusion_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cnn_feat, vit_feat):
        # 1. Align dimensions
        cnn_aligned = self.cnn_proj(cnn_feat)
        
        # Up-sample CNN if needed to match ViT spatial dims (usually ViT is 14x14)
        if cnn_aligned.shape[2:] != vit_feat.shape[2:]:
            cnn_aligned = nn.functional.interpolate(
                cnn_aligned, size=vit_feat.shape[2:], mode="bilinear", align_corners=False
            )
        
        vit_aligned = self.vit_proj(vit_feat)
        
        # 2. Compute Gating Map
        cat_feat = torch.cat([cnn_aligned, vit_aligned], dim=1)
        gate_weights = self.gate(cat_feat) # [B, 2, H, W]
        
        # 3. Gated Fusion
        # weight[0] for local, weight[1] for global
        fused = (cnn_aligned * gate_weights[:, 0:1, :, :]) + (vit_aligned * gate_weights[:, 1:2, :, :])
        
        # 4. Refinement (This specific conv is the best target for Grad-CAM)
        out = self.relu(self.bn(self.output_conv(fused)))
        
        return out, gate_weights
