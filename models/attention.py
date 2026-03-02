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
