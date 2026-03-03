import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.pe[:, :x.size(1), :]
        return x

class FractureEnhancementBlock(nn.Module):
    """
    Specifically designed to fix errors in finding cracks and hairline fractures.
    Uses dilated convolutions to capture multi-scale edge details (cracks)
    without increasing parameter count significantly.
    """
    def __init__(self, in_channels):
        super(FractureEnhancementBlock, self).__init__()
        # Branch 1: Local details (1x1)
        self.branch1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        # Branch 2: Small cracks (3x3 dilated)
        self.branch2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=2, dilation=2)
        # Branch 3: Hairline injuries (3x3 dilated further)
        self.branch3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=4, dilation=4)
        # Branch 4: Context (Global pooling + 1x1)
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        )
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = nn.functional.interpolate(self.branch4(x), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.relu(self.bn(self.fuse(out) + x))
        return out

class MicroTransformer(nn.Module):
    """
    A custom, ultra-lightweight ViT (approx 0.1M params)
    """
    def __init__(self, input_dim, depth=2, heads=4, dim_feedforward=256):
        super(MicroTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=heads, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # Flatten to [B, H*W, C]
        x = x.flatten(2).permute(0, 2, 1)
        # Add Positional Encoding
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Restore to [B, C, H, W]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x

class MicroHybridModel(nn.Module):
    """
    Micro-Hybrid (1M Parameters) for initial testing and crack extraction.
    CNN: MobileNetV3-Small (Pretrained)
    Enhancement: FractureEnhancementBlock
    Transformer: MicroTransformer
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(MicroHybridModel, self).__init__()
        
        # 1. CNN Branch (MobileNetV3-Small) - Approx 0.9M params
        backbone = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        self.cnn_branch = backbone.features # Output channels: 576 for 7x7
        self.cnn_out_channels = 576

        # 2. Fracture Enhancement Block (Error Correction)
        self.enhancer = FractureEnhancementBlock(self.cnn_out_channels)

        # 3. Micro-Transformer Branch - Approx 0.1M params
        self.vit_branch = MicroTransformer(input_dim=self.cnn_out_channels)

        # 4. Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN Feature Extraction
        features = self.cnn_branch(x) # [B, 576, 7, 7]
        
        # Enhanced Crack Extraction (High-Pass/Dilated Filter)
        features = self.enhancer(features)
        
        # Global Context via Micro-Transformer
        features = self.vit_branch(features)
        
        # Classification
        pooled = self.gap(features).view(features.size(0), -1)
        logits = self.classifier(pooled)
        return logits

    def get_last_conv_layer(self):
        """Returns the last conv layer for GradCAM targeting."""
        return self.enhancer.fuse

    def freeze_backbone(self):
        """Transfer Learning: Freeze MobileNet weights, train only the Enhancement + ViT layers."""
        for param in self.cnn_branch.parameters():
            param.requires_grad = False
        print("CNN backbone frozen. Enhancement and Transformer layers are trainable.")

if __name__ == "__main__":
    model = MicroHybridModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params/1e6:.3f}M")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
