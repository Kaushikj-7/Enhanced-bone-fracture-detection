import torch
import torch.nn as nn
import torchvision.models as models
import math
from models.lora import inject_lora


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class FractureEnhancementBlock(nn.Module):
    def __init__(self, in_channels):
        super(FractureEnhancementBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.branch2 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size=3, padding=2, dilation=2
        )
        self.branch3 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size=3, padding=4, dilation=4
        )
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


class MicroTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=8, embed_dim=64, num_heads=4, num_layers=2
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        if return_attention:
            return x[:, 0], None
        return x[:, 0]


class MicroHybridModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(MicroHybridModel, self).__init__()
        self.backbone = models.mobilenet_v3_small(
            weights="DEFAULT" if pretrained else None
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.cnn_branch = self.backbone.features
        self.cnn_out_channels = 576
        self.enhancer = FractureEnhancementBlock(self.cnn_out_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.vit_branch = MicroTransformer(
            img_size=224, patch_size=8, embed_dim=64, num_heads=4, num_layers=2
        )
        fused_dim = self.cnn_out_channels + 64
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Inject LoRA into the ViT branch to significantly reduce trainable parameters
        inject_lora(
            self.vit_branch.transformer,
            r=4,
            alpha=8.0,
            target_modules=["linear1", "linear2"],
        )

    def forward(self, x):
        features = self.cnn_branch(x)
        features = self.enhancer(features)
        cnn_out = self.gap(features).view(features.size(0), -1)
        vit_out = self.vit_branch(x)
        fused = torch.cat([cnn_out, vit_out], dim=1)
        logits = self.classifier(fused)
        return logits

    def get_last_conv_layer(self):
        return self.enhancer.fuse

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_stage(self, stage: int):
        layers = list(self.backbone.features.children())
        if stage >= 1:
            for param in layers[-1].parameters():
                param.requires_grad = True
        if stage >= 2:
            for param in layers[-2].parameters():
                param.requires_grad = True


if __name__ == "__main__":
    model = MicroHybridModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.3f}M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.3f}M")
    print(f"Output shape: {model(torch.randn(1, 3, 224, 224)).shape}")
