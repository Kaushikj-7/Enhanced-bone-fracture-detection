import torch
import torch.nn as nn
import timm
import math

class ViTBranch(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super(ViTBranch, self).__init__()

        # Load Pretrained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_features = self.vit.num_features

    def forward(self, x):
        # x: [B, 3, H, W]
        # Get all patch tokens + CLS token
        # output: [B, N, D]
        x = self.vit.forward_features(x)

        # timm models return [B, N, D] where N = num_patches + 1 (for CLS token)
        # Take only patch tokens (excluding CLS token)
        # Often the CLS token is at index 0 or index -1 depending on the specific ViT, 
        # but in standard vit_base_patch16_224 it is at index 0.
        if self.vit.global_pool in ('avg', 'token'):
            # If global_pool was somehow enabled and applied, shape might be [B, D]. 
            # num_classes=0 usually avoids this, but let's be safe.
            if len(x.shape) == 2:
                raise ValueError("ViT forward_features returned [B, D]. Expected sequence [B, N, D].")
        
        # Check if it has class token
        has_cls_token = getattr(self.vit, 'has_class_token', True)
        if has_cls_token and x.shape[1] > 1:
            patch_tokens = x[:, 1:, :]  # [B, NumPatches, D]
        else:
            patch_tokens = x
        
        # Determine spatial dimensions
        num_patches = patch_tokens.shape[1]
        h = w = math.isqrt(num_patches)
        
        # Ensure it's a perfect square (which it is for 224x224 and 16x16 patches -> 14x14=196)
        if h * w != num_patches:
            raise ValueError(f"Cannot reshape {num_patches} patches into a square grid.")

        # Reshape to spatial map
        # [B, N, D] -> [B, D, H, W]
        spatial_features = patch_tokens.permute(0, 2, 1).reshape(
            patch_tokens.shape[0], self.out_features, h, w
        )

        return spatial_features
