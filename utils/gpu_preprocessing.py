import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.cuda_ops import apply_fused_bone_enhancement


class GPUFracturePreprocessor(nn.Module):
    """
    GPU-native implementation of Wavelet Detail Boost and Frangi-like Ridge Enhancement.
    Moves the "heaviness" from CPU to GPU.
    """

    def __init__(self, device, use_custom_kernels=True, simple_pre=False):
        super().__init__()
        self.device = device
        self.use_custom_kernels = use_custom_kernels
        self.simple_pre = simple_pre

        # Standard ImageNet normalization figures on GPU
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Haar Wavelet Kernels
        self.register_buffer(
            "haar_weights",
            torch.tensor(
                [
                    [[[1, 1], [1, 1]]],  # LL
                    [[[1, -1], [1, -1]]],  # LH
                    [[[1, 1], [-1, -1]]],  # HL
                    [[[1, -1], [-1, 1]]],  # HH
                ],
                dtype=torch.float32,
            )
            / 2.0,
        )

    def wavelet_boost_gpu(self, x):
        """Standard PyTorch-based Wavelet Isolation"""
        coeffs = F.conv2d(x, self.haar_weights, stride=2)
        LL, LH, HL, HH = coeffs.split(1, dim=1)
        # Suppress LL (Structural) and Boost high frequencies
        LH = LH * 2.5
        HL = HL * 2.5
        HH = HH * 3.5
        reconstructed = F.conv_transpose2d(
            torch.cat([LL * 0.1, LH, HL, HH], dim=1), self.haar_weights, stride=2
        )
        return (reconstructed - reconstructed.min()) / (
            reconstructed.max() - reconstructed.min() + 1e-6
        )

    def ridge_enhancement_gpu(self, x):
        """Standard PyTorch-based Ridge (Hessian approximation)"""
        # 1. Image Gradients (Sobel)
        kx = (
            torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], device=self.device)
            .float()
            .unsqueeze(0)
        )
        ky = (
            torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], device=self.device)
            .float()
            .unsqueeze(0)
        )
        ix = F.conv2d(x, kx, padding=1)
        iy = F.conv2d(x, ky, padding=1)
        # 2. Second Derivatives (Hessian)
        ixx = F.conv2d(ix, kx, padding=1)
        iyy = F.conv2d(iy, ky, padding=1)
        # 3. Eigenvalue approximation (Vesselness)
        vesselness = torch.abs(ixx + iyy)
        return (vesselness - vesselness.min()) / (
            vesselness.max() - vesselness.min() + 1e-6
        )

    def forward(self, x):
        """
        Input: B x 3 x 224 x 224 (Raw tensors from CPU - usually just 1 channel duplicated or already CLAHE)
        Output: B x 3 x 224 x 224 (Fully preprocessed & Normalized)
        """
        x = x.to(memory_format=torch.channels_last)
        # Channel 0 is always the structural CLAHE-balanced image from CPU
        structural = x[:, 0:1, :, :]

        if self.simple_pre:
            # Replicate structural channel to 3 channels (Standard Greyscale-to-RGB)
            combined = torch.cat([structural, structural, structural], dim=1)
        else:
            if self.use_custom_kernels and self.device.type == "cuda":
                # FUSED MODE: High performance
                frequency = apply_fused_bone_enhancement(
                    structural, wavelet_gain=2.5, ridge_alpha=4.0
                )
                if frequency is not None:
                    ridge = apply_fused_bone_enhancement(
                        structural, wavelet_gain=0.5, ridge_alpha=10.0
                    )
                    combined = torch.cat([structural, frequency, ridge], dim=1)
                else:
                    # Fallback
                    frequency = self.wavelet_boost_gpu(structural)
                    ridge = self.ridge_enhancement_gpu(structural)
                    combined = torch.cat([structural, frequency, ridge], dim=1)
            else:
                # FALLBACK MODE
                frequency = self.wavelet_boost_gpu(structural)
                ridge = self.ridge_enhancement_gpu(structural)
                combined = torch.cat([structural, frequency, ridge], dim=1)

        # Apply ImageNet normalization for backbone compatibility
        return (combined - self.mean) / self.std
