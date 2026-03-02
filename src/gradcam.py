import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def find_last_conv(module: nn.Module) -> nn.Module:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")
    return last_conv


class GradCAM:
    """Minimal Grad-CAM for binary (single-logit) classifiers.

    This is intended for the CNN branch of the hybrid/baseline models.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._hook_a = target_layer.register_forward_hook(self._save_activation)
        self._hook_g = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self._hook_a.remove()
        self._hook_g.remove()

    def generate(self, x: torch.Tensor) -> np.ndarray:
        """Return a normalized CAM as float32 array in [0,1] with shape (H,W)."""
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(
                "GradCAM.generate expects a single image tensor of shape (1,C,H,W)"
            )

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits.backward()  # single-logit binary classifier

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (self.activations * weights).sum(dim=1).squeeze(0)  # (H,W)
        cam = torch.clamp(cam, min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy().astype(np.float32)


def overlay_cam_rgb(
    rgb_01: np.ndarray, cam_01: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Overlay a CAM onto an RGB image.

    Args:
        rgb_01: float image in [0,1], shape (H,W,3)
        cam_01: float CAM in [0,1], shape (H,W)
    Returns:
        uint8 RGB image, shape (H,W,3)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_01), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * rgb_01 + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def visualize_gradcam(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    save_path: Optional[str] = None,
    save_prefix: str = "gradcam",
    target_layer: Optional[nn.Module] = None,
):
    """Generate and save a Grad-CAM overlay image.

    Preprocessing is paper-aligned: Resize(224,224) + ToTensor (scales to [0,1]).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb = (np.asarray(img).astype(np.float32) / 255.0).clip(0, 1)

    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    cnn_branch = getattr(model, "cnn_branch", None) or getattr(model, "cnn", None)
    if cnn_branch is None:
        raise ValueError("Model has no cnn_branch/cnn attribute for Grad-CAM")

    if target_layer is None:
        target_layer = find_last_conv(cnn_branch)

    gradcam = GradCAM(model, target_layer)
    try:
        cam = gradcam.generate(x)
    finally:
        gradcam.remove_hooks()

    out = overlay_cam_rgb(rgb, cam)

    if save_path is None:
        base = os.path.basename(image_path)
        save_path = f"{save_prefix}_{base}"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    return save_path
