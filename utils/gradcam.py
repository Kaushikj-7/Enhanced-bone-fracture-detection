import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM implementation for interpretability.
    Typically targeted at the last convolutional layer of the CNN branch.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        # Using register_full_backward_hook for compatibility with modern PyTorch
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.target_layer.register_full_backward_hook(self.save_gradient)
        else:
            self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Handle layers that return (output, weights) tuples like our AttentionModule
        if isinstance(output, tuple):
            self.activations = output[0]
        else:
            self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple containing the gradient w.r.t the output
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        """
        Generates the Grad-CAM heatmap.
        For binary classification with a single output logit:
        - If logit > 0, it predicts 'positive' (fracture).
        - We target the logit directly.
        """
        device = next(self.model.parameters()).device
        input_image = input_image.to(device)

        # Ensure gradients are enabled for the backward pass
        self.model.zero_grad()
        output = self.model(input_image) # [1, 1] for binary
        
        # Backward pass on the raw logit
        # For a single logit, we just call backward() on it
        output.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("ERROR: Gradients or Activations were not captured!")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        # 1. Gradients: [1, C, H, W]
        # 2. Activations: [1, C, H, W]
        
        # Global Average Pooling of gradients (Global Importance weights)
        # We take the mean across spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=[2, 3]) # [1, C]
        
        # Linear combination of weighted activations
        # heatmap = sum(w_i * A_i)
        # We use Einstein summation for clean batch-wise multiplication
        # [1, C] * [1, C, H, W] -> [1, H, W]
        heatmap = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * self.activations, dim=1).squeeze()
        
        # ReLU: We only care about features that have a POSITIVE influence on the class
        heatmap = torch.clamp(heatmap, min=0)
        
        # Normalize to [0, 1]
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 1e-8:
            heatmap = heatmap / heatmap_max
        
        return heatmap.detach().cpu().numpy()

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap on an image.
    :param img: Path to image or BGR image array (numpy).
    :param heatmap: 2D numpy array of the heatmap.
    :param alpha: Transparency of the heatmap.
    :return: BGR image with overlay.
    """
    if isinstance(img, str):
        original_img = cv2.imread(img)
    else:
        # If it's a float array [0, 1], convert to [0, 255] uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        original_img = img

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert heatmap to BGR using the specified colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Superimpose
    superimposed_img = cv2.addWeighted(
        heatmap_colored, alpha, original_img, 1 - alpha, 0
    )

    return superimposed_img
