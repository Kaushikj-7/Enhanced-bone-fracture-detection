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
        """
        # Ensure model and input are on same device
        device = next(self.model.parameters()).device
        input_image = input_image.to(device)

        # Forward pass
        self.model.zero_grad()
        output = self.model(input_image)
        
        if target_class is None:
            # Target the logit directly for binary classification
            target_class_idx = 0
        else:
            target_class_idx = target_class

        # Backward pass
        output.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("ERROR: Gradients or Activations were not captured!")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        # Pool the gradients across the channels
        # Gradients shape: [B, C, H, W]
        # We take the mean across H and W dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        # Activations shape: [B, C, H, W]
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU on top of the heatmap
        heatmap = np.maximum(heatmap.cpu(), 0)

        # Normalize the heatmap
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap.numpy()

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
