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
        # grad_output is usually a tuple where index 0 is the gradient w.r.t the output
        if isinstance(grad_output[0], tuple):
            self.gradients = grad_output[0][0]
        else:
            self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        print(f"Model output shape: {output.shape}")

        if target_class is None:
            target_class = 0

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        output.backward(retain_graph=True)
        
        if self.gradients is None:
            print("ERROR: Gradients were not captured by hook!")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        print(f"Gradients captured: shape={self.gradients.shape}, mean={self.gradients.mean().item():.6f}")
        print(f"Activations captured: shape={self.activations.shape}, mean={self.activations.mean().item():.6f}")

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
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


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap on an image.
    :param img: Path to image or image array (numpy). If array, assumed to be BGR or RGB depending on source.
    :param heatmap: 2D numpy array of the heatmap.
    :param alpha: Transparency of the heatmap.
    :param colormap: OpenCV colormap.
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

    # Normalize heatmap for visualization
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Convert heatmap to BGR using the specified colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Superimpose
    superimposed_img = cv2.addWeighted(
        heatmap_colored, alpha, original_img, 1 - alpha, 0
    )

    return superimposed_img
