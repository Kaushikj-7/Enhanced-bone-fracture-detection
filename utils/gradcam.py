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
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            # Assuming binary outcome (sigmoid > 0.5)
            target_class = 0  # Single output neuron index

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        # Since output is [B, 1], we can backward directly
        output.backward(retain_graph=True)

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


def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    original_img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Superimpose
    superimposed_img = cv2.addWeighted(
        heatmap_colored, alpha, original_img, 1 - alpha, 0
    )

    return superimposed_img
