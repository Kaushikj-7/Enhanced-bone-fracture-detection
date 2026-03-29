import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha

        # Original linear layer (frozen)
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(self.linear.weight.size(1), r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.linear.weight.size(0)))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Scaling factor
        self.scaling = alpha / r

        self.reset_parameters()

    @property
    def weight(self):
        # W_merged = W + (A @ B)^T = W + B^T @ A^T
        return self.linear.weight + (self.lora_B.t() @ self.lora_A.t()) * self.scaling

    @property
    def bias(self):
        return self.linear.bias

    def reset_parameters(self):
        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.linear(x)

        # Add LoRA output: x @ A @ B * scaling
        lora_out = self.dropout(x)
        lora_out = lora_out @ self.lora_A @ self.lora_B
        lora_out = lora_out * self.scaling

        return result + lora_out


def inject_lora(
    model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules=["qkv", "fc1", "fc2", "proj", "out_proj"],
):
    """
    Recursively replaces target nn.Linear layers with LoRALinear.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Replace with LoRALinear
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(model, name, lora_layer)
        else:
            # Recursively apply
            inject_lora(
                module, r=r, alpha=alpha, dropout=dropout, target_modules=target_modules
            )
    return model
