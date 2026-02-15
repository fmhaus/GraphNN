import torch
import torch.nn as nn

class MaskedMAE(nn.Module):
    def __init__(self):
        super(MaskedMAE, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_error = torch.abs(output - target) * mask
        return masked_error.sum() / mask.sum().clamp(min=1e-8)