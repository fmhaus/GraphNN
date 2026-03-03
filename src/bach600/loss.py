import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMAE(nn.Module):
    def __init__(self):
        super(MaskedMAE, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_error = torch.abs(output - target) * mask
        return masked_error.sum() / mask.sum(dtype=torch.float32).clamp(min=1e-8)


class MaskedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super(MaskedHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.huber_loss(output, target, delta=self.delta, reduction="none")
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum(dtype=torch.float32).clamp(min=1e-8)
    
class SMAPEMeter:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.numerator = 0.0
        self.denominator = 0.0
        self.n_masked = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            num = 2 * torch.abs(output - target)
            denom = torch.abs(output) + torch.abs(target) + self.eps
            self.numerator += (num * mask).sum().item()
            self.denominator += (denom * mask).sum().item()
            self.n_masked += mask.sum().item()

    def compute(self) -> float:
        if self.n_masked == 0:
            return 0.0
        return self.numerator / self.denominator  # multiply by 100 for percentage