import math
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
    
class MaskedZINBLoss(nn.Module):
    """
    Masked Zero-Inflated Negative Binomial negative log-likelihood for count data.

    Expects raw (pre-activation) model outputs; activations are applied internally:
        mu_raw  → softplus  → NB mean μ > 0
        r_raw   → softplus  → NB dispersion r > 0
        pi_raw  → sigmoid   → zero-inflation probability π ∈ (0, 1)

    NB parameterisation: P(Y=k | μ, r) ∝ Γ(r+k)/Γ(r) · (r/(r+μ))^r · (μ/(r+μ))^k

    ZINB log-likelihood:
        y = 0:  log( π + (1−π) · NB(0|μ,r) )   [stable via logaddexp]
        y > 0:  log(1−π) + log NB(y|μ,r)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu_raw: torch.Tensor,
        r_raw: torch.Tensor,
        pi_raw: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mu = F.softplus(mu_raw)
        r  = F.softplus(r_raw)
        pi = torch.sigmoid(pi_raw)

        log_r         = torch.log(r + self.eps)
        log_mu        = torch.log(mu + self.eps)
        log_r_plus_mu = torch.log(r + mu + self.eps)

        # log NB(y | μ, r)
        log_nb = (
            torch.lgamma(r + target)
            - torch.lgamma(r)
            - torch.lgamma(target + 1)
            + r * (log_r - log_r_plus_mu)
            + target * (log_mu - log_r_plus_mu)
        )

        log_nb_0   = r * (log_r - log_r_plus_mu)   # log NB(0 | μ, r)
        log_pi     = torch.log(pi + self.eps)
        log_1_pi   = torch.log(1.0 - pi + self.eps)

        # y == 0: -log( π + (1−π)·NB(0) )  via logaddexp for numerical stability
        zero_nll    = -torch.logaddexp(log_pi, log_1_pi + log_nb_0)
        nonzero_nll = -(log_1_pi + log_nb)

        nll = torch.where(target == 0, zero_nll, nonzero_nll)
        return (nll * mask).sum() / mask.sum(dtype=torch.float32).clamp(min=1e-8)


class EvalMeter:
    """
    Accumulates predictions and targets (denormalized, masked) across batches
    and computes MAE, RMSE, KL divergence, zero-class P/R/F1, and NLL.

    KL is computed analytically between fitted Gaussians: KL(pred || target).
    NLL assumes a Gaussian with MLE variance (= MSE), giving: 0.5*(log(2π·MSE) + 1).
    Zero P/R/F1 treats values < zero_threshold (default 0.5) as the "zero" class.
    """

    def __init__(self, zero_threshold: float = 0.5, eps: float = 1e-8):
        self.zero_threshold = zero_threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0
        self.sum_pred = 0.0
        self.sum_pred_sq = 0.0
        self.sum_target = 0.0
        self.sum_target_sq = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            m = mask.bool()
            pred = output[m].float()
            tgt = target[m].float()

            err = pred - tgt
            self.sum_abs_err += err.abs().sum().item()
            self.sum_sq_err += (err ** 2).sum().item()
            self.sum_pred += pred.sum().item()
            self.sum_pred_sq += (pred ** 2).sum().item()
            self.sum_target += tgt.sum().item()
            self.sum_target_sq += (tgt ** 2).sum().item()

            pred_zero = pred < self.zero_threshold
            tgt_zero = tgt < self.zero_threshold
            self.tp += (pred_zero & tgt_zero).sum().item()
            self.fp += (pred_zero & ~tgt_zero).sum().item()
            self.fn += (~pred_zero & tgt_zero).sum().item()
            self.n += m.sum().item()

    def compute(self) -> dict:
        if self.n == 0:
            return {k: 0.0 for k in ['mae', 'rmse', 'kl', 'zero_precision', 'zero_recall', 'zero_f1', 'nll']}

        n = self.n
        mae = self.sum_abs_err / n
        rmse = math.sqrt(self.sum_sq_err / n)

        # Gaussian KL: KL(pred || target)
        mu_p = self.sum_pred / n
        var_p = max(self.sum_pred_sq / n - mu_p ** 2, self.eps)
        mu_q = self.sum_target / n
        var_q = max(self.sum_target_sq / n - mu_q ** 2, self.eps)
        kl = 0.5 * (math.log(var_q / var_p) + var_p / var_q + (mu_p - mu_q) ** 2 / var_q - 1.0)

        # Gaussian NLL with MLE variance (sigma^2 = MSE)
        sigma2 = max(self.sum_sq_err / n, self.eps)
        nll = 0.5 * (math.log(2 * math.pi * sigma2) + 1.0)

        # Zero-class precision / recall / F1
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'mae':            mae,
            'rmse':           rmse,
            'kl':             kl,
            'zero_precision': precision,
            'zero_recall':    recall,
            'zero_f1':        f1,
            'nll':            nll,
        }