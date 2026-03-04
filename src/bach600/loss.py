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

    Expects raw model_output [B, N, H, 3]; activations are applied internally:
        [..., 0]  mu_raw  → softplus  → NB mean μ > 0
        [..., 1]  r_raw   → softplus  → NB dispersion r > 0
        [..., 2]  pi_raw  → sigmoid   → zero-inflation probability π ∈ (0, 1)

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
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mu_raw, r_raw, pi_raw = model_output.unbind(dim=-1)
        mu = F.softplus(mu_raw.float())
        r  = F.softplus(r_raw.float())
        pi = torch.sigmoid(pi_raw.float())

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
    Accumulates ZINB parameters and targets (raw counts, masked) across batches
    and computes MAE, RMSE, KL divergence, zero-class P/R/F1, and ZINB NLL.

    update() expects raw model_output [B, N, H, 3] and raw count targets.
      E[Y] = (1−π)·μ                    → MAE, RMSE, KL
      P(Y=0) = π + (1−π)·NB(0|μ,r)     → zero-class classification
      exact ZINB log-likelihood          → NLL
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.sum_abs_err   = 0.0
        self.sum_sq_err    = 0.0
        self.sum_pred      = 0.0
        self.sum_pred_sq   = 0.0
        self.sum_target    = 0.0
        self.sum_target_sq = 0.0
        self.sum_nll       = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n  = 0

    def update(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ):
        with torch.no_grad():
            mu_raw, r_raw, pi_raw = model_output.unbind(dim=-1)
            m    = mask.bool()
            mu_m = F.softplus(mu_raw[m].float())
            r_m  = F.softplus(r_raw[m].float())
            pi_m = torch.sigmoid(pi_raw[m].float())
            tgt  = target[m].float()

            pred = (1.0 - pi_m) * mu_m  # ZINB expected value E[Y]

            err = pred - tgt
            self.sum_abs_err   += err.abs().sum().item()
            self.sum_sq_err    += (err ** 2).sum().item()
            self.sum_pred      += pred.sum().item()
            self.sum_pred_sq   += (pred ** 2).sum().item()
            self.sum_target    += tgt.sum().item()
            self.sum_target_sq += (tgt ** 2).sum().item()

            # Exact ZINB NLL
            log_r         = torch.log(r_m + self.eps)
            log_mu        = torch.log(mu_m + self.eps)
            log_r_plus_mu = torch.log(r_m + mu_m + self.eps)
            log_nb = (
                torch.lgamma(r_m + tgt)
                - torch.lgamma(r_m)
                - torch.lgamma(tgt + 1)
                + r_m * (log_r - log_r_plus_mu)
                + tgt * (log_mu - log_r_plus_mu)
            )
            log_nb_0 = r_m * (log_r - log_r_plus_mu)
            log_pi   = torch.log(pi_m + self.eps)
            log_1_pi = torch.log(1.0 - pi_m + self.eps)
            log_lik  = torch.where(
                tgt == 0,
                torch.logaddexp(log_pi, log_1_pi + log_nb_0),
                log_1_pi + log_nb,
            )
            self.sum_nll += (-log_lik).sum().item()

            # Zero-class: predict zero when P(Y=0) > 0.5
            log_p_zero = torch.logaddexp(log_pi, log_1_pi + log_nb_0)
            pred_zero  = log_p_zero > math.log(0.5)
            tgt_zero   = tgt == 0
            self.tp += (pred_zero & tgt_zero).sum().item()
            self.fp += (pred_zero & ~tgt_zero).sum().item()
            self.fn += (~pred_zero & tgt_zero).sum().item()
            self.n  += m.sum().item()

    def compute(self) -> dict:
        if self.n == 0:
            return {k: 0.0 for k in ['mae', 'rmse', 'kl', 'zero_precision', 'zero_recall', 'zero_f1', 'nll']}

        n = self.n
        mae  = self.sum_abs_err / n
        rmse = math.sqrt(self.sum_sq_err / n)
        nll  = self.sum_nll / n

        # Gaussian KL: KL(pred || target) from accumulated moments
        mu_p  = self.sum_pred / n
        var_p = max(self.sum_pred_sq / n - mu_p ** 2, self.eps)
        mu_q  = self.sum_target / n
        var_q = max(self.sum_target_sq / n - mu_q ** 2, self.eps)
        kl = 0.5 * (math.log(var_q / var_p) + var_p / var_q + (mu_p - mu_q) ** 2 / var_q - 1.0)

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