"""Differentiable Fréchet-distance training loss (arXiv:2604.28190v1, "Representation Fréchet
Loss for Visual Generation"). Real-side stats are fixed and precomputed once, offline (see
scripts/compute_real_frechet_stats_vjepa.py); generated-side stats are tracked with a running EMA
so a small per-step batch still gives a low-variance distance estimate -- the paper's core trick,
decoupling the population size used for the estimate from the batch size used for gradients.

Not reusable: mini_mira.world_model.full_eval_metrics.frechet_distance() round-trips through
scipy.linalg.sqrtm/numpy -- non-differentiable by construction, not just by an incidental
no_grad() wrapper. This is a fresh, pure-torch implementation for training-time use specifically.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RealFrechetStats:
    """Fixed, precomputed once from real held-out data -- mean/cov_sqrt/trace_cov never need
    gradients, so cov_sqrt (the expensive part) is computed once here, not every training step."""

    mean: Tensor
    cov_sqrt: Tensor
    trace_cov: Tensor

    @staticmethod
    def from_mean_cov(mean: Tensor, cov: Tensor, eps: float = 1e-6) -> "RealFrechetStats":
        cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        cov_sqrt = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
        return RealFrechetStats(mean=mean, cov_sqrt=cov_sqrt, trace_cov=eigenvalues.sum())

    def to(self, device: torch.device | str) -> "RealFrechetStats":
        return RealFrechetStats(
            mean=self.mean.to(device), cov_sqrt=self.cov_sqrt.to(device), trace_cov=self.trace_cov.to(device)
        )


def differentiable_frechet_distance(real: RealFrechetStats, mean_g: Tensor, cov_g: Tensor, eps: float = 1e-6) -> Tensor:
    """FD = ||mean_r-mean_g||^2 + Tr(cov_r) + Tr(cov_g) - 2*Tr((cov_r@cov_g)^0.5). The cross term
    uses M = cov_sqrt_r @ cov_g @ cov_sqrt_r (symmetric PSD by construction): Tr(M^0.5) ==
    Tr((cov_r@cov_g)^0.5) exactly, and M^0.5's trace only needs M's eigenvalues, not a second full
    matrix sqrt -- torch.linalg.eigvalsh is autograd-differentiable for symmetric input, unlike
    scipy's sqrtm. mean_g/cov_g are the only differentiable inputs; real's fields are fixed."""
    mean_diff = real.mean - mean_g
    mean_term = mean_diff @ mean_diff

    m = real.cov_sqrt @ cov_g @ real.cov_sqrt
    m = 0.5 * (m + m.T)  # exact symmetry required by eigvalsh; guards float roundoff
    cross_eigenvalues = torch.linalg.eigvalsh(m)
    cross_trace = torch.sqrt(torch.clamp(cross_eigenvalues, min=eps)).sum()

    cov_term = real.trace_cov + torch.diagonal(cov_g).sum() - 2 * cross_trace
    return mean_term + cov_term


class FDLossEMAState(torch.nn.Module):
    """Tracks the generated-side distribution's running mean/second-moment via an EMA (Algorithm 1
    in arXiv:2604.28190v1) -- gradients flow only through the current batch's contribution to the
    update, not through the detached running history. `beta` default (0.97) is deliberately much
    lower than the paper's own 0.999: that value is tuned for thousands of post-training steps and
    would barely move from its seed within this project's 500-step fine-tune budget. Double
    precision throughout, matching full_eval_metrics.OnlineGaussian's own convention -- the real
    stats this compares against are computed the same way."""

    mu_ema: Tensor
    m_ema: Tensor
    initialized: Tensor

    def __init__(self, dim: int, beta: float = 0.97):
        super().__init__()
        self.beta = beta
        self.register_buffer("mu_ema", torch.zeros(dim, dtype=torch.double))
        self.register_buffer("m_ema", torch.eye(dim, dtype=torch.double))
        self.register_buffer("initialized", torch.tensor(False))

    def update(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """features: (n, dim), gradient-tracked (call under no_grad for warm-start-only use).
        Returns (mean_g, cov_g) for this step's loss; also updates mu_ema/m_ema in place
        (detached) for the next step. The very first call seeds the EMA directly from that batch
        instead of blending with the arbitrary zero/identity init."""
        features = features.double()
        mu_batch = features.mean(dim=0)
        m_batch = (features.T @ features) / features.shape[0]

        if bool(self.initialized):
            mu_g = self.beta * self.mu_ema.detach() + (1 - self.beta) * mu_batch
            m_g = self.beta * self.m_ema.detach() + (1 - self.beta) * m_batch
        else:
            mu_g, m_g = mu_batch, m_batch
            self.initialized.fill_(True)

        cov_g = m_g - torch.outer(mu_g, mu_g)
        with torch.no_grad():
            self.mu_ema.copy_(mu_g.detach())
            self.m_ema.copy_(m_g.detach())
        return mu_g, cov_g
