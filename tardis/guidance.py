"""Stage II: bidirectional Chamfer guidance during reverse diffusion (Sec. 3.2, App. C)."""

import math
from typing import Optional

import torch

from .chamfer import chamfer, embed


def gamma_schedule(kind: str, g_min: float, g_max: float, tau: float) -> float:
    """Gamma(t) of App. C.1 with tau = t / T in [0, 1]: tau = 1 is pure noise, tau = 0 the
    final sample."""
    if kind == "constant":
        return g_max
    if kind == "linear":
        return g_min + (g_max - g_min) * tau
    if kind == "cosine":
        return g_min + 0.5 * (g_max - g_min) * (1 + math.cos(math.pi * tau))
    if kind == "sine":
        return g_max - 0.5 * (g_max - g_min) * (1 + math.cos(math.pi * tau))
    raise ValueError(f"unknown guidance schedule {kind!r}")


class ChamferGuidance:
    """Score perturbation  eps~ = eps - Gamma(t) * grad_{x_t} C(phi(x_t), phi(x_r))   (Eq. 4).

    The functional is evaluated on the backbone's prediction of the clean sample. The
    reference batch x_r is drawn from D_real at the start of each generation and held
    fixed across diffusion steps (App. C.3): a fraction ``ref_ratio`` of D_real, drawn
    and matched per class for classification (class-conditional batching) and globally
    for regression. ``ref_ratio=None`` uses the whole of D_real, globally.
    """

    def __init__(self, phi, x_real: torch.Tensor, num_timesteps: int, schedule="constant",
                 gamma_min=1e-6, gamma_max=0.5, t_g=5, ref_ratio: Optional[float] = None,
                 target_index: Optional[int] = None, device="cpu", seed=0):
        self.phi = phi.to(device)
        self.x_real = x_real.to(device)
        self.T = num_timesteps
        self.schedule, self.gamma_min, self.gamma_max, self.t_g = schedule, gamma_min, gamma_max, t_g
        self.ref_ratio = ref_ratio
        # class-conditional matching needs a categorical target (classification)
        self.target_index = target_index if ref_ratio is not None else None
        self.rng = torch.Generator().manual_seed(seed)
        self.device = device

    def active(self, i: int) -> bool:
        """Guidance is applied every ``t_g`` steps of the reverse trajectory (i = T-1, ..., 0)."""
        return (self.T - i) % self.t_g == 0

    def gamma(self, i: int) -> float:
        return gamma_schedule(self.schedule, self.gamma_min, self.gamma_max, i / max(self.T - 1, 1))

    def reset(self):
        """Draw the reference batch x_r for a new generation."""
        n = len(self.x_real)
        if self.ref_ratio is None:
            idx = torch.arange(n)
        elif self.target_index is None:
            idx = torch.randperm(n, generator=self.rng)[: max(1, int(self.ref_ratio * n))]
        else:  # the same fraction of every class
            y = self.x_real[:, self.target_index].long().cpu()
            idx = torch.cat([
                c_idx[torch.randperm(len(c_idx), generator=self.rng)[: max(1, int(self.ref_ratio * len(c_idx)))]]
                for c_idx in (torch.nonzero(y == c).flatten() for c in y.unique())
            ])
        ref = self.x_real[idx.to(self.x_real.device)]
        self.ref_feats = embed(self.phi, ref, self.device)
        self.ref_labels = ref[:, self.target_index].long() if self.target_index is not None else None

    def gradient(self, x_noisy, x0_num, x0_cat):
        """grad of C w.r.t. the noisy numerical input; ``x0_num`` must depend on ``x_noisy``."""
        x0 = torch.cat([x0_num, x0_cat.to(x0_num.dtype)], dim=1)
        feats = self.phi(x0)
        if self.ref_labels is None:
            loss = chamfer(feats, self.ref_feats)
        else:
            labels = x0[:, self.target_index].long()
            terms = [chamfer(feats[labels == c], self.ref_feats[self.ref_labels == c])
                     for c in labels.unique() if (self.ref_labels == c).any()]
            if not terms:
                return torch.zeros_like(x_noisy)
            loss = torch.stack(terms).mean()
        return torch.autograd.grad(loss, x_noisy)[0]
