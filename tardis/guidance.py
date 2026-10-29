"""Stage II: bidirectional Chamfer guidance during reverse diffusion (Sec. 3.2, App. C)."""

import math
from typing import Optional

import torch
from sklearn.cluster import KMeans

from .chamfer import chamfer_guidance
from .config import GAMMA_CONSTANT, TARGET_CLUSTERS


def gamma_schedule(kind: str, g_min: float, g_max: float, s: int, T: int) -> float:
    """Gamma at the s-th step of the reverse trajectory (s = 1, ..., T), i.e. at t = T - s in
    the convention of App. C.1 (t = T is pure noise, t = 0 the final sample)."""
    if kind == "constant":
        return GAMMA_CONSTANT
    if kind == "linear":
        return g_max + (g_min - g_max) * s / T
    if kind == "cosine":
        return g_min + 0.5 * (g_max - g_min) * (1 + math.cos(math.pi * s / T))
    if kind == "sine":
        return g_max - 0.5 * (g_max - g_min) * (1 + math.cos(math.pi * s / T))
    raise ValueError(f"unknown guidance schedule {kind!r}")


class ChamferGuidance:
    """Score perturbation  eps~ = eps - Gamma(t) * grad_{x_t} C(x_t, x_r)   (Eq. 4).

    The functional is evaluated on the backbone's prediction of the clean sample and
    differentiated w.r.t. the noisy numerical coordinates. The reference set x_r is D_real,
    held fixed across diffusion steps, and is matched as in App. C.3: globally for ratio
    None or 1, by class (classification) or by k-means cluster of the target (regression)
    for a ratio r < 1.
    """

    def __init__(self, phi, x_real: torch.Tensor, num_timesteps: int, schedule="constant",
                 gamma_min=1e-6, gamma_max=0.5, t_g=5, ref_ratio: Optional[float] = None,
                 target_index: int = 0, task="binclass", n_classes=0, n_clusters=TARGET_CLUSTERS,
                 device="cpu", seed=0):
        self.phi = phi.to(device)
        self.T = num_timesteps
        self.schedule, self.gamma_min, self.gamma_max, self.t_g = schedule, gamma_min, gamma_max, t_g
        self.ratio, self.ti, self.task, self.n_classes = ref_ratio, target_index, task, n_classes
        x_real = x_real.to(device)
        with torch.no_grad():
            self.real_feats = self.phi(x_real)
        self.conditional = ref_ratio is not None and ref_ratio < 1
        if self.conditional:
            y = x_real[:, target_index]
            if task == "regression":
                self.kmeans = KMeans(n_clusters, random_state=seed, n_init=10).fit(y.cpu().numpy()[:, None])
                self.real_labels = torch.as_tensor(self.kmeans.labels_, device=device).long()
            else:
                self.real_labels = y.round().long()

    def active(self, i: int) -> bool:
        """Guidance is applied every ``t_g`` steps of the reverse trajectory (i = T-1, ..., 0)."""
        return (self.T - i) % self.t_g == 0

    def gamma(self, i: int) -> float:
        return gamma_schedule(self.schedule, self.gamma_min, self.gamma_max, self.T - i, self.T)

    def gradient(self, x_noisy, x0_num, x0_cat):
        """grad of C w.r.t. the noisy numerical input; ``x0_num`` must depend on ``x_noisy``."""
        x0 = torch.cat([x0_num, x0_cat.to(x0_num.dtype)], dim=1)
        feats = self.phi(x0)
        pairs = self._class_pairs(x0, feats) if self.conditional else self._global_pairs(feats)
        if not pairs:
            return torch.zeros_like(x_noisy)
        loss = torch.stack([chamfer_guidance(r, g) for r, g in pairs]).mean()
        return torch.autograd.grad(loss, x_noisy)[0]

    def _global_pairs(self, gen):
        """The whole of D_real against the candidates; the larger set is split into chunks
        the size of the smaller one when their sizes differ by more than 10%."""
        real = self.real_feats
        n_r, n_g = len(real), len(gen)
        if abs(n_r - n_g) < 0.1 * max(n_r, n_g):
            return [(real, gen)]
        if n_r > n_g:
            return [(chunk, gen) for chunk in real.split(n_g)]
        return [(real, chunk) for chunk in gen.split(n_r)]

    def _class_pairs(self, x0, gen):
        """Conditional matching (App. C.3)."""
        target = x0[:, self.ti]
        if self.task == "regression":
            labels = self.kmeans.predict(target.detach().cpu().numpy()[:, None])
            gen_labels = torch.as_tensor(labels, device=gen.device).long()
        else:
            gen_labels = target.round().clamp(0, self.n_classes - 1).long()
        pairs = []
        for c in torch.unique(torch.cat([self.real_labels, gen_labels])).tolist():
            real_c, gen_c = self.real_feats[self.real_labels == c], gen[gen_labels == c]
            size = int(len(real_c) * self.ratio)
            if len(real_c) == 0 or len(gen_c) == 0 or size == 0:
                continue
            if len(real_c) > len(gen_c):
                pairs += [(chunk, gen_c) for chunk in real_c.split(size)]
            else:
                pairs = [(real_c, chunk) for chunk in gen_c.split(size)]
            pairs.append((real_c, gen_c))
        return pairs
