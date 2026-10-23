"""The bidirectional Chamfer functional (Eq. 2) and the representation maps phi."""

import torch
import torch.nn as nn

# pairwise distances are computed in blocks of at most this many entries
_BLOCK = 2 ** 24


@torch.no_grad()
def nn_dist(a: torch.Tensor, b: torch.Tensor, offset: torch.Tensor = None):
    """For each row of ``a``: distance to (and index of) its nearest row of ``b``.

    With ``offset`` (one value per row of ``b``) the quantity minimised is
    ``||a_i - b_j|| - offset_j`` instead of the plain distance.
    """
    rows = max(1, _BLOCK // max(len(b), 1))
    dist = torch.empty(len(a), device=a.device)
    idx = torch.empty(len(a), dtype=torch.long, device=a.device)
    for s in range(0, len(a), rows):
        d = torch.cdist(a[s:s + rows], b)
        if offset is not None:
            d = d - offset[None]
        dist[s:s + rows], idx[s:s + rows] = d.min(dim=1)
    return dist, idx


def chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C(A, B) = mean_a min_b ||a - b|| + mean_b min_a ||a - b||   (fidelity + coverage).

    Differentiable w.r.t. both arguments: nearest neighbours are found without gradient,
    then the distances of the matched pairs are recomputed with gradient, which gives the
    exact (sub)gradient of the min with O(|A| + |B|) memory.
    """
    _, i_ab = nn_dist(a.detach(), b.detach())
    _, i_ba = nn_dist(b.detach(), a.detach())
    fidelity = _norm(a - b[i_ab]).mean()
    coverage = _norm(b - a[i_ba]).mean()
    return fidelity + coverage


def _norm(x):
    return (x.pow(2).sum(dim=1) + 1e-12).sqrt()


class IdentityMap(nn.Module):
    """phi = identity on z-score-normalised features (one-hot for categorical columns)."""

    def __init__(self, space, x_real: torch.Tensor):
        super().__init__()
        self.space = space
        z = space.onehot(x_real)
        self.register_buffer("mean", z.mean(dim=0))
        self.register_buffer("std", z.std(dim=0) + 1e-6)

    def forward(self, x):
        return (self.space.onehot(x) - self.mean) / self.std


class VAEMap(nn.Module):
    """phi = posterior mean of a beta-VAE encoder trained on D_real."""

    def __init__(self, space, vae):
        super().__init__()
        self.space, self.vae = space, vae.eval().requires_grad_(False)

    def forward(self, x):
        return self.vae.encode(self.space.onehot(x))


@torch.no_grad()
def embed(phi: nn.Module, x: torch.Tensor, device, batch_size=65536) -> torch.Tensor:
    """phi applied to a (possibly large) matrix in batches."""
    return torch.cat([phi(x[s:s + batch_size].to(device)) for s in range(0, len(x), batch_size)])
