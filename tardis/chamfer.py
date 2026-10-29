"""The bidirectional Chamfer functional (Eq. 2), its Stage II form, and the maps phi."""

import torch
import torch.nn as nn

# pairwise distances are computed in blocks of at most this many entries
_BLOCK = 2 ** 24


@torch.no_grad()
def nn_dist(a: torch.Tensor, b: torch.Tensor):
    """For each row of ``a``: Euclidean distance to (and index of) its nearest row of ``b``."""
    rows = max(1, _BLOCK // max(len(b), 1))
    dist = torch.empty(len(a), device=a.device)
    idx = torch.empty(len(a), dtype=torch.long, device=a.device)
    for s in range(0, len(a), rows):
        dist[s:s + rows], idx[s:s + rows] = torch.cdist(a[s:s + rows], b).min(dim=1)
    return dist, idx


@torch.no_grad()
def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """C(A, B) = mean_a min_b ||a - b|| + mean_b min_a ||a - b||   (Eq. 2, fidelity + coverage)."""
    return (nn_dist(a, b)[0].mean() + nn_dist(b, a)[0].mean()).item()


def chamfer_guidance(real: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
    """Stage II form of the functional: squared Euclidean distances scaled by 1/(4d) (App. C.1).

    Differentiable w.r.t. ``gen``: nearest neighbours are found without gradient, then the
    squared distances of the matched pairs are recomputed with gradient, which gives the
    exact (sub)gradient of the min.
    """
    _, i_rg = nn_dist(real.detach(), gen.detach())
    _, i_gr = nn_dist(gen.detach(), real.detach())
    coverage = (real - gen[i_rg]).pow(2).sum(dim=1).mean()
    fidelity = (gen - real[i_gr]).pow(2).sum(dim=1).mean()
    return (coverage + fidelity) / (4 * real.shape[1])


class IdentityMap(nn.Module):
    """phi = identity on the backbone's numerical encoding, z-scored with the statistics of
    the set it is applied to."""

    def forward(self, x):
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)


class VAEMap(nn.Module):
    """phi = posterior mean of a beta-VAE encoder trained on D_real."""

    def __init__(self, space, vae):
        super().__init__()
        self.space, self.vae = space, vae.eval().requires_grad_(False)

    def forward(self, x):
        return self.vae.encode(self.space.onehot(x))


@torch.no_grad()
def embed(phi: nn.Module, x: torch.Tensor, device, batch_size=65536) -> torch.Tensor:
    """A row-wise map phi (e.g. :class:`VAEMap`) applied to a large matrix in batches."""
    return torch.cat([phi(x[s:s + batch_size].to(device)) for s in range(0, len(x), batch_size)])
