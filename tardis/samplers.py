"""Stage III: post-hoc selection of K * N_r rows from the candidate pool (Sec. 3.3, App. D).

Every sampler returns indices into the pool. ``P`` and ``R`` are the pool and D_real in
phi-space; ``y_pool`` / ``y_real`` are their targets in model space (class code, or the
quantile-normalised value for regression).
"""

import math

import numpy as np
import torch

from .chamfer import nn_dist

# Chamfer Sampler batch size b = |D_cand| / N_BATCHES (the same b is HDBSCAN's min_cluster_size)
N_BATCHES = 10


def chamfer_sampler(P, R, n_out, **_):
    """Rank pool batches B_k by C(B_k, D_real) (Eq. 5) and keep the best ones until n_out rows."""
    b = math.ceil(len(P) / N_BATCHES)
    fid_all, _ = nn_dist(P, R)  # m_sr: synthetic-to-real, per candidate
    scores = []
    for s in range(0, len(P), b):
        fidelity = fid_all[s:s + b].mean()
        coverage = nn_dist(R, P[s:s + b])[0].mean()  # m_rs: real-to-synthetic
        scores.append((fidelity + coverage).item())
    order = np.argsort(scores, kind="stable")
    idx = np.concatenate([np.arange(k * b, min((k + 1) * b, len(P))) for k in order])
    return idx[:n_out]


def md_sampler(P, R, n_out, k=10, **_):
    """Keep the candidates closest to the k-NN-hypersphere manifold of D_real (Eq. 15)."""
    k = min(k, len(R) - 1)
    rows = max(1, 2 ** 24 // len(R))
    radius = torch.cat([torch.cdist(R[s:s + rows], R).topk(k + 1, largest=False).values[:, -1]
                        for s in range(0, len(R), rows)])
    d_md, _ = nn_dist(P, R, offset=radius)  # min_r ||x - r|| - rho_k(r)
    return np.argsort(d_md.clamp(min=0).cpu().numpy(), kind="stable")[:n_out]


def stratified_sampler(y_pool, y_real, n_out, task, rng, n_bins=10, **_):
    """Sample within target strata so as to match the class (or target-bin) frequencies of D_real."""
    if task == "regression":
        edges = np.unique(np.quantile(y_real, np.linspace(0, 1, n_bins + 1)[1:-1]))
        s_pool, s_real = np.digitize(y_pool, edges), np.digitize(y_real, edges)
    else:
        s_pool, s_real = y_pool.astype(int), y_real.astype(int)
    strata, counts = np.unique(s_real, return_counts=True)
    quota = np.floor(n_out * counts / counts.sum()).astype(int)
    quota[np.argsort(-counts)[: n_out - quota.sum()]] += 1  # largest strata absorb rounding
    idx = []
    for s, q in zip(strata, quota):
        members = np.flatnonzero(s_pool == s)
        if len(members) and q:
            idx.append(rng.choice(members, q, replace=len(members) < q))
    idx = np.concatenate(idx) if idx else np.empty(0, dtype=int)
    if len(idx) < n_out:  # strata absent from the pool
        idx = np.concatenate([idx, rng.choice(len(y_pool), n_out - len(idx), replace=False)])
    return rng.permutation(idx)


def iboss_sampler(X_num, n_out, rng, **_):
    """IBOSS (Wang et al., 2019): for each numerical feature, take the most extreme
    remaining candidates on both sides; target-agnostic and independent of D_real."""
    n, p = X_num.shape
    if p == 0:
        return rng.choice(n, n_out, replace=False)
    r = max(1, n_out // (2 * p))
    available = np.ones(n, dtype=bool)
    for j in range(p):
        cand = np.flatnonzero(available)
        if n - len(cand) >= n_out:
            break
        order = cand[np.argsort(X_num[cand, j], kind="stable")]
        available[order[:r]] = available[order[-r:]] = False
    idx = np.flatnonzero(~available)[:n_out]
    if len(idx) < n_out:
        idx = np.concatenate([idx, rng.choice(np.flatnonzero(available), n_out - len(idx), replace=False)])
    return idx


def hdbscan_sampler(P, n_out, rng, max_fit=20000, **_):
    """Cluster the pool in phi-space with HDBSCAN (min_cluster_size = b) and keep whole
    clusters in decreasing order of density until n_out rows; clusters are chosen
    without reference to D_real.

    HDBSCAN is fitted on at most ``max_fit`` random candidates (the remaining ones join
    the cluster of their nearest fitted candidate). Cluster density is size over mean
    distance to the centroid.
    """
    from sklearn.cluster import HDBSCAN

    n = len(P)
    fit = rng.choice(n, min(n, max_fit), replace=False)
    b = max(5, round(math.ceil(n / N_BATCHES) * len(fit) / n))
    labels_fit = HDBSCAN(min_cluster_size=b).fit_predict(P[fit].cpu().numpy())
    _, nearest = nn_dist(P, P[fit])
    labels = labels_fit[nearest.cpu().numpy()]

    clusters = [c for c in np.unique(labels) if c >= 0]
    density = []
    for c in clusters:
        members = P[torch.as_tensor(np.flatnonzero(labels == c), device=P.device)]
        spread = (members - members.mean(0)).norm(dim=1).mean().item()
        density.append(len(members) / (spread + 1e-12))
    idx = [np.flatnonzero(labels == clusters[i]) for i in np.argsort(density)[::-1]]
    idx.append(rng.permutation(np.flatnonzero(labels < 0)))  # noise last
    return np.concatenate(idx)[:n_out]


SAMPLER_FNS = {
    "chamfer": chamfer_sampler,
    "md": md_sampler,
    "stratified": stratified_sampler,
    "iboss": iboss_sampler,
    "hdbscan": hdbscan_sampler,
}
