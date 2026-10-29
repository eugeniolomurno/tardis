"""Stage III: post-hoc selection of K * N_r rows from the candidate pool (Sec. 3.3, App. D).

Every sampler returns indices into the pool. The arrays each one needs are prepared by
:meth:`tardis.TARDIS.sampler_inputs`.
"""

import numpy as np
import pandas as pd
import torch

from .chamfer import chamfer

N_BATCHES = 10  # the Chamfer Sampler splits the pool into ten batches (App. D.1)


def chamfer_sampler(P, R, n_out, rng, **_):
    """Rank pool batches B_k by C(B_k, D_real) (Eq. 5) and keep the best ones until n_out rows.

    ``P`` / ``R``: candidate and real features, one-hot encoded (target excluded) and z-scored
    with the statistics of D_real and the pool together. Batches follow the generation order;
    the last batch contributes its first rows.
    """
    n = len(P)
    b = max(1, n // N_BATCHES)
    starts = range(0, n, b)
    order = np.argsort([chamfer(P[s:s + b], R) for s in starts], kind="stable")
    chosen = []
    for k in order:
        batch = np.arange(starts[k], min(starts[k] + b, n))
        chosen.extend(batch[: n_out - len(chosen)])
        if len(chosen) >= n_out:
            break
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) < n_out:
        rest = np.setdiff1d(np.arange(n), chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, n_out - len(chosen), replace=False)])
    return chosen


def md_sampler(P, R, n_out, k=10, **_):
    """Keep the candidates closest to the k-NN-hypersphere manifold of D_real (Eq. 15):
    d(x) = max(0, ||x - r*||^2 - rho_k(r*)^2), with r* the real record nearest to x and
    rho_k(r) the distance from r to its k-th nearest real record (r itself included).

    ``P`` / ``R``: one-hot encoded records with the target, or beta-VAE posterior means.
    """
    rows = max(1, 2 ** 24 // len(R))
    k = min(k, len(R))
    radius = torch.cat([torch.cdist(R[s:s + rows], R).pow(2).topk(k, largest=False).values[:, -1]
                        for s in range(0, len(R), rows)])
    d_md = torch.empty(len(P), device=P.device)
    for s in range(0, len(P), rows):
        d, nearest = torch.cdist(P[s:s + rows], R).pow(2).min(dim=1)
        d_md[s:s + rows] = (d - radius[nearest]).clamp(min=0)
    return np.argsort(d_md.cpu().numpy(), kind="stable")[:n_out]


def stratified_sampler(y_pool, n_out, task, rng, n_bins=10, **_):
    """Draw the same number of candidates from every class (with replacement for classes
    smaller than their share) or, for regression, from every decile of the candidates'
    target (without replacement)."""
    y = pd.Series(y_pool)
    if task == "regression":
        strata = pd.qcut(y, n_bins, duplicates="drop") if y.nunique() > n_bins else y
    else:
        strata = y.astype(int)
    groups = [np.flatnonzero((strata == s).to_numpy()) for s in strata.unique()]
    share = n_out // len(groups)
    idx = []
    for members in groups:
        if task == "regression":
            idx.append(rng.choice(members, min(len(members), share), replace=False))
        else:
            idx.append(rng.choice(members, share, replace=len(members) < share))
    return rng.permutation(np.concatenate(idx))


def iboss_sampler(X, n_out, rng, **_):
    """IBOSS (Wang et al., 2019): for each of the p numerical features keep the n_out/(2p)
    candidates with the smallest and the largest values; categorical features then add an
    equal share of candidates per category; random candidates fill any remainder.

    ``X``: candidate features (target excluded) as a DataFrame.
    """
    n = len(X)
    num = [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]
    cat = [c for c in X.columns if c not in num]
    selected = set()
    if num:
        r = min(max(1, n_out // (2 * len(num))), n - 1)
        for c in num:
            values = X[c].to_numpy()
            selected.update(np.argpartition(values, r)[:r])
            selected.update(np.argpartition(values, -r)[-r:])
    remaining = n_out - len(selected)
    if remaining > 0:
        for c in cat:
            if len(selected) >= n_out:
                break
            counts = X[c].value_counts()
            share = max(1, remaining // len(counts))
            for level in counts.index:
                if len(selected) >= n_out:
                    break
                selected.update(np.flatnonzero((X[c] == level).to_numpy())[:share])
    selected = np.asarray(sorted(selected), dtype=int)
    if len(selected) < n_out:
        rest = np.setdiff1d(np.arange(n), selected)
        selected = np.sort(np.concatenate([selected, rng.choice(rest, n_out - len(selected), replace=False)]))
    return selected[:n_out]


SAMPLER_FNS = {
    "chamfer": chamfer_sampler,
    "md": md_sampler,
    "stratified": stratified_sampler,
    "iboss": iboss_sampler,
}

# samplers whose selection does not depend on the seed
DETERMINISTIC = ("chamfer", "md")
