"""Tabular beta-VAE (Eq. 3) whose encoder serves as the representation map phi."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig


def beta_at(epoch: int, cfg: VAEConfig) -> float:
    """KL weight beta(t) of App. C.2 at epoch t = 1, 2, ...: non-constant schedules run over
    W = ``cfg.warmup`` epochs and then stay at beta_max (W = 0 means constant); the cyclical
    schedule repeats every T_cyc epochs."""
    t, W, lo, hi = epoch, cfg.warmup, cfg.beta_min, cfg.beta_max
    if cfg.beta_schedule == "constant" or W == 0:
        return hi
    if cfg.beta_schedule == "cyclical":  # Fu et al. (2019)
        return hi * (t % cfg.cycle) / cfg.cycle
    if t >= W:
        return hi
    if cfg.beta_schedule == "linear":
        return hi * t / W
    if cfg.beta_schedule == "cosine":
        return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * t / W))
    if cfg.beta_schedule == "sine":
        return hi - 0.5 * (hi - lo) * (1 + math.cos(math.pi * t / W))
    raise ValueError(f"unknown beta schedule {cfg.beta_schedule!r}")


class _MLP(nn.Module):
    """Linear -> [BatchNorm] -> ReLU -> Dropout blocks, with optional residual connections."""

    def __init__(self, dims, cfg: VAEConfig):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            layers = [nn.Linear(a, b)] + ([nn.BatchNorm1d(b)] if cfg.batch_norm else [])
            self.blocks.append(nn.Sequential(*layers, nn.ReLU(), nn.Dropout(cfg.dropout)))
            if cfg.skip_connections:
                self.skips.append(nn.Linear(a, b) if a != b else nn.Identity())

    def forward(self, h):
        for i, block in enumerate(self.blocks):
            h = block(h) + self.skips[i](h) if self.skips else block(h)
        return h


class BetaVAE(nn.Module):
    """Encoder and decoder on ``[numerical | one-hot(categorical)]`` rows; categorical
    inputs are embedded (dimension min(k, 8)) before the encoder MLP."""

    def __init__(self, d_num, categories, cfg: VAEConfig):
        super().__init__()
        self.d_num, self.categories = d_num, list(categories)
        d_in = d_num + sum(categories)
        step = max((d_in - cfg.latent_dim) // max(cfg.n_layers, 1), 1)
        hidden = [max(d_in - step * (i + 1), cfg.latent_dim + 1) for i in range(cfg.n_layers)]
        self.embeddings = nn.ModuleList(nn.Embedding(k, min(k, cfg.embedding_dim)) for k in categories)
        d_emb = d_num + sum(min(k, cfg.embedding_dim) for k in categories)
        self.encoder = _MLP([d_emb] + hidden, cfg)
        self.mu = nn.Linear(hidden[-1], cfg.latent_dim)
        self.logvar = nn.Linear(hidden[-1], cfg.latent_dim)
        self.decoder = _MLP([cfg.latent_dim] + hidden[::-1], cfg)
        self.num_head = nn.Linear(hidden[0], d_num) if d_num else None
        self.cat_heads = nn.ModuleList(nn.Linear(hidden[0], k) for k in categories)

    def _groups(self, x):
        off = self.d_num
        for k in self.categories:
            yield x[:, off:off + k]
            off += k

    def _encode(self, x):
        parts = [x[:, : self.d_num]] + [emb(g.argmax(1)) for emb, g in zip(self.embeddings, self._groups(x))]
        h = self.encoder(torch.cat(parts, dim=1))
        return self.mu(h), self.logvar(h)

    def encode(self, x):
        return self._encode(x)[0]

    def loss(self, x, beta):
        mu, logvar = self._encode(x)
        h = self.decoder(mu + torch.randn_like(mu) * (0.5 * logvar).exp())
        # reconstruction: MSE averaged over numerical columns + cross-entropy averaged over
        # categorical groups, so that both feature types weigh the same
        rec = torch.zeros(len(x), device=x.device)
        if self.d_num:
            rec = rec + F.mse_loss(self.num_head(h), x[:, : self.d_num], reduction="none").mean(dim=1)
        if self.categories:
            ce = [F.cross_entropy(head(h), g.argmax(1), reduction="none") for head, g in zip(self.cat_heads, self._groups(x))]
            rec = rec + torch.stack(ce).mean(dim=0)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).clamp(min=0)
        return rec.mean() + beta * kl.mean()


def train_vae(x_onehot: torch.Tensor, d_num, categories, cfg: VAEConfig, device, seed=0, val_ratio=0.15) -> BetaVAE:
    """Adam, ReduceLROnPlateau (factor 0.5, patience 10) on a 15% validation split, gradient
    clipping at 1.0; the weights of the last epoch are returned."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vae = BetaVAE(d_num, categories, cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.lr)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    x = x_onehot.to(device)
    perm = torch.as_tensor(rng.permutation(len(x)), device=device)
    n_val = int(len(x) * val_ratio)
    x_val, x_tr = x[perm[:n_val]], x[perm[n_val:]]
    for epoch in range(1, cfg.epochs + 1):
        beta = beta_at(epoch, cfg)
        vae.train()
        for idx in torch.randperm(len(x_tr), device=device).split(cfg.batch_size):
            if len(idx) < 2:  # BatchNorm needs two rows
                continue
            opt.zero_grad(set_to_none=True)
            vae.loss(x_tr[idx], beta).backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
        if n_val:
            vae.eval()
            with torch.no_grad():
                plateau.step(vae.loss(x_val, beta).item())
    return vae.eval()
