"""Tabular beta-VAE (Eq. 3) whose encoder serves as the representation map phi."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig


def beta_at(epoch: int, cfg: VAEConfig) -> float:
    """KL weight beta(t) of App. C.2, with t the epoch and T_VAE the number of epochs."""
    t, T, lo, hi = epoch, cfg.epochs, cfg.beta_min, cfg.beta_max
    if cfg.beta_schedule == "constant":
        return hi
    if cfg.beta_schedule == "linear":
        return hi * min(1.0, t / T)
    if cfg.beta_schedule == "cosine":
        return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * t / T))
    if cfg.beta_schedule == "sine":
        return hi - 0.5 * (hi - lo) * (1 + math.cos(math.pi * t / T))
    if cfg.beta_schedule == "cyclical":  # Fu et al. (2019)
        return hi * (t % cfg.cycle) / cfg.cycle
    raise ValueError(f"unknown beta schedule {cfg.beta_schedule!r}")


class BetaVAE(nn.Module):
    """MLP encoder/decoder on ``[numerical | one-hot(categorical)]`` rows."""

    def __init__(self, d_num, categories, cfg: VAEConfig):
        super().__init__()
        self.d_num, self.categories = d_num, list(categories)
        d_in = d_num + sum(categories)
        step = max((d_in - cfg.latent_dim) // max(cfg.n_layers, 1), 1)
        hidden = [max(d_in - step * (i + 1), cfg.latent_dim + 1) for i in range(cfg.n_layers)]

        def mlp(dims):
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(cfg.dropout)]
            return nn.Sequential(*layers)

        self.encoder = mlp([d_in] + hidden)
        self.mu = nn.Linear(hidden[-1], cfg.latent_dim)
        self.logvar = nn.Linear(hidden[-1], cfg.latent_dim)
        self.decoder = nn.Sequential(mlp([cfg.latent_dim] + hidden[::-1]), nn.Linear(hidden[0], d_in))

    def encode(self, x):
        return self.mu(self.encoder(x))

    def loss(self, x, beta):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        out = self.decoder(z)
        # reconstruction: MSE on numerical columns + cross-entropy on categorical groups,
        # each averaged within its group so that both feature types weigh the same
        terms = []
        if self.d_num:
            terms.append(F.mse_loss(out[:, : self.d_num], x[:, : self.d_num]))
        if self.categories:
            off, ce = self.d_num, []
            for k in self.categories:
                ce.append(F.cross_entropy(out[:, off:off + k], x[:, off:off + k].argmax(1)))
                off += k
            terms.append(torch.stack(ce).mean())
        rec = sum(terms)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return rec + beta * kl


def train_vae(x_onehot: torch.Tensor, d_num, categories, cfg: VAEConfig, device, seed=0) -> BetaVAE:
    torch.manual_seed(seed)
    np.random.seed(seed)
    vae = BetaVAE(d_num, categories, cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.lr)
    x = x_onehot.to(device)
    vae.train()
    for epoch in range(cfg.epochs):
        beta = beta_at(epoch, cfg)
        for idx in torch.randperm(len(x), device=device).split(cfg.batch_size):
            if len(idx) < 2:  # BatchNorm needs two rows
                continue
            opt.zero_grad(set_to_none=True)
            vae.loss(x[idx], beta).backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
    return vae.eval()
