"""Pre-trained backbone: TabDiff, trained once per dataset and then kept frozen."""

import copy
import logging
from pathlib import Path

import numpy as np
import torch

from ..config import BackboneConfig
from .diffusion import TabDiff
from .network import Model, UniModMLP

log = logging.getLogger("tardis")


def build_tabdiff(d_num, categories, cfg: BackboneConfig) -> TabDiff:
    net = UniModMLP(
        d_numerical=d_num,
        categories=[k + 1 for k in categories],  # + [MASK]
        num_layers=cfg.num_layers,
        d_token=cfg.d_token,
        n_head=cfg.n_head,
        factor=cfg.factor,
        dim_t=cfg.dim_t,
    )
    return TabDiff(Model(net), d_num, categories, num_timesteps=cfg.num_timesteps)


def train_tabdiff(x: torch.Tensor, d_num, categories, cfg: BackboneConfig, device, seed=0) -> TabDiff:
    """Train TabDiff on the model-space training matrix ``x`` and return the EMA model.

    Follows the official schedule: AdamW, ReduceLROnPlateau on the epoch loss, continuous
    loss weight annealed linearly to 0, EMA of the weights updated once per epoch. The
    returned EMA weights are those with the lowest training loss over the second half
    of training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_tabdiff(d_num, categories, cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)
    use_amp = torch.device(device).type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    x = x.to(device)
    best, best_state = float("inf"), None

    for epoch in range(cfg.epochs):
        model.train()
        c_weight = 1.0 - epoch / cfg.epochs  # "anneal" continuous-loss weight
        tot_d = tot_c = 0.0
        for idx in torch.randperm(len(x), device=device).split(cfg.batch_size):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                d_loss, c_loss = model.mixed_loss(x[idx])
                loss = d_loss + c_weight * c_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_d += d_loss.item() * len(idx)
            tot_c += c_loss.item() * len(idx)
        epoch_loss = (tot_d + tot_c) / len(x)
        if not np.isfinite(epoch_loss):
            raise RuntimeError(f"TabDiff loss diverged at epoch {epoch + 1}")
        plateau.step(epoch_loss)
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.mul_(cfg.ema_decay).add_(p.detach(), alpha=1 - cfg.ema_decay)
        if epoch >= cfg.epochs // 2:
            ema_loss = _full_loss(ema, x, cfg.batch_size)
            if ema_loss < best:
                best, best_state = ema_loss, copy.deepcopy(ema.state_dict())
        if (epoch + 1) % max(1, cfg.epochs // 20) == 0:
            log.info(f"TabDiff epoch {epoch + 1}/{cfg.epochs}  loss={epoch_loss:.4f}")

    if best_state is not None:
        ema.load_state_dict(best_state)
    return ema.eval()


@torch.no_grad()
def _full_loss(model, x, batch_size):
    model.eval()
    tot = 0.0
    for idx in torch.arange(len(x), device=x.device).split(batch_size):
        d_loss, c_loss = model.mixed_loss(x[idx])
        tot += (d_loss + c_loss).item() * len(idx)
    return tot / len(x)


def save_tabdiff(model: TabDiff, cfg: BackboneConfig, path):
    torch.save({"state_dict": model.state_dict(), "d_num": model.d_num,
                "categories": model.num_classes.tolist(), "config": vars(cfg)}, path)


def load_tabdiff(path, device) -> TabDiff:
    ckpt = torch.load(Path(path), map_location=device)
    model = build_tabdiff(ckpt["d_num"], ckpt["categories"], BackboneConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
