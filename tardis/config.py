"""Hyperparameters. Every default is the value used in the paper."""

from dataclasses import asdict, dataclass, field
from typing import Optional

# Stage I pool multiplier: M * N_r candidates (M = 50 in all reported configurations, Sec. 3).
POOL_MULTIPLIER = 50

# Stage III grid (Sec. 3.3, App. D): five samplers x GKD on/off = 10 combinations per trial.
SAMPLERS = ("chamfer", "md", "stratified", "iboss", "hdbscan")

# Stage II search space (Table 7).
GUIDANCE_VARIANTS = ("none", "identity", "vae")
GAMMA_SCHEDULES = ("constant", "linear", "cosine", "sine")
REF_RATIOS = {"none": None, "1": 1.0, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125, "1/10": 0.1}
VAE_LATENT_DIMS = (4, 8, 16, 32)
BETA_SCHEDULES = ("constant", "linear", "cosine", "sine", "cyclical")


@dataclass
class BackboneConfig:
    """TabDiff backbone (Shi et al., ICLR 2025) with the official hyperparameters and
    learnable per-column noise schedules."""

    epochs: int = 8000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    ema_decay: float = 0.997
    lr_factor: float = 0.9  # ReduceLROnPlateau
    lr_patience: int = 50
    num_timesteps: int = 50
    num_layers: int = 2
    d_token: int = 4
    n_head: int = 1
    factor: int = 32
    dim_t: int = 1024
    sample_batch_size: int = 10000


@dataclass
class VAEConfig:
    """Tabular beta-VAE used as representation map phi (Eq. 3, App. C.2)."""

    latent_dim: int = 16
    beta_schedule: str = "constant"
    beta_max: float = 1.0
    beta_min: float = 0.5
    cycle: int = 10  # T_cyc of the cyclical schedule
    n_layers: int = 3
    dropout: float = 0.1
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 1024


@dataclass
class TardisConfig:
    """One point of the TARDIS search space.

    The defaults are the fixed global configuration of App. J.3
    (Chamfer Sampler at K = 10, no GKD, no guidance)."""

    K: float = 10.0  # cardinality multiplier: the output has K * N_r rows
    # Stage II
    guidance: str = "none"  # none | identity | vae
    schedule: str = "constant"  # Gamma(t) family (App. C.1)
    gamma_max: float = 0.5
    gamma_min: float = 1e-6
    t_g: int = 5  # guidance is applied every t_g reverse-diffusion steps
    ref_ratio: Optional[float] = None  # reference batch fraction; None = full D_real, global
    vae: VAEConfig = field(default_factory=VAEConfig)
    # Stage III
    sampler: str = "chamfer"
    gkd: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TardisConfig":
        d = dict(d)
        d["vae"] = VAEConfig(**d.get("vae", {}))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
