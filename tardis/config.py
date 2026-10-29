"""Hyperparameters. Every default is the value used in the paper."""

from dataclasses import asdict, dataclass, field
from typing import Optional

# Base seed: data split, backbone training, candidate pools and the search.
SEED = 420

# Stage I pool multiplier: M * N_r candidates (M = 50 in all reported configurations, Sec. 3).
POOL_MULTIPLIER = 50

# Stage III grid (Sec. 3.3, App. D): four samplers x GKD on/off = 8 combinations per trial.
SAMPLERS = ("chamfer", "md", "stratified", "iboss")

# Seeds averaged by every cell of the Stage III grid and by the report (Sec. 4).
EVAL_SEEDS = (0, 1, 2, 3, 4)

# Stage II search space (Table 7).
GUIDANCE_VARIANTS = ("none", "identity", "vae")
GAMMA_SCHEDULES = ("constant", "linear", "cosine", "sine")
GAMMA_CONSTANT = 0.05  # Gamma of the constant family (Eq. 6)
REF_RATIOS = {"none": None, "1": 1.0, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125, "1/10": 0.1}
TARGET_CLUSTERS = 5  # k-means clusters of the target for conditional matching in regression (App. C.3)

# beta-VAE inner study (Table 7b).
VAE_LATENT_DIMS = (4, 8, 16, 32)
BETA_SCHEDULES = ("constant", "linear", "cosine", "sine", "cyclical")
INNER_TRIALS = 20
INNER_SEEDS = (0, 1, 2, 4, 5)  # seeds of the inner-study objective


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
    """Tabular beta-VAE (Eq. 3). One point of the inner study of Table 7b; the defaults are
    used when no inner study runs."""

    latent_dim: int = 16
    n_layers: int = 3
    beta_max: float = 1.0
    beta_schedule: str = "constant"
    warmup: int = 0  # W: epochs over which a non-constant schedule runs (App. C.2)
    cycle: int = 10  # T_cyc of the cyclical schedule
    dropout: float = 0.1
    batch_norm: bool = True
    skip_connections: bool = False
    lr: float = 1e-3
    epochs: int = 100
    beta_min: float = 0.5
    batch_size: int = 1024
    embedding_dim: int = 8


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
    ref_ratio: Optional[float] = None  # None or 1: global matching; < 1: conditional matching (App. C.3)
    latent_sampler: bool = False  # MDSampler in the beta-VAE latent space
    vae: VAEConfig = field(default_factory=VAEConfig)
    # Stage III
    sampler: str = "chamfer"
    gkd: bool = False

    @property
    def needs_vae(self) -> bool:
        return self.guidance == "vae" or self.latent_sampler

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TardisConfig":
        d = dict(d)
        d["vae"] = VAEConfig(**d.get("vae", {}))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
