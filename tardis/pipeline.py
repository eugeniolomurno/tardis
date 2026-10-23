"""TARDIS: Tabular generation through Refinement, Distillation, and Inference-time Sampling.

Stage I   oversample M * N_r noise vectors from the frozen backbone's prior;
Stage II  denoise them, optionally with bidirectional Chamfer guidance (Eq. 4);
Stage III select K * N_r rows from the candidate pool with one of five samplers,
          optionally after relabelling the pool with GKD soft labels.
Stage II hyperparameters are searched with TPE; each trial's objective is the best
validation utility of the Stage III grid (5 samplers x GKD on/off).
"""

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import optuna
import pandas as pd
import torch

from .backbone import load_tabdiff, save_tabdiff, train_tabdiff
from .chamfer import IdentityMap, VAEMap, embed
from .config import (BETA_SCHEDULES, GAMMA_SCHEDULES, GUIDANCE_VARIANTS, POOL_MULTIPLIER, REF_RATIOS,
                     SAMPLERS, VAE_LATENT_DIMS, BackboneConfig, TardisConfig)
from .data import Dataset, ModelSpace
from .guidance import ChamferGuidance
from .learners import make_learner
from .metrics import METRIC_NAME, delta_percent, score, utility
from .samplers import SAMPLER_FNS
from .vae import train_vae

log = logging.getLogger("tardis")

PHI_SAMPLERS = ("chamfer", "md", "hdbscan")  # samplers that operate in phi-space


class TARDIS:
    def __init__(self, dataset: Dataset, workdir="runs/tardis", learner="xgboost", M: float = POOL_MULTIPLIER,
                 device: Optional[str] = None, seed: int = 0, backbone_config: Optional[BackboneConfig] = None):
        self.ds, self.space = dataset, ModelSpace(dataset)
        self.x_real = torch.from_numpy(self.space.encode(dataset.train))
        self.n_real = len(dataset.train)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.learner, self.M, self.seed = learner, M, seed
        self.backbone_config = backbone_config or BackboneConfig()
        self.backbone = None
        self._vaes, self._unguided, self._teacher = {}, {}, None

    # ------------------------------------------------------------------ backbone
    def fit_backbone(self):
        """Train TabDiff once (or load ``workdir/tabdiff.pt``); it stays frozen afterwards."""
        path = self.workdir / "tabdiff.pt"
        if path.exists():
            self.backbone = load_tabdiff(path, self.device)
            log.info(f"loaded backbone from {path}")
        else:
            log.info(f"training TabDiff for {self.backbone_config.epochs} epochs on {self.n_real} rows")
            self.backbone = train_tabdiff(self.x_real, self.space.d_num, self.space.categories,
                                          self.backbone_config, self.device, self.seed)
            save_tabdiff(self.backbone, self.backbone_config, path)
        self.backbone.requires_grad_(False)
        return self.backbone

    # --------------------------------------------------------- Stage I and II
    def representation(self, cfg: TardisConfig):
        """phi: the beta-VAE encoder for ``guidance='vae'``, the identity map otherwise."""
        if cfg.guidance != "vae":
            return IdentityMap(self.space, self.x_real)
        key = json.dumps(vars(cfg.vae), sort_keys=True)
        if key not in self._vaes:
            x = self.space.onehot(self.x_real)
            self._vaes[key] = train_vae(x, self.space.d_num, self.space.categories, cfg.vae, self.device, self.seed)
        return VAEMap(self.space, self._vaes[key])

    def candidate_pool(self, cfg: TardisConfig, seed: int) -> torch.Tensor:
        """D_cand: M * N_r rows in model space, generated with the Stage II setting of ``cfg``."""
        if self.backbone is None:
            self.fit_backbone()
        n = int(self.M * self.n_real)
        if cfg.guidance == "none" and seed in self._unguided:
            return self._unguided[seed]
        guidance = None
        if cfg.guidance != "none":
            guidance = ChamferGuidance(
                self.representation(cfg), self.x_real, self.backbone.num_timesteps,
                schedule=cfg.schedule, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max, t_g=cfg.t_g,
                ref_ratio=cfg.ref_ratio,
                target_index=self.space.target_index if self.ds.is_classification else None,
                device=self.device, seed=seed,
            )
        torch.manual_seed(seed)
        pool = self.backbone.sample_all(n, self.backbone_config.sample_batch_size, guidance)
        if cfg.guidance == "none":
            self._unguided[seed] = pool
        return pool

    # --------------------------------------------------------------- Stage III
    def select(self, pool: torch.Tensor, cfg: TardisConfig, seed: int, K: Optional[float] = None,
               sampler: Optional[str] = None, inputs: Optional[dict] = None) -> np.ndarray:
        """Indices of the K * N_r pool rows retained by ``sampler`` (default ``cfg.sampler``)."""
        sampler = sampler or cfg.sampler
        n_out = min(int(self.n_real * (cfg.K if K is None else K)), len(pool))
        inputs = inputs if inputs is not None else self._sampler_inputs(pool, cfg, sampler)
        rng = np.random.default_rng(seed)
        return SAMPLER_FNS[sampler](n_out=n_out, rng=rng, task=self.ds.task, **inputs)

    def _sampler_inputs(self, pool, cfg, sampler):
        """Arrays a sampler needs; phi-space features are inherited from Stage II."""
        ti = self.space.target_index
        if sampler in PHI_SAMPLERS:
            phi = self.representation(cfg).to(self.device)
            return {"P": embed(phi, pool, self.device), "R": embed(phi, self.x_real, self.device)}
        if sampler == "stratified":
            return {"y_pool": pool[:, ti].numpy(), "y_real": self.x_real[:, ti].numpy()}
        num = [j for j in range(self.space.d_num) if j != ti]  # iboss: numerical features, target excluded
        return {"X_num": pool[:, num].numpy()}

    # ------------------------------------------------- learner, GKD and utility
    def teacher(self):
        """f_GKD: the downstream learner trained on D_real (App. E)."""
        if self._teacher is None:
            self._teacher = self._fit(self.ds.X(self.ds.train), self.ds.y(self.ds.train))
        return self._teacher

    def _fit(self, X, y):
        learner = make_learner(self.learner, self.ds.task, self.ds.n_classes, self.seed, str(self.device))
        return learner.fit(X, y, self.ds.X(self.ds.val), self.ds.y(self.ds.val))

    def utility(self, X, y, split="val") -> float:
        """psi(A(D), D_split): train on (X, y), early-stop on validation, score on ``split``."""
        part = getattr(self.ds, split)
        return utility(self.ds.task, self.ds.y(part), self._fit(X, y).predict(self.ds.X(part)))

    def _pool_frame(self, pool):
        df = self.space.decode(pool.numpy())
        return self.ds.X(df), self.ds.y(df)

    # ------------------------------------------------------------------ search
    def search(self, n_trials: int = 500) -> TardisConfig:
        """TPE over the Stage II space (Table 7); each trial runs the Stage III grid.
        Trials are stored in ``workdir/search.db`` and a search can be resumed."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name="tardis", direction="maximize", load_if_exists=True,
            storage=f"sqlite:///{(self.workdir / 'search.db').resolve()}",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if n_trials > done:
            study.optimize(self._objective, n_trials=n_trials - done)
        best = TardisConfig.from_dict(study.best_trial.user_attrs["config"])
        (self.workdir / "best_config.json").write_text(json.dumps(best.to_dict(), indent=2))
        log.info(f"best validation {METRIC_NAME[self.ds.task]}: {abs(study.best_value):.4f}  config: {best}")
        return best

    def _objective(self, trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, k_max=min(25.0, self.M))  # BCR needs K <= M
        pool = self.candidate_pool(cfg, self.seed)
        X_pool, y_hard = self._pool_frame(pool)
        y_soft = self.teacher().predict(X_pool)
        phi_inputs = self._sampler_inputs(pool, cfg, "chamfer")  # shared by chamfer / md / hdbscan
        grid = {}
        for sampler in SAMPLERS:
            try:
                inputs = phi_inputs if sampler in PHI_SAMPLERS else self._sampler_inputs(pool, cfg, sampler)
                idx = self.select(pool, cfg, self.seed, sampler=sampler, inputs=inputs)
                for gkd in (False, True):
                    y = y_soft[idx] if gkd else y_hard[idx]
                    grid[(sampler, gkd)] = self.utility(X_pool.iloc[idx], y, "val")
            except Exception as e:  # a sampler failing must not kill the trial
                log.warning(f"trial {trial.number}: sampler {sampler} failed ({e})")
        if not grid:
            raise optuna.TrialPruned("every sampler failed")
        (cfg.sampler, cfg.gkd), best = max(grid.items(), key=lambda kv: score(self.ds.task, kv[1]))
        trial.set_user_attr("config", cfg.to_dict())
        trial.set_user_attr("grid", {f"{s}{'+gkd' if g else ''}": v for (s, g), v in grid.items()})
        log.info(f"trial {trial.number}: val {METRIC_NAME[self.ds.task]}={best:.4f}  "
                 f"K={cfg.K} guidance={cfg.guidance} sampler={cfg.sampler} gkd={cfg.gkd}")
        return score(self.ds.task, best)

    # ------------------------------------------------------------------- apply
    def generate(self, cfg: TardisConfig, seed: int = 0, K: Optional[float] = None) -> pd.DataFrame:
        """The synthetic dataset D_syn (K * N_r rows) in the original column space.

        With GKD, classification targets keep the hard label and gain the teacher's
        probabilities as ``<target>__p_<class>`` columns; regression targets are replaced by
        the teacher's prediction.
        """
        pool = self.candidate_pool(cfg, seed)
        idx = self.select(pool, cfg, seed, K)
        df = self.space.decode(pool[idx].numpy())
        if cfg.gkd:
            soft = self.teacher().predict(self.ds.X(df))
            t = self.ds.target
            if self.ds.task == "regression":
                df[t] = self.ds.y_inverse(soft)
            else:
                soft = np.stack([1 - soft, soft], 1) if soft.ndim == 1 else soft
                for j, c in enumerate(self.ds.classes):
                    df[f"{t}__p_{c}"] = soft[:, j]
        return df

    def report(self, cfg: TardisConfig, seeds: Iterable[int] = range(5)) -> dict:
        """Test utility of Real, TabDiff (N_r unguided rows), TARDIS_{K=1} and TARDIS, as in Table 1."""
        task = self.ds.task
        if self.backbone is None:
            self.fit_backbone()
        res = {"metric": METRIC_NAME[task], "real": self.utility(self.ds.X(self.ds.train), self.ds.y(self.ds.train), "test"),
               "tabdiff": [], "tardis_k1": [], "tardis": [], "config": cfg.to_dict()}
        for s in seeds:
            torch.manual_seed(s)
            tabdiff = self.backbone.sample_all(self.n_real, self.backbone_config.sample_batch_size)
            res["tabdiff"].append(self.utility(*self._pool_frame(tabdiff), "test"))
            pool = self.candidate_pool(cfg, s)
            X_pool, y_hard = self._pool_frame(pool)
            y = self.teacher().predict(X_pool) if cfg.gkd else y_hard
            inputs = self._sampler_inputs(pool, cfg, cfg.sampler)
            for key, K in (("tardis_k1", 1.0), ("tardis", cfg.K)):
                idx = self.select(pool, cfg, s, K, inputs=inputs)
                res[key].append(self.utility(X_pool.iloc[idx], y[idx], "test"))
            log.info(f"seed {s}: TabDiff={res['tabdiff'][-1]:.4f} TARDIS_K=1={res['tardis_k1'][-1]:.4f} "
                     f"TARDIS={res['tardis'][-1]:.4f} (Real={res['real']:.4f})")
        for key in ("tabdiff", "tardis_k1", "tardis"):
            res[f"{key}_mean"], res[f"{key}_std"] = float(np.mean(res[key])), float(np.std(res[key]))
        res["delta_vs_real"] = delta_percent(task, res["tardis_mean"], res["real"])
        res["delta_vs_tabdiff"] = delta_percent(task, res["tardis_mean"], res["tabdiff_mean"])
        (self.workdir / "report.json").write_text(json.dumps(res, indent=2))
        return res


def suggest_config(trial: optuna.Trial, k_max: float = 25.0) -> TardisConfig:
    """Stage II search space of Table 7 (plus the beta-VAE parameters of App. C.2)."""
    cfg = TardisConfig(K=trial.suggest_float("K", 0.1, k_max, step=0.1),
                       guidance=trial.suggest_categorical("guidance", GUIDANCE_VARIANTS))
    if cfg.guidance != "none":
        cfg.schedule = trial.suggest_categorical("schedule", GAMMA_SCHEDULES)
        cfg.gamma_max = trial.suggest_float("gamma_max", 0.1, 5.0, log=True)
        cfg.gamma_min = trial.suggest_float("gamma_min", 1e-8, cfg.gamma_max - 0.1 + 1e-7, log=True)
        cfg.t_g = trial.suggest_int("t_g", 3, 20)
        cfg.ref_ratio = REF_RATIOS[trial.suggest_categorical("ref_ratio", list(REF_RATIOS))]
    if cfg.guidance == "vae":
        cfg.vae.latent_dim = trial.suggest_categorical("vae_latent_dim", VAE_LATENT_DIMS)
        cfg.vae.beta_schedule = trial.suggest_categorical("vae_beta_schedule", BETA_SCHEDULES)
        cfg.vae.beta_max = trial.suggest_float("vae_beta_max", 0.5, 10.0, log=True)
        if cfg.vae.beta_schedule == "cyclical":
            cfg.vae.cycle = trial.suggest_int("vae_beta_cycle", 5, 30)
    return cfg
