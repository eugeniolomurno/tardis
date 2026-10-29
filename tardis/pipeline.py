"""TARDIS: Tabular generation through Refinement, Distillation, and Inference-time Sampling.

Stage I   oversample M * N_r noise vectors from the frozen backbone's prior;
Stage II  denoise them, optionally with bidirectional Chamfer guidance (Eq. 4);
Stage III select K * N_r rows from the candidate pool with one of four samplers,
          optionally after relabelling the pool with GKD soft labels.
Stage II hyperparameters are searched with TPE; each trial's objective is the best validation
utility of the Stage III grid (4 samplers x GKD on/off), each cell averaging five evaluation seeds.
When a trial selects beta-VAE mapping or the latent sampler, an inner TPE study first
selects the beta-VAE (Table 8).
"""

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .backbone import load_tabdiff, save_tabdiff, train_tabdiff
from .chamfer import IdentityMap, VAEMap, embed
from .config import (BETA_SCHEDULES, EVAL_SEEDS, GAMMA_SCHEDULES, GUIDANCE_VARIANTS, INNER_SEEDS, INNER_TRIALS,
                     POOL_MULTIPLIER, REF_RATIOS, SAMPLERS, SEED, TARGET_CLUSTERS, VAE_LATENT_DIMS, BackboneConfig,
                     TardisConfig, VAEConfig)
from .data import Dataset, ModelSpace
from .guidance import ChamferGuidance
from .learners import XGBoostLearner, make_learner
from .metrics import METRIC_NAME, delta_percent, score, utility
from .samplers import DETERMINISTIC, SAMPLER_FNS
from .vae import train_vae

log = logging.getLogger("tardis")


class TARDIS:
    def __init__(self, dataset: Dataset, workdir="runs/tardis", learner="xgboost", M: float = POOL_MULTIPLIER,
                 device: Optional[str] = None, seed: int = SEED, backbone_config: Optional[BackboneConfig] = None,
                 inner_trials: int = INNER_TRIALS, target_clusters: int = TARGET_CLUSTERS):
        self.ds, self.space = dataset, ModelSpace(dataset)
        self.x_real = torch.from_numpy(self.space.encode(dataset.train))
        self.n_real = len(dataset.train)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.learner, self.M, self.seed = learner, M, seed
        self.backbone_config = backbone_config or BackboneConfig()
        self.inner_trials, self.target_clusters = inner_trials, target_clusters
        self.backbone = None
        self._vaes, self._pools, self._teachers = {}, {}, {}

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
    def vae(self, cfg: VAEConfig):
        """beta-VAE trained on D_real (one per parameter setting)."""
        key = json.dumps(asdict(cfg), sort_keys=True)
        if key not in self._vaes:
            self._vaes[key] = train_vae(self.space.onehot(self.x_real), self.space.d_num, self.space.categories,
                                        cfg, self.device, self.seed)
        return self._vaes[key]

    def representation(self, cfg: TardisConfig):
        """phi of Stage II: the beta-VAE encoder for ``guidance='vae'``, the identity map otherwise."""
        return VAEMap(self.space, self.vae(cfg.vae)) if cfg.guidance == "vae" else IdentityMap()

    def candidate_pool(self, cfg: TardisConfig, seed: int) -> torch.Tensor:
        """D_cand: M * N_r rows in model space, generated with the Stage II setting of ``cfg``."""
        if self.backbone is None:
            self.fit_backbone()
        key = (seed, _stage2_key(cfg))
        if key in self._pools:
            return self._pools[key]
        guidance = None
        if cfg.guidance != "none":
            guidance = ChamferGuidance(
                self.representation(cfg), self.x_real, self.backbone.num_timesteps,
                schedule=cfg.schedule, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max, t_g=cfg.t_g,
                ref_ratio=cfg.ref_ratio, target_index=self.space.target_index, task=self.ds.task,
                n_classes=self.ds.n_classes, n_clusters=self.target_clusters, device=self.device, seed=seed,
            )
        torch.manual_seed(seed)
        pool = self.backbone.sample_all(int(self.M * self.n_real), self.backbone_config.sample_batch_size, guidance)
        guided = [k for k in self._pools if k[1] != _stage2_key(TardisConfig())]
        if len(guided) >= 2:  # keep the unguided pools and the two most recent guided ones
            self._pools.pop(guided[0])
        self._pools[key] = pool
        return pool

    # --------------------------------------------------------------- Stage III
    def frames(self, pool: torch.Tensor):
        """The pool decoded to the original columns, its learner features and hard targets."""
        df = self.space.decode(pool.numpy())
        return df, self.ds.X(df), self.ds.y(df)

    def sampler_inputs(self, sampler: str, cfg: TardisConfig, pool: torch.Tensor, df: pd.DataFrame, X: pd.DataFrame):
        """Arrays each sampler works on (App. D)."""
        if sampler == "chamfer":  # one-hot features, target excluded, z-scored on D_real and the pool
            real = self.ds.X(self.ds.train)
            both = pd.get_dummies(pd.concat([real, X], ignore_index=True), drop_first=True).to_numpy(np.float32)
            both = torch.as_tensor(StandardScaler().fit_transform(both), dtype=torch.float32, device=self.device)
            return {"P": both[len(real):], "R": both[: len(real)]}
        if sampler == "md":
            if cfg.latent_sampler:  # beta-VAE posterior means
                phi = VAEMap(self.space, self.vae(cfg.vae)).to(self.device)
                return {"P": embed(phi, pool, self.device), "R": embed(phi, self.x_real, self.device)}
            cats = self.ds.cat_cols + ([self.ds.target] if self.ds.is_classification else [])
            real = pd.get_dummies(self.ds.train, columns=cats)
            fake = pd.get_dummies(df, columns=cats)
            real, fake = real.align(fake, join="outer", axis=1, fill_value=0)
            to = lambda d: torch.as_tensor(d.to_numpy(np.float32), device=self.device)
            return {"P": to(fake), "R": to(real)}
        if sampler == "stratified":
            return {"y_pool": pool[:, self.space.target_index].numpy()}
        return {"X": X}  # iboss

    def select(self, pool: torch.Tensor, cfg: TardisConfig, seed: int, K: Optional[float] = None,
               sampler: Optional[str] = None) -> np.ndarray:
        """Indices of the K * N_r pool rows retained by ``sampler`` (default ``cfg.sampler``)."""
        sampler = sampler or cfg.sampler
        df, X, _ = self.frames(pool)
        inputs = self.sampler_inputs(sampler, cfg, pool, df, X)
        return SAMPLER_FNS[sampler](n_out=self._n_out(pool, cfg.K if K is None else K),
                                    rng=np.random.default_rng(seed), task=self.ds.task, **inputs)

    def _n_out(self, pool, K):
        return min(int(self.n_real * K), len(pool))

    # ------------------------------------------------- learner, GKD and utility
    def teacher(self, seed: int) -> XGBoostLearner:
        """f_GKD (App. E): XGBoost trained on one half of D_real, early-stopped on the other."""
        if seed not in self._teachers:
            X_a, X_b, y_a, y_b = train_test_split(self.ds.X(self.ds.train), self.ds.y(self.ds.train),
                                                  test_size=0.5, random_state=seed)
            self._teachers[seed] = XGBoostLearner(self.ds.task, self.ds.n_classes, seed, str(self.device)).fit(
                X_a, y_a, X_b, y_b)
        return self._teachers[seed]

    def utility(self, X, y, split="val") -> float:
        """psi(A(D), D_split): train on (X, y), early-stop on the validation split, score on ``split``."""
        learner = make_learner(self.learner, self.ds.task, self.ds.n_classes, self.seed, str(self.device))
        learner.fit(X, y, self.ds.X(self.ds.val), self.ds.y(self.ds.val))
        part = getattr(self.ds, split)
        return utility(self.ds.task, self.ds.y(part), learner.predict(self.ds.X(part)))

    def cell(self, frames, inputs, sampler, gkd, K, split="val"):
        """Utility of one Stage III cell, per evaluation seed, on a pool generated once. A seed
        changes only the selection of the stochastic samplers and the half of D_real the GKD
        teacher is trained on; the downstream learner has a fixed seed and a fixed split."""
        df, X, y_hard = frames
        deterministic = sampler in DETERMINISTIC
        values, idx = [], None
        for s in EVAL_SEEDS:
            if deterministic and not gkd and values:  # same rows, deterministic learner: same value
                values.append(values[0])
                continue
            if idx is None or not deterministic:
                idx = SAMPLER_FNS[sampler](n_out=self._n_out(df, K), rng=np.random.default_rng(s),
                                           task=self.ds.task, **inputs)
            y = self.teacher(s).predict(X) if gkd else y_hard
            values.append(self.utility(X.iloc[idx], y[idx], split))
        return values

    # ------------------------------------------------------------------ search
    def search(self, n_trials: int = 500) -> TardisConfig:
        """TPE over the Stage II space (Table 7); each trial runs the Stage III grid on the
        validation split. Trials are stored in ``workdir/search.db`` and a search can be resumed."""
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
        log.info(f"best {METRIC_NAME[self.ds.task]}: {abs(study.best_value):.4f}  config: {best}")
        return best

    def _objective(self, trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, k_max=min(25.0, self.M))  # K <= M
        if cfg.needs_vae:
            cfg.vae, pool = self.inner_study(cfg)
        else:
            pool = self.candidate_pool(cfg, self.seed)
        frames = self.frames(pool)
        grid = {}
        for sampler in SAMPLERS:
            try:
                inputs = self.sampler_inputs(sampler, cfg, pool, frames[0], frames[1])
                for gkd in (False, True):
                    grid[(sampler, gkd)] = float(np.mean(self.cell(frames, inputs, sampler, gkd, cfg.K)))
            except Exception as e:  # a sampler failing must not kill the trial
                log.warning(f"trial {trial.number}: sampler {sampler} failed ({e})")
        if not grid:
            raise optuna.TrialPruned("every sampler failed")
        (cfg.sampler, cfg.gkd), best = max(grid.items(), key=lambda kv: score(self.ds.task, kv[1]))
        trial.set_user_attr("config", cfg.to_dict())
        trial.set_user_attr("grid", {f"{s}{'+gkd' if g else ''}": v for (s, g), v in grid.items()})
        log.info(f"trial {trial.number}: {METRIC_NAME[self.ds.task]}={best:.4f}  K={cfg.K} guidance={cfg.guidance} "
                 f"latent_sampler={cfg.latent_sampler} sampler={cfg.sampler} gkd={cfg.gkd}")
        return score(self.ds.task, best)

    def inner_study(self, cfg: TardisConfig):
        """beta-VAE inner study (Table 8): a separate TPE study whose objective is the utility of
        a uniformly random subset of K * N_r candidates, averaged over five seeds. Returns the
        selected beta-VAE and the candidate pool generated with it."""
        best = {"score": -np.inf, "pool": None}

        def objective(trial):
            trial_cfg = replace(cfg, vae=suggest_vae(trial))
            pool = self.candidate_pool(trial_cfg, self.seed)
            _, X, y = self.frames(pool)
            values = []
            for s in INNER_SEEDS:
                idx = np.random.default_rng(s).choice(len(pool), self._n_out(pool, cfg.K), replace=False)
                values.append(self.utility(X.iloc[idx], y[idx], "val"))
            value = score(self.ds.task, float(np.mean(values)))
            trial.set_user_attr("vae", asdict(trial_cfg.vae))
            if value > best["score"]:
                best.update(score=value, pool=pool)
            return value

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.inner_trials)
        return VAEConfig(**study.best_trial.user_attrs["vae"]), best["pool"]

    # ------------------------------------------------------------------- apply
    def generate(self, cfg: TardisConfig, seed: Optional[int] = None, K: Optional[float] = None) -> pd.DataFrame:
        """The synthetic dataset D_syn (K * N_r rows) in the original column space.

        With GKD, classification targets keep the hard label and gain the teacher's
        probabilities as ``<target>__p_<class>`` columns; regression targets are replaced by
        the teacher's prediction.
        """
        seed = self.seed if seed is None else seed
        pool = self.candidate_pool(cfg, self.seed)
        idx = self.select(pool, cfg, seed, K)
        df = self.space.decode(pool[idx].numpy())
        if cfg.gkd:
            soft = self.teacher(seed).predict(self.ds.X(df))
            t = self.ds.target
            if self.ds.task == "regression":
                df[t] = self.ds.y_inverse(soft)
            else:
                soft = np.stack([1 - soft, soft], 1) if soft.ndim == 1 else soft
                for j, c in enumerate(self.ds.classes):
                    df[f"{t}__p_{c}"] = soft[:, j]
        return df

    def report(self, cfg: TardisConfig) -> dict:
        """Test utility of Real, TabDiff (N_r unguided rows), TARDIS_{K=1} and TARDIS, as in
        Table 1; the TARDIS columns average the five evaluation seeds over one candidate pool."""
        task = self.ds.task
        if self.backbone is None:
            self.fit_backbone()
        res = {"metric": METRIC_NAME[task], "config": cfg.to_dict(),
               "real": self.utility(self.ds.X(self.ds.train), self.ds.y(self.ds.train), "test"), "tabdiff": []}
        for s in EVAL_SEEDS:
            torch.manual_seed(s)
            rows = self.backbone.sample_all(self.n_real, self.backbone_config.sample_batch_size)
            _, X, y = self.frames(rows)
            res["tabdiff"].append(self.utility(X, y, "test"))
        pool = self.candidate_pool(cfg, self.seed)
        frames = self.frames(pool)
        inputs = self.sampler_inputs(cfg.sampler, cfg, pool, frames[0], frames[1])
        res["tardis_k1"] = self.cell(frames, inputs, cfg.sampler, cfg.gkd, 1.0, "test")
        res["tardis"] = self.cell(frames, inputs, cfg.sampler, cfg.gkd, cfg.K, "test")
        for key in ("tabdiff", "tardis_k1", "tardis"):
            res[f"{key}_mean"], res[f"{key}_std"] = float(np.mean(res[key])), float(np.std(res[key]))
        res["delta_vs_real"] = delta_percent(task, res["tardis_mean"], res["real"])
        res["delta_vs_tabdiff"] = delta_percent(task, res["tardis_mean"], res["tabdiff_mean"])
        log.info(f"test {res['metric']}: Real={res['real']:.4f} TabDiff={res['tabdiff_mean']:.4f} "
                 f"TARDIS_K=1={res['tardis_k1_mean']:.4f} TARDIS={res['tardis_mean']:.4f}")
        (self.workdir / "report.json").write_text(json.dumps(res, indent=2))
        return res


def _stage2_key(cfg: TardisConfig) -> str:
    """What determines the candidate pool: the Stage II parameters, and the beta-VAE only
    when it is the guidance map."""
    if cfg.guidance == "none":
        return "none"
    keys = ("guidance", "schedule", "gamma_max", "gamma_min", "t_g", "ref_ratio")
    d = {k: getattr(cfg, k) for k in keys}
    if cfg.guidance == "vae":
        d["vae"] = asdict(cfg.vae)
    return json.dumps(d, sort_keys=True)


def suggest_config(trial: optuna.Trial, k_max: float = 25.0) -> TardisConfig:
    """Stage II search space of Table 7."""
    cfg = TardisConfig(K=trial.suggest_float("K", 0.1, k_max, step=0.1),
                       guidance=trial.suggest_categorical("guidance", GUIDANCE_VARIANTS))
    if cfg.guidance != "none":
        cfg.schedule = trial.suggest_categorical("schedule", GAMMA_SCHEDULES)
        cfg.gamma_max = trial.suggest_float("gamma_max", 0.1, 5.0, log=True)
        cfg.gamma_min = trial.suggest_float("gamma_min", 1e-8, cfg.gamma_max - 0.1 + 1e-7, log=True)
        cfg.t_g = trial.suggest_int("t_g", 3, 20)
        cfg.ref_ratio = REF_RATIOS[trial.suggest_categorical("ref_ratio", list(REF_RATIOS))]
    cfg.latent_sampler = trial.suggest_categorical("latent_sampler", [True, False])
    return cfg


def suggest_vae(trial: optuna.Trial) -> VAEConfig:
    """beta-VAE search space of Table 8."""
    v = VAEConfig(
        n_layers=trial.suggest_int("n_layers", 1, 6),
        latent_dim=trial.suggest_categorical("latent_dim", VAE_LATENT_DIMS),
        beta_max=trial.suggest_float("beta_max", 0.5, 10.0, log=True),
        beta_schedule=trial.suggest_categorical("beta_schedule", BETA_SCHEDULES),
    )
    if v.beta_schedule != "constant":
        v.warmup = trial.suggest_int("warmup", 10, 100)
    if v.beta_schedule == "cyclical":
        v.cycle = trial.suggest_int("cycle", 5, 30)
    v.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    v.batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    v.skip_connections = trial.suggest_categorical("skip_connections", [True, False])
    v.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    v.epochs = trial.suggest_int("epochs", 20, 500, step=5)
    return v
