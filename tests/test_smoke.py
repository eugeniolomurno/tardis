"""End-to-end smoke test on tiny random data (about two minutes on CPU)."""

import numpy as np
import pandas as pd
import pytest
import torch

from tardis import TARDIS, BackboneConfig, TardisConfig, VAEConfig, load_dataset
from tardis.chamfer import chamfer, chamfer_guidance
from tardis.config import SAMPLERS
from tardis.guidance import gamma_schedule


def toy(task, n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, n).astype(float),
                       "c": rng.choice(["x", "y", "z"], n)})
    signal = df["a"] + (df["c"] == "x")
    if task == "regression":
        df["t"] = signal + 0.1 * rng.normal(size=n)
    elif task == "binclass":
        df["t"] = np.where(signal > 0.3, "yes", "no")
    else:
        df["t"] = pd.cut(signal, 3, labels=["lo", "mid", "hi"]).astype(str)
    df.loc[::17, "a"] = np.nan  # exercise imputation
    return df


def test_chamfer():
    a, b = torch.randn(50, 3, requires_grad=True), torch.randn(40, 3)
    chamfer_guidance(b, a).backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert chamfer(b, b) < 1e-2 < chamfer(a.detach(), b)


def test_gamma_schedules():
    T = 50
    assert gamma_schedule("cosine", 0.0, 1.0, 1, T) > gamma_schedule("cosine", 0.0, 1.0, T, T)  # decreasing
    assert gamma_schedule("sine", 0.0, 1.0, 1, T) < gamma_schedule("sine", 0.0, 1.0, T, T)  # increasing
    assert gamma_schedule("linear", 0.0, 1.0, T, T) == 0.0


@pytest.mark.parametrize("task", ["binclass", "multiclass", "regression"])
def test_pipeline(task, tmp_path):
    ds = load_dataset(toy(task), target="t", task=task)
    assert (len(ds.train), len(ds.val), len(ds.test)) == (270, 15, 15)
    model = TARDIS(ds, tmp_path, M=3, device="cpu", inner_trials=2,
                   backbone_config=BackboneConfig(epochs=4, sample_batch_size=256))
    model.fit_backbone()

    vae = VAEConfig(epochs=3, beta_schedule="cyclical", warmup=10, cycle=5)
    for guidance, ratio in (("identity", 0.5), ("vae", None)):
        cfg = TardisConfig(K=1.5, guidance=guidance, schedule="cosine", gamma_max=0.5, t_g=3,
                           ref_ratio=ratio, latent_sampler=True, vae=vae)
        pool = model.candidate_pool(cfg, seed=0)
        assert pool.shape == (3 * model.n_real, model.x_real.shape[1])
        for sampler in SAMPLERS:
            idx = model.select(pool, cfg, seed=0, sampler=sampler)
            assert 0 < len(idx) <= int(1.5 * model.n_real)

    cfg = model.search(n_trials=2)
    df = model.generate(cfg, seed=0)
    assert 0 < len(df) <= int(cfg.K * model.n_real) and set(ds.train.columns) <= set(df.columns)
    report = model.report(cfg)
    assert np.isfinite(report["tardis_mean"]) and np.isfinite(report["real"])


@pytest.mark.parametrize("task", ["multiclass", "regression"])
def test_inner_study(task, tmp_path):
    ds = load_dataset(toy(task), target="t", task=task)
    model = TARDIS(ds, tmp_path, M=2, device="cpu", inner_trials=2,
                   backbone_config=BackboneConfig(epochs=2, sample_batch_size=256))
    cfg = TardisConfig(K=1.0, guidance="vae", schedule="sine", t_g=5, ref_ratio=0.25, latent_sampler=True)
    vae, pool = model.inner_study(cfg)
    assert isinstance(vae, VAEConfig) and pool.shape[0] == 2 * model.n_real


def test_split_sizes_and_other_learner(tmp_path):
    ds = load_dataset(toy("binclass"), target="t", task="binclass", val_size=0.1, test_size=0.2)
    assert (len(ds.train), len(ds.val), len(ds.test)) == (210, 30, 60)
    with pytest.raises(ValueError):
        load_dataset(toy("binclass"), target="t", task="binclass", val_size=0.6, test_size=0.4)
    model = TARDIS(ds, tmp_path, learner="linear", M=2, device="cpu",
                   backbone_config=BackboneConfig(epochs=2, sample_batch_size=256))
    pool = model.candidate_pool(TardisConfig(K=1.0, gkd=True), 0)
    _, X, _ = model.frames(pool)
    soft = model.teacher(0).predict(X)
    assert 0.0 <= model.utility(X, soft, "test") <= 1.0
