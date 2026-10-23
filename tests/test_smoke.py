"""End-to-end smoke test on tiny random data (a few seconds on CPU)."""

import numpy as np
import pandas as pd
import pytest
import torch

from tardis import TARDIS, BackboneConfig, TardisConfig, load_dataset
from tardis.chamfer import chamfer
from tardis.config import SAMPLERS


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


def test_chamfer_gradient():
    a = torch.randn(50, 3, requires_grad=True)
    b = torch.randn(40, 3)
    loss = chamfer(a, b)
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(a.grad).all()
    assert chamfer(b, b).item() < 1e-4


@pytest.mark.parametrize("task", ["binclass", "multiclass", "regression"])
def test_pipeline(task, tmp_path):
    ds = load_dataset(toy(task), target="t", task=task)
    assert len(ds.train) + len(ds.val) + len(ds.test) == 300
    model = TARDIS(ds, tmp_path, M=3, device="cpu",
                   backbone_config=BackboneConfig(epochs=4, sample_batch_size=256))
    model.fit_backbone()

    # every sampler, with and without GKD, under both guidance maps
    for guidance in ("identity", "vae"):
        cfg = TardisConfig(K=1.5, guidance=guidance, schedule="cosine", gamma_max=0.5, t_g=3, ref_ratio=0.5)
        cfg.vae.epochs = 3
        pool = model.candidate_pool(cfg, seed=0)
        assert pool.shape == (3 * model.n_real, model.x_real.shape[1])
        for sampler in SAMPLERS:
            idx = model.select(pool, cfg, seed=0, sampler=sampler)
            assert len(idx) == int(1.5 * model.n_real)

    cfg = model.search(n_trials=2)
    df = model.generate(cfg, seed=0)
    assert len(df) == int(cfg.K * model.n_real) and set(ds.train.columns) <= set(df.columns)
    report = model.report(cfg, seeds=[0])
    assert np.isfinite(report["tardis_mean"]) and np.isfinite(report["real"])


def test_other_learner(tmp_path):
    ds = load_dataset(toy("binclass"), target="t", task="binclass")
    model = TARDIS(ds, tmp_path, learner="linear", M=2, device="cpu",
                   backbone_config=BackboneConfig(epochs=2, sample_batch_size=256))
    cfg = TardisConfig(K=1.0, gkd=True)
    pool = model.candidate_pool(cfg, 0)
    X, y = model._pool_frame(pool)
    soft = model.teacher().predict(X)
    assert 0.0 <= model.utility(X, soft, "test") <= 1.0
