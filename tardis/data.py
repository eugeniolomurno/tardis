"""Dataset loading, the 90/5/5 split, and the numeric layout used by the backbone."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from .config import SEED

TASKS = ("binclass", "multiclass", "regression")


@dataclass
class Dataset:
    """Train / validation / test splits of one tabular dataset.

    The validation split drives early stopping of the downstream learner and the
    configuration search; the test split is used only for the final evaluation.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target: str
    task: str
    num_cols: List[str]
    cat_cols: List[str]

    def __post_init__(self):
        y = self.train[self.target]
        if self.is_classification:
            self.classes = sorted(y.unique())
        else:
            self.y_min, self.y_max = float(y.min()), float(y.max())
        self.levels = {c: sorted(self.train[c].unique()) for c in self.cat_cols}

    @property
    def is_classification(self) -> bool:
        return self.task != "regression"

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.is_classification else 0

    def X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for the downstream learner, with categories aligned to the train split."""
        X = df[self.num_cols + self.cat_cols].copy()
        for c in self.num_cols:
            X[c] = X[c].astype(float)
        for c in self.cat_cols:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.levels[c])
        return X

    def y(self, df: pd.DataFrame) -> np.ndarray:
        """Target in the space where the metric is computed: class index for classification,
        min-max normalised value (fitted on the train split) for regression."""
        y = df[self.target]
        if self.is_classification:
            return pd.Categorical(y.astype(str), categories=self.classes).codes.astype(int)
        return ((y.astype(float) - self.y_min) / (self.y_max - self.y_min + 1e-12)).to_numpy()

    def y_inverse(self, y: np.ndarray) -> np.ndarray:
        if self.is_classification:
            return np.asarray(self.classes, dtype=object)[np.asarray(y, dtype=int)]
        return np.asarray(y) * (self.y_max - self.y_min + 1e-12) + self.y_min


def load_dataset(
    data: Union[str, Path, pd.DataFrame],
    target: str,
    task: str,
    cat_cols: Optional[Sequence[str]] = None,
    seed: int = SEED,
    val_size: float = 0.05,
    test_size: float = 0.05,
) -> Dataset:
    """Read a CSV (or DataFrame) and split it into train / validation / test; ``val_size``
    and ``test_size`` are fractions of all rows (90/5/5 by default).

    Columns with a non-numeric dtype are categorical unless ``cat_cols`` is given.
    Missing values are imputed with the train-split median (numerical) or mode
    (categorical), as in App. F.1.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in {list(df.columns)}")
    df = df.dropna(subset=[target]).reset_index(drop=True)

    features = [c for c in df.columns if c != target]
    if cat_cols is None:
        cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = list(cat_cols)
    num_cols = [c for c in features if c not in cat_cols]
    df[target] = df[target].astype(str) if task != "regression" else df[target].astype(float)

    n_val, n_test = round(val_size * len(df)), round(test_size * len(df))
    if n_val < 1 or n_test < 1 or n_val + n_test >= len(df):
        raise ValueError("val_size and test_size must each cover at least one row and leave rows for training")
    train, rest = _split(df, n_val + n_test, seed, target, task)
    val, test = _split(rest, n_test, seed, target, task)

    medians = train[num_cols].median()
    modes = {c: train[c].mode().iloc[0] for c in cat_cols}

    def impute(part):
        part = part.copy()
        part[num_cols] = part[num_cols].fillna(medians)
        for c in cat_cols:
            part[c] = part[c].fillna(modes[c]).astype(str)
        return part.reset_index(drop=True)

    return Dataset(impute(train), impute(val), impute(test), target=target, task=task, num_cols=num_cols,
                   cat_cols=cat_cols)


def _split(df, size, seed, target, task):
    stratify = df[target] if task != "regression" else None
    try:
        return train_test_split(df, test_size=size, random_state=seed, stratify=stratify)
    except ValueError:  # a class too rare to stratify
        return train_test_split(df, test_size=size, random_state=seed)


class ModelSpace:
    """Bijection between rows of the dataset and the backbone's numeric layout.

    As in TabDiff, a row is ``[numerical | categorical]`` with the target placed first in
    its block (numerical for regression, categorical for classification). Numerical
    columns are mapped through a normal-output quantile transform, categorical columns
    to integer codes.
    """

    def __init__(self, ds: Dataset):
        regression = ds.task == "regression"
        self.columns = list(ds.train.columns)
        self.num = ([ds.target] if regression else []) + ds.num_cols
        self.cat = ([] if regression else [ds.target]) + ds.cat_cols
        self.levels = {c: (ds.classes if c == ds.target else ds.levels[c]) for c in self.cat}
        train = ds.train
        self.quantile = None
        if self.num:
            n_q = max(min(len(train) // 30, 1000), 10)
            self.quantile = QuantileTransformer(
                output_distribution="normal", n_quantiles=n_q, subsample=int(1e9), random_state=0
            ).fit(train[self.num].astype(float).to_numpy())
        self.int_cols = [c for c in self.num if np.all(np.mod(train[c].astype(float), 1) == 0)]
        self.target_index = 0 if regression else len(self.num)

    @property
    def d_num(self) -> int:
        return len(self.num)

    @property
    def categories(self) -> List[int]:
        return [len(self.levels[c]) for c in self.cat]

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.num:
            parts.append(self.quantile.transform(df[self.num].astype(float).to_numpy()))
        for c in self.cat:
            codes = pd.Categorical(df[c].astype(str), categories=self.levels[c]).codes
            parts.append(np.where(codes < 0, 0, codes)[:, None])
        return np.concatenate(parts, axis=1).astype(np.float32)

    def decode(self, x: np.ndarray) -> pd.DataFrame:
        out = {}
        if self.num:
            num = self.quantile.inverse_transform(x[:, : self.d_num])
            for j, c in enumerate(self.num):
                out[c] = np.round(num[:, j]) if c in self.int_cols else num[:, j]
        for j, c in enumerate(self.cat):
            codes = np.clip(np.round(x[:, self.d_num + j]).astype(int), 0, len(self.levels[c]) - 1)
            out[c] = np.asarray(self.levels[c], dtype=object)[codes]
        return pd.DataFrame(out)[self.columns]

    def onehot(self, x: torch.Tensor) -> torch.Tensor:
        """``[numerical | one-hot(categorical)]``; differentiable w.r.t. the numerical part."""
        num = x[:, : self.d_num]
        cats = [
            F.one_hot(x[:, self.d_num + j].long().clamp(0, k - 1), k).to(x.dtype)
            for j, k in enumerate(self.categories)
        ]
        return torch.cat([num] + cats, dim=1)
