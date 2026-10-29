"""Downstream learners A. XGBoost is the learner of the paper; the MLP and the linear
models are two of the zero-shot transfer learners of App. L, and the random forest is an
additional option.

A learner is fitted on ``(X, y)`` where ``y`` is either a hard target (class index or
normalised regression value) or a GKD soft target (probability of class 1 for binary
tasks, an ``(n, n_classes)`` probability matrix for multiclass tasks). ``predict`` returns
scores: P(class 1), class probabilities, or the regression value.
"""

import numpy as np
import pandas as pd


class XGBoostLearner:
    """XGBoost with default hyperparameters, up to 10,000 rounds and early stopping
    (patience 250) on the validation split."""

    def __init__(self, task, n_classes=0, seed=0, device="cpu", n_rounds=10000, patience=250, **_):
        self.task, self.k, self.seed = task, n_classes, seed
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.n_rounds, self.patience = n_rounds, patience

    def fit(self, X, y, X_val, y_val):
        import xgboost as xgb

        params = {"tree_method": "hist", "device": self.device, "seed": self.seed, "verbosity": 0}
        obj = metric = None
        y = np.asarray(y)
        label = y
        if self.task == "regression":
            params.update(objective="reg:squarederror", eval_metric="rmse")
        elif self.task == "binclass":  # binary:logistic accepts soft labels in [0, 1]
            params.update(objective="binary:logistic", eval_metric="logloss")
        elif y.ndim == 1:
            params.update(objective="multi:softprob", eval_metric="mlogloss", num_class=self.k)
        else:  # multiclass soft labels: softmax cross-entropy against the teacher distribution
            params.update(num_class=self.k, disable_default_eval_metric=1)
            obj, metric, label = _soft_softmax_objective(y), _mlogloss, y.argmax(1)
        dtrain = xgb.DMatrix(X, label=label, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        self.model = xgb.train(params, dtrain, self.n_rounds, evals=[(dval, "val")], obj=obj,
                               custom_metric=metric, early_stopping_rounds=self.patience, verbose_eval=False)
        self.soft_multiclass = obj is not None
        return self

    def predict(self, X):
        import xgboost as xgb

        out = self.model.predict(xgb.DMatrix(X, enable_categorical=True),
                                 iteration_range=(0, self.model.best_iteration + 1),
                                 output_margin=self.soft_multiclass)
        return _softmax(out) if self.soft_multiclass else out


def _softmax(z):
    z = np.asarray(z).reshape(len(z), -1)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _soft_softmax_objective(soft):
    def obj(predt, dtrain):
        p = _softmax(predt)
        return (p - soft).reshape(predt.shape), np.maximum(p * (1 - p), 1e-6).reshape(predt.shape)
    return obj


def _mlogloss(predt, dmat):
    p = _softmax(predt)
    y = dmat.get_label().astype(int)
    return "mlogloss", float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


class SklearnLearner:
    """One-hot + standardisation followed by a scikit-learn estimator. GKD soft labels are
    fitted exactly via sample weights (each row repeated once per class, weighted by its
    probability) when the estimator accepts them, and by their argmax otherwise."""

    def __init__(self, task, n_classes=0, seed=0, kind="mlp", **_):
        self.task, self.k, self.seed, self.kind = task, n_classes, seed, kind

    def _estimator(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        reg = self.task == "regression"
        if self.kind == "mlp":  # hidden layers (128, 64) as in App. L
            cls = MLPRegressor if reg else MLPClassifier
            return cls(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=self.seed)
        if self.kind == "linear":
            return Ridge() if reg else LogisticRegression(max_iter=2000)
        if self.kind == "random_forest":
            cls = RandomForestRegressor if reg else RandomForestClassifier
            return cls(n_estimators=500, n_jobs=-1, random_state=self.seed)
        raise ValueError(f"unknown learner {self.kind!r}")

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.utils.validation import has_fit_parameter

        cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        num = [c for c in X.columns if c not in cat]
        prep = ColumnTransformer([("num", StandardScaler(), num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        est = self._estimator()
        self.model = Pipeline([("prep", prep), ("est", est)])
        y = np.asarray(y)
        weights = None
        if self.task != "regression" and (y.dtype.kind == "f" or y.ndim == 2):
            probs = np.stack([1 - y, y], 1) if y.ndim == 1 else y
            if has_fit_parameter(est, "sample_weight"):
                X = pd.concat([X] * probs.shape[1], ignore_index=True)
                y = np.repeat(np.arange(probs.shape[1])[None], len(probs), 0).T.ravel()
                weights = probs.T.ravel()
            else:
                y = probs.argmax(1)
        fit_kw = {"est__sample_weight": weights} if weights is not None else {}
        self.model.fit(X, y, **fit_kw)
        return self

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X)
        proba = np.zeros((len(X), max(self.k, 2)))
        proba[:, self.model.classes_.astype(int)] = self.model.predict_proba(X)
        return proba[:, 1] if self.task == "binclass" else proba


LEARNERS = {"xgboost": XGBoostLearner, "mlp": SklearnLearner, "linear": SklearnLearner,
            "random_forest": SklearnLearner}


def make_learner(name, task, n_classes=0, seed=0, device="cpu"):
    """``name`` is a key of LEARNERS, or a callable ``(task, n_classes, seed) -> learner``."""
    if callable(name):
        return name(task, n_classes, seed)
    if name not in LEARNERS:
        raise ValueError(f"learner must be one of {sorted(LEARNERS)}, got {name!r}")
    return LEARNERS[name](task, n_classes=n_classes, seed=seed, device=device, kind=name)
