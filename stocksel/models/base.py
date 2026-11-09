"""Model zoo behind one interface: ``fit(X, y, y_reg, w)`` and ``raw_logit(X)``.

Every model returns a *raw logit-like score* (higher = more likely to beat the
benchmark). Calibration to probabilities happens separately (models.calibration).

* logit / enet : L2 / elastic-net logistic regression (mandatory baselines)
* ridge        : ridge regression on the cross-sectional rank of excess return
                 (secondary ranking target; its score is calibrated like the others)
* lgbm         : LightGBM, shallow and heavily regularized (captures interactions)
* mlp          : small MLP (2-4 hidden layers) with weight decay
* xgb          : XGBoost if installed (optional)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier

EPS = 1e-6

import sklearn  # noqa: E402

_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class BaseModel:
    name = "base"
    kind = "linear"

    def fit(self, X, y, y_reg=None, w=None):
        raise NotImplementedError

    def raw_logit(self, X) -> np.ndarray:
        raise NotImplementedError


class LogitModel(BaseModel):
    name, kind = "logit", "linear"

    def __init__(self, C=0.05, seed=0, l1_ratio=None):
        kw = dict(C=C, max_iter=2000)
        if l1_ratio is not None:
            # saga shuffles the data: fix its seed so research runs are exactly reproducible
            kw.update(solver="saga", l1_ratio=l1_ratio, tol=1e-3, random_state=seed)
            if _SKLEARN < (1, 8):  # older sklearn needs the explicit penalty name
                kw["penalty"] = "elasticnet"
        self.m = LogisticRegression(**kw)
        self.mean_ = None

    def fit(self, X, y, y_reg=None, w=None):
        self.mean_ = X.mean(axis=0)
        self.m.fit(X, y, sample_weight=w)
        return self

    def raw_logit(self, X):
        return self.m.decision_function(X)

    def contributions(self, X):
        return (X - self.mean_) * self.m.coef_[0]


class EnetModel(LogitModel):
    name = "enet"

    def __init__(self, C=0.05, l1_ratio=0.5, seed=0):
        super().__init__(C=C, seed=seed, l1_ratio=l1_ratio)


class RidgeRankModel(BaseModel):
    name, kind = "ridge", "linear"

    def __init__(self, alpha=50.0, seed=0):
        self.m = Ridge(alpha=alpha)
        self.mean_ = None

    def fit(self, X, y, y_reg=None, w=None):
        target = y_reg if y_reg is not None else y - 0.5
        self.mean_ = X.mean(axis=0)
        self.m.fit(X, target, sample_weight=w)
        return self

    def raw_logit(self, X):
        return self.m.predict(X)

    def contributions(self, X):
        return (X - self.mean_) * self.m.coef_


class LGBMModel(BaseModel):
    name, kind = "lgbm", "tree"

    def __init__(self, seed=0, **params):
        import lightgbm as lgb
        p = dict(n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=300,
                 subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0)
        p.update(params)
        self.m = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=1, **p)

    def fit(self, X, y, y_reg=None, w=None):
        self.m.fit(X, y, sample_weight=w)
        return self

    def raw_logit(self, X):
        return self.m.predict(X, raw_score=True)

    def contributions(self, X):
        c = self.m.predict(X, pred_contrib=True)
        return c[:, :-1]  # exact TreeSHAP in log-odds; last column is the bias


class XGBModel(BaseModel):
    name, kind = "xgb", "tree"

    def __init__(self, seed=0, **params):
        import xgboost as xgb  # optional
        p = dict(n_estimators=250, learning_rate=0.03, max_depth=3, subsample=0.7, colsample_bytree=0.7,
                 reg_lambda=10.0, min_child_weight=50)
        p.update(params)
        self.m = xgb.XGBClassifier(random_state=seed, n_jobs=1, **p)

    def fit(self, X, y, y_reg=None, w=None):
        self.m.fit(X, y, sample_weight=w)
        return self

    def raw_logit(self, X):
        return self.m.predict(X, output_margin=True)

    def contributions(self, X):
        import xgboost as xgb
        return self.m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


class TunedLGBMModel(BaseModel):
    """LightGBM with hyper-parameters chosen per training window by inner purged K-fold log-loss.

    Only the training window is used for the choice (nested validation); the test months of the
    outer walk-forward never influence it. Every grid point is an inner fit, not a reported trial.
    """
    name, kind = "lgbm_tuned", "tree"

    def __init__(self, seed=0, grid=None, fixed=None, inner_folds=3):
        self.seed = seed
        self.grid = grid or {"num_leaves": [7, 15, 31], "min_child_samples": [200, 800]}
        self.fixed = dict(fixed or {})
        self.inner_folds = int(inner_folds)
        self._t0 = self._t1 = None
        self.best_params_ = None
        self.m = None

    def set_time_index(self, start, end):
        self._t0, self._t1 = np.asarray(start), np.asarray(end)

    def _make(self, **params):
        import lightgbm as lgb
        p = dict(self.fixed)
        p.update(params)
        return lgb.LGBMClassifier(random_state=self.seed, verbose=-1, n_jobs=1, subsample_freq=1, **p)

    def fit(self, X, y, y_reg=None, w=None):
        import itertools

        import pandas as pd
        from sklearn.metrics import log_loss

        from ..validation.splits import PurgedKFold
        keys = list(self.grid)
        combos = [dict(zip(keys, v)) for v in itertools.product(*self.grid.values())]
        if self._t0 is not None and len(combos) > 1:
            cv = PurgedKFold(self.inner_folds, embargo_pct=0.01)
            scores = {i: [] for i in range(len(combos))}
            for tr, te in cv.split(pd.Series(self._t0), pd.Series(self._t1)):
                if len(tr) < 200 or len(np.unique(y[te])) < 2:
                    continue
                for i, c in enumerate(combos):
                    m = self._make(**c).fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
                    scores[i].append(log_loss(y[te], np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)))
            mean = {i: np.mean(v) for i, v in scores.items() if v}
            best = combos[min(mean, key=mean.get)] if mean else combos[0]
        else:
            best = combos[0]
        self.best_params_ = best
        self.m = self._make(**best).fit(X, y, sample_weight=w)
        return self

    def raw_logit(self, X):
        return self.m.predict(X, raw_score=True)

    def contributions(self, X):
        return self.m.predict(X, pred_contrib=True)[:, :-1]


class MLPModel(BaseModel):
    name, kind = "mlp", "nn"

    def __init__(self, hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=40, learning_rate_init=0.001, seed=0):
        hls = tuple(hidden_layer_sizes)
        if not 2 <= len(hls) <= 4:
            raise ValueError("MLP must have 2-4 hidden layers")
        self.m = MLPClassifier(hidden_layer_sizes=hls, alpha=alpha, max_iter=max_iter,
                               learning_rate_init=learning_rate_init, random_state=seed, early_stopping=False)
        self.mean_ = None

    def fit(self, X, y, y_reg=None, w=None):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        self.mean_ = X.mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.m.fit(X, y)
        return self

    def raw_logit(self, X):
        return _logit(self.m.predict_proba(X)[:, 1])

    def contributions(self, X):
        """Occlusion attribution (SHAP-like approximation): logit change when a feature is set to its mean."""
        base = self.raw_logit(X)
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            Xo = X.copy()
            Xo[:, j] = self.mean_[j]
            out[:, j] = base - self.raw_logit(Xo)
        return out


REGISTRY = {"logit": LogitModel, "enet": EnetModel, "ridge": RidgeRankModel, "lgbm": LGBMModel,
            "lgbm_tuned": TunedLGBMModel, "mlp": MLPModel, "xgb": XGBModel}


def make_model(name: str, params: dict | None = None, seed: int = 0) -> BaseModel:
    params = dict(params or {})
    if name == "mlp" and "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return REGISTRY[name](seed=seed, **params)


def is_complex(name: str) -> bool:
    return name in {"lgbm", "lgbm_tuned", "mlp", "xgb"}
