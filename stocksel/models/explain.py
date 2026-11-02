"""Attribution (SHAP) for recommendations.

* linear models: exact SHAP = coef * (x - E[x]) (independent features);
* LightGBM / XGBoost: exact TreeSHAP via ``pred_contrib``;
* MLP: occlusion approximation (feature set to its training mean).
All attributions are scaled into calibrated-logit units and averaged across
ensemble members.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_contributions(members: list, X: pd.DataFrame) -> pd.DataFrame:
    arr = np.mean([m.contributions(X.to_numpy()) for m in members], axis=0)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def top_signals(contrib: pd.DataFrame, k: int = 3) -> pd.Series:
    def fmt(row):
        top = row.reindex(row.abs().sort_values(ascending=False).index[:k])
        return "; ".join(f"{n} ({v:+.3f})" for n, v in top.items())
    return contrib.apply(fmt, axis=1)


def mean_abs_importance(contrib: pd.DataFrame) -> dict:
    return contrib.abs().mean().sort_values(ascending=False).to_dict()
