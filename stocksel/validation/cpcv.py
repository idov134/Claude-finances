"""Combinatorial Purged CV backtest paths for a fixed model set (distribution of OOS Sharpe)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.train import fit_calibrated
from .splits import CombinatorialPurgedCV


def cpcv_path_predictions(X: pd.DataFrame, L: pd.DataFrame, members: list[str], cfg) -> list[pd.DataFrame]:
    c = cfg.validation.cpcv
    cv = CombinatorialPurgedCV(int(c.n_groups), int(c.n_test_groups), float(cfg.validation.purged_kfold.embargo_pct))
    lab = L["y"].notna().to_numpy()
    Xl, Ll = X[lab], L[lab]
    d = Xl.index.get_level_values(0)
    udates = np.sort(d.unique())
    groups = [g for g in np.array_split(udates, cv.N) if len(g)]
    by_split = {}
    for test_groups, tr, te in cv.split(pd.Series(d), Ll["t1"]):
        if len(tr) < 500:
            continue
        probs, scores = [], []
        for m in members:
            fm = fit_calibrated(m, Xl.iloc[tr], Ll.iloc[tr], cfg, int(cfg.models.seed))
            pr = fm.predict(Xl.iloc[te].to_numpy())
            probs.append(pr["prob"]); scores.append(pr["prob"])
        P = pd.DataFrame({"prob": np.mean(probs, 0), "score": np.mean(scores, 0), "er": 0.0}, index=Xl.index[te])
        for g in test_groups:
            by_split[(g, test_groups)] = P[P.index.get_level_values(0).isin(groups[g])]
    paths = []
    for path in cv.paths():
        parts = [by_split.get(gs) for gs in path if gs is not None]
        parts = [p for p in parts if p is not None]
        if parts:
            paths.append(pd.concat(parts).sort_index())
    return paths
