"""Online-gated ensemble.

A complex model (LightGBM / MLP / XGBoost) joins the ensemble at retrain date R
only if, on out-of-sample months fully realized before R, its net-of-cost
portfolio returns beat the logistic baseline with bootstrap probability >=
``models.gate.min_prob_outperform`` (and at least 24 months of evidence).
The gate never looks past R, so the ensemble's track record stays out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..validation.metrics import stationary_bootstrap_idx
from .base import is_complex
from .train import model_view


def prob_outperform(diff: np.ndarray, block: int, n: int = 1000, seed: int = 0) -> float:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 6:
        return 0.0
    rng = np.random.default_rng(seed)
    means = np.array([diff[stationary_bootstrap_idx(len(diff), block, rng)].mean() for _ in range(n)])
    return float((means > 0).mean())


def gate_decisions(net_by_model: dict[str, pd.Series], exit_dates: pd.Series, schedule: list, cfg,
                   min_months: int = 24) -> dict:
    """{retrain date: {model: admitted?}} using only months whose exit <= retrain date."""
    base = cfg.models.baseline
    g = cfg.models.gate
    out = {}
    for R in schedule:
        dec = {base: True}
        b = net_by_model.get(base)
        for m, r in net_by_model.items():
            if m == base:
                continue
            if not is_complex(m):
                dec[m] = True  # linear models (ridge / enet) are baselines too
                continue
            if b is None:
                dec[m] = False
                continue
            realized = exit_dates.reindex(r.index) <= R
            diff = (r - b).loc[realized[realized].index].dropna()
            dec[m] = bool(len(diff) >= min_months and diff.mean() > 0 and
                          prob_outperform(diff.to_numpy(), int(g.block_size)) >= float(g.min_prob_outperform))
        out[R] = dec
    return out


def ensemble_predictions(preds: pd.DataFrame, decisions: dict, names: list[str]) -> pd.DataFrame:
    d = preds.index.get_level_values(0)
    sched = sorted(decisions)
    parts = []
    for k, R in enumerate(sched):
        end = sched[k + 1] if k + 1 < len(sched) else d.max() + pd.Timedelta(days=1)
        m = (d >= R) & (d < end)
        members = [n for n in names if decisions[R].get(n, False)]
        sub = preds[m]
        views = [model_view(sub, n) for n in members]
        probs = np.column_stack([v["prob"].to_numpy() for v in views])
        ers = np.column_stack([v["er"].to_numpy() for v in views])
        scores = np.column_stack([v["score"].to_numpy() for v in views])
        parts.append(pd.DataFrame({"prob": probs.mean(1), "score": scores.mean(1), "er": ers.mean(1),
                                   "prob_std": probs.std(1), "n_members": len(members),
                                   "members": ",".join(members)}, index=sub.index))
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()
