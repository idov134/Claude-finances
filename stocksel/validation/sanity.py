"""Sanity checks that a real edge must pass and a leak would fail.

1. Shuffled labels: a model trained on labels permuted within each date must
   show no OOS skill on the true outcomes (AUC ~ 0.5, no excess return).
2. Influence: removing the 0.1% most influential stock-month contributions must
   not erase the result.
3. Noise canary: a pure-noise feature must not rank among the top signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.train import fit_calibrated, model_view, walk_forward
from .metrics import classification_metrics, sharpe


def shuffle_within_date(L: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    L2 = L.copy()
    d = L.index.get_level_values(0)
    perm = np.arange(len(L))
    for _, idx in pd.Series(np.arange(len(L)), index=d).groupby(level=0):
        v = idx.to_numpy()
        perm[v] = rng.permutation(v)
    L2[["excess", "y", "fwd"]] = L[["excess", "y", "fwd"]].to_numpy()[perm]
    return L2


def shuffled_label_test(X, L, cfg, backtest_fn=None, names=("logit",), n_perm: int = 5,
                        true_ic: float | None = None) -> dict:
    """Permutation test. Each permutation shuffles labels within every date and re-runs the full
    walk-forward. A single shuffle is not enough: a model fit on noise learns a random direction that
    can align with true signals by chance, so we look at the *distribution* over permutations.

    Pass if (a) the mean OOS AUC of shuffled models (vs true labels) is within 0.01 of 0.5,
    (b) the mean shuffled IC is statistically indistinguishable from 0 (|t| < 2 across permutations),
    and (c) the real model's IC exceeds every shuffled IC.
    Portfolio returns of shuffled models are reported for information only (their raw excess vs the
    benchmark mostly reflects market exposure, not skill).
    """
    rows = []
    for k in range(n_perm):
        Ls = shuffle_within_date(L, seed=int(cfg.models.seed) + 101 * k)
        wf = walk_forward(X, Ls, list(names), cfg, keep_models=False)
        for n in names:
            v = model_view(wf.preds, n)
            Lt = L.reindex(v.index)
            cm = classification_metrics(v["prob"], Lt["y"], Lt["excess"], v.index.get_level_values(0))
            row = {"perm": k, "model": n, "auc_vs_true": cm.get("auc"), "ic_vs_true": cm.get("ic_mean")}
            if backtest_fn is not None and k == 0:
                bt = backtest_fn(v)
                ex = (bt.net - bt.bench).dropna()
                row["excess_ann_info"] = float(ex.mean() * 12)
            rows.append(row)
    df = pd.DataFrame(rows)
    ics, aucs = df["ic_vs_true"].astype(float), df["auc_vs_true"].astype(float)
    ic_t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 1 and ics.std() > 0 else 0.0
    a = abs(float(aucs.mean()) - 0.5) < 0.01
    b = abs(ic_t) < 2.0
    c = true_ic is None or bool(true_ic > ics.max())
    return {"passed": bool(a and b and c), "n_perm": n_perm, "mean_auc": float(aucs.mean()),
            "mean_ic": float(ics.mean()), "max_ic": float(ics.max()), "ic_t_across_perms": ic_t,
            "true_ic": true_ic, "detail": df.to_dict("records"),
            "rule": "mean shuffled |AUC-0.5| < 0.01, mean shuffled IC not significant, real IC > every shuffled IC"}


def influence_test(bt, frac: float = 0.001) -> dict:
    c = bt.contrib
    if c is None or len(c) == 0:
        return {"passed": False, "reason": "no contributions"}
    k = max(1, int(np.ceil(len(c) * frac)))
    top = c.nlargest(k, "contrib")
    adj = bt.monthly["net"].copy()
    removed = top.groupby("date")["contrib"].sum()
    adj.loc[removed.index] -= removed
    ex0 = float((bt.monthly["net"] - bt.monthly["bench"]).mean() * 12)
    ex1 = float((adj - bt.monthly["bench"]).mean() * 12)
    sr0, sr1 = sharpe(bt.monthly["net"] - bt.monthly["rf"]), sharpe(adj - bt.monthly["rf"])
    passed = (ex0 <= 0) or (ex1 > 0 and ex1 >= 0.5 * ex0)
    return {"passed": bool(passed), "n_removed": int(k), "excess_before": ex0, "excess_after": ex1,
            "sharpe_before": sr0, "sharpe_after": sr1,
            "rule": "if the strategy has positive excess, it must keep >= 50% of it after removing the top 0.1%"}


def noise_canary(X: pd.DataFrame, L: pd.DataFrame, cfg, model: str = "lgbm", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    Xn["__noise__"] = rng.standard_normal(len(X))
    lab = L["y"].notna().to_numpy()
    d = Xn.index.get_level_values(0)
    last = np.sort(d[lab].unique())
    window = last[-min(len(last), 120):]
    m = lab & np.isin(d, window)
    fm = fit_calibrated(model, Xn[m], L[m], cfg, seed)
    imp = np.abs(fm.contributions(Xn[m].to_numpy()[-20000:])).mean(0)
    order = pd.Series(imp, index=Xn.columns).sort_values(ascending=False)
    rank = int(list(order.index).index("__noise__")) + 1
    return {"passed": rank > 3, "noise_rank": rank, "n_features": len(order), "importance": order.to_dict(),
            "rule": "a pure-noise feature must not be among the top-3 by mean |SHAP|"}
