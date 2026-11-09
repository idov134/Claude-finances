"""Calibrated model fitting and walk-forward out-of-sample prediction.

Per retrain date R (rolling window, retrained every k months):
  train rows : decision date in [R - window, R) AND label end t1 <= R (purging)
  fit part A : earlier rows; calibration part B: the last ``calibration_frac`` of
               training dates (A is additionally purged so its labels end before B starts)
  predict    : decision dates in [R, next R)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..validation.splits import walk_forward_schedule
from .base import make_model
from .calibration import Calibrator, ExpectedReturnMap


@dataclass
class FittedModel:
    name: str
    model: object
    calibrator: Calibrator
    er_map: ExpectedReturnMap
    features: list
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train: int

    def predict(self, X: np.ndarray) -> dict:
        s = self.model.raw_logit(X)
        p = self.calibrator.transform(s)
        return {"raw": s, "prob": p, "er": self.er_map.predict(p)}

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Feature attributions in calibrated-logit units."""
        return self.model.contributions(X) * self.calibrator.slope


def xs_rank_target(excess: pd.Series) -> pd.Series:
    return excess.groupby(level=0).rank(pct=True) - 0.5


def fit_calibrated(name: str, X: pd.DataFrame, L: pd.DataFrame, cfg, seed: int = 0) -> FittedModel:
    d = X.index.get_level_values(0)
    udates = np.sort(d.unique())
    frac = float(cfg.models.calibration_frac)
    cal_start = udates[max(1, int(len(udates) * (1 - frac)))]
    A = (d < cal_start) & (pd.to_datetime(L["t1"]).to_numpy() < cal_start)
    B = d >= cal_start
    yreg = xs_rank_target(L["excess"]).to_numpy()
    y_all = L["y"].to_numpy()
    target = cfg.labels.get("target", "binary")
    fitA = A
    if target == "tails" and name != "ridge":
        # Learn from the informative extremes only: top vs bottom ``tail_frac`` of each month's
        # excess returns (the middle is mostly noise around zero). Calibration below still uses
        # every row and the full binary event, so probabilities keep their stated meaning.
        q = float(cfg.labels.get("tail_frac", 0.3))
        pct = yreg + 0.5
        fitA = A & ((pct <= q) | (pct >= 1 - q))
        y_all = np.where(fitA, (pct >= 1 - q).astype(float), y_all)
    elif target not in ("binary", "tails"):
        raise ValueError("labels.target must be 'binary' or 'tails'")
    Xa, Xb = X.to_numpy()[fitA], X.to_numpy()[B]
    ya, yb = y_all[fitA], L["y"].to_numpy()[B]
    w = 1.0 / max(1, int(cfg.labels.horizon_months))  # average-uniqueness weight for overlapping labels
    model = make_model(name, cfg.models.params.get(name, {}), seed=seed)
    if hasattr(model, "set_time_index"):  # inner purged CV needs sample start/end times
        model.set_time_index(d[fitA], pd.to_datetime(L["t1"]).to_numpy()[fitA])
    model.fit(Xa, ya, yreg[fitA], np.full(fitA.sum(), w))
    sb = model.raw_logit(Xb)
    cal = Calibrator(cfg.models.calibration).fit(sb, yb)
    er = ExpectedReturnMap().fit(cal.transform(sb), L["excess"].to_numpy()[B])
    return FittedModel(name, model, cal, er, list(X.columns), pd.Timestamp(udates[0]), pd.Timestamp(udates[-1]),
                       int(fitA.sum()))


@dataclass
class WalkForwardResult:
    preds: pd.DataFrame                      # (date, ticker) -> prob_<m>, er_<m>, raw_<m>
    models: dict = field(default_factory=dict)  # retrain date -> {name: FittedModel}
    schedule: list = field(default_factory=list)


def _fit_window(names, Xtr, Ltr, Xte, cfg, seed):
    out, fitted = {}, {}
    for n in names:
        fm = fit_calibrated(n, Xtr, Ltr, cfg, seed)
        pr = fm.predict(Xte.to_numpy())
        for k, v in pr.items():
            out[f"{k}_{n}"] = v
        fitted[n] = fm
    return pd.DataFrame(out, index=Xte.index), fitted


def walk_forward(X: pd.DataFrame, L: pd.DataFrame, names: list[str], cfg, keep_models: bool = True,
                 test_start: pd.Timestamp | None = None) -> WalkForwardResult:
    wf = cfg.validation.walk_forward
    d = X.index.get_level_values(0)
    udates = pd.DatetimeIndex(np.sort(d.unique()))
    sched = walk_forward_schedule(udates, int(wf.min_train_months), wf.get("train_window_months"),
                                  int(wf.retrain_every_months))
    t1 = pd.to_datetime(L["t1"])
    labelled = L["y"].notna().to_numpy()
    jobs = []
    for start, R, end in sched:
        tr = (d >= start) & (d < R) & labelled & (t1 <= R).to_numpy()
        te = (d >= R) & (d < end)
        if tr.sum() < 500 or te.sum() == 0 or (test_start is not None and end <= test_start):
            continue
        jobs.append((R, X[tr], L[tr], X[te]))
    seed = int(cfg.models.seed)
    res = Parallel(n_jobs=int(cfg.models.get("n_jobs", 1)))(
        delayed(_fit_window)(names, Xtr, Ltr, Xte, cfg, seed) for _, Xtr, Ltr, Xte in jobs)
    preds = pd.concat([r[0] for r in res]).sort_index() if res else pd.DataFrame()
    models = {jobs[k][0]: res[k][1] for k in range(len(jobs))} if keep_models else {}
    return WalkForwardResult(preds=preds, models=models, schedule=[j[0] for j in jobs])


def smooth_predictions(preds: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Exponential smoothing of each stock's prediction across consecutive decision dates.

    s~_t = (1 - weight) * s_t + weight * s~_{t-1}, only when the stock was also scored at the
    previous decision date; otherwise s~_t = s_t. Uses past predictions only. Damps month-to-month
    rank churn (fast signals such as reversal/PEAD), trading a little signal for lower turnover.
    """
    if not weight:
        return preds
    if not 0 < weight < 1:
        raise ValueError("score smoothing weight must be in [0, 1)")
    cols = [c for c in ("prob", "score", "er") if c in preds.columns]
    out = preds.copy()
    dates = preds.index.get_level_values(0).unique().sort_values()
    prev = None
    for d in dates:
        cur = preds.loc[d, cols]
        if prev is not None:
            common = cur.index.intersection(prev.index)
            cur = cur.copy()
            cur.loc[common] = (1 - weight) * cur.loc[common] + weight * prev.loc[common]
        mask = out.index.get_level_values(0) == d
        out.loc[mask, cols] = cur.reindex(out.index[mask].get_level_values(1)).to_numpy()
        prev = cur
    return out


def model_view(preds: pd.DataFrame, name: str) -> pd.DataFrame:
    """Columns prob/score/er for one model (score = probability with a tiny raw-score tie-break)."""
    raw = preds[f"raw_{name}"]
    z = (raw - raw.groupby(level=0).transform("mean")) / raw.groupby(level=0).transform("std").replace(0, 1)
    return pd.DataFrame({"prob": preds[f"prob_{name}"], "score": preds[f"prob_{name}"] + 1e-6 * z.fillna(0),
                         "er": preds[f"er_{name}"]})
