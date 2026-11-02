"""Live recommendations from the frozen research spec.

Output per name: ticker | calibrated probability | expected return after haircut |
position size (% of portfolio) | entry date | planned exit date | stop |
top-3 signals (SHAP) | confidence level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.monthly import market_vol
from ..data import load_bundle
from ..features.news import make_scorer
from ..features.pipeline import build_features
from ..models.explain import ensemble_contributions, top_signals
from ..models.labels import forward_returns, label_panel
from ..models.train import fit_calibrated
from ..portfolio.sizing import SizingParams, haircut_prob, target_weights


def confidence_level(p_h: float, prob_std: float, system_validated: bool) -> str:
    if not system_validated:
        return "Low (system not validated: DSR<0.95 or holdout not passed)"
    if p_h >= 0.56 and prob_std <= 0.03:
        return "High"
    if p_h >= 0.53:
        return "Medium"
    return "Low"


def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(d) + pd.offsets.BDay(1)).normalize()


def recommend(cfg, asof: str | None = None, bundle=None, prev_weights: pd.Series | None = None,
              spec: dict | None = None) -> tuple[pd.DataFrame, dict]:
    art = Path(cfg.paths.artifacts)
    spec = spec or json.loads((art / "final_spec.json").read_text())
    full = bundle if bundle is not None else load_bundle(cfg)
    data = full.truncate(asof) if asof else full
    asof_d = data.dates[-1]
    panel = build_features(data, cfg, make_scorer(cfg))
    h = int(spec["horizon"])
    labels = forward_returns(data, panel.dates, h, cfg.labels.benchmark, cfg.data.default_delisting_return, panel.universe)
    L = label_panel(labels, panel.X.index)
    X = panel.features(spec["features"])
    d = X.index.get_level_values(0)
    window = cfg.validation.walk_forward.get("train_window_months") or 10_000
    start = asof_d - pd.DateOffset(months=int(window))
    tr = (d >= start) & (d < asof_d) & L["y"].notna().to_numpy() & (pd.to_datetime(L["t1"]) <= asof_d).to_numpy()
    cfg_h = cfg.copy_with({"labels.horizon_months": h})
    members = [fit_calibrated(m, X[tr], L[tr], cfg_h, int(cfg.models.seed)) for m in spec["members"]]
    last = panel.dates[-1]
    Xn = X[d == last]
    preds = [m.predict(Xn.to_numpy()) for m in members]
    P = np.column_stack([p["prob"] for p in preds])
    E = np.column_stack([p["er"] for p in preds])
    cand = pd.DataFrame({"prob": P.mean(1), "prob_std": P.std(1), "er": E.mean(1)}, index=Xn.index.get_level_values(1))
    cand["score"] = cand["prob"]
    a = panel.aux
    cand["mcap"] = a["mcap"].loc[last].reindex(cand.index)
    cand["idio_vol"] = a["idio_vol"].loc[last].reindex(cand.index)
    cand["beta"] = a["beta"].loc[last].reindex(cand.index)
    cand["sector"] = data.sectors.reindex(cand.index)
    keep = float(spec.get("keep", 1 - cfg.signals.haircut))
    prm = SizingParams.from_cfg(cfg, keep, float(spec["vol_target"]))
    mv = float(market_vol(data, pd.DatetimeIndex([last])).iloc[0])
    w, diag = target_weights(cand, prev_weights if prev_weights is not None else pd.Series(dtype=float), prm, mv)
    contrib = ensemble_contributions(members, Xn)
    contrib.index = Xn.index.get_level_values(1)
    tops = top_signals(contrib.loc[w.index]) if len(w) else pd.Series(dtype=str)
    entry = next_business_day(last)
    exit_planned = (entry + pd.offsets.BMonthEnd(h)) + pd.offsets.BDay(1)
    atr = a["atr"].loc[last]
    close = a["close"].loc[last]
    validated = (spec.get("dsr") or 0) >= cfg.validation.red_flags.min_dsr and bool(spec.get("holdout_passed", False))
    rows = []
    for t in w.sort_values(ascending=False).index:
        ph = float(haircut_prob(cand.at[t, "prob"], keep))
        rows.append({
            "ticker": t,
            "calibrated_prob": round(float(cand.at[t, "prob"]), 4),
            "prob_after_haircut": round(ph, 4),
            "expected_excess_after_haircut": round(float(cand.at[t, "er"]) * keep, 5),
            "position_pct": round(float(w[t]) * 100, 2),
            "entry_date": str(entry.date()) + " (open)",
            "planned_exit": str(exit_planned.date()) + " (open, or earlier on stop / rank drop)",
            "stop": round(float(close[t] - cfg.execution.atr_stop_mult * atr[t]), 2) if np.isfinite(atr.get(t, np.nan)) else None,
            "ref_close": round(float(close[t]), 2),
            "top3_signals_shap": tops.get(t, ""),
            "confidence": confidence_level(ph, float(cand.at[t, "prob_std"]), validated),
        })
    out = pd.DataFrame(rows)
    meta = {"asof": str(asof_d.date()), "n_candidates": len(cand), "gross_exposure": float(w.sum()),
            "cash_pct": float(1 - w.sum()), "diag": diag, "members": spec["members"], "system_validated": validated}
    return out, meta
