"""Live monitoring: feature drift (PSI, KS), weekly comparison vs backtest, kill switch, quarterly decay review."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..portfolio.risk import evaluate


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < 20 or len(a) < 20:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    pe = np.histogram(e, edges)[0] / len(e) + 1e-6
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(train_raw: dict, recent_raw: dict, alert: float = 0.25) -> pd.DataFrame:
    """Compare raw feature distributions (training window vs recent months)."""
    rows = {}
    for k in train_raw:
        if k not in recent_raw:
            continue
        e = train_raw[k].to_numpy().ravel()
        a = recent_raw[k].to_numpy().ravel()
        e, a = e[np.isfinite(e)], a[np.isfinite(a)]
        if len(e) < 20 or len(a) < 20:
            continue
        p = psi(e, a)
        ks = stats.ks_2samp(e, a)
        rows[k] = {"psi": p, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "alert": bool(p > alert)}
    return pd.DataFrame(rows).T


def weekly_comparison(live_daily: pd.DataFrame, bt_daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly live returns placed in the backtest's weekly return distribution (percentile)."""
    lw = live_daily["equity"].resample("W-FRI").last().pct_change().dropna()
    bw = bt_daily["equity"].resample("W-FRI").last().pct_change().dropna()
    lb = live_daily["bench"].resample("W-FRI").last().pct_change().reindex(lw.index)
    pct = [float((bw < x).mean()) for x in lw]
    return pd.DataFrame({"live": lw, "bench": lb, "bt_percentile": pct,
                         "outside_90pct_band": [(p < 0.05) or (p > 0.95) for p in pct]})


def kill_switch_check(live_daily: pd.DataFrame, bt_monthly_net: pd.Series, cfg):
    eq = live_daily["equity"]
    live_m = eq.resample("ME").last().pct_change().dropna()
    return evaluate(eq, live_m, bt_monthly_net, cfg)
