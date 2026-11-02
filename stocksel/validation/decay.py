"""Signal decay: rolling IC per signal, pre/post publication, haircut single-signal performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.registry import EXPECTED_SIGN, TIME_SERIES_FEATURES


def signal_ic(X: pd.DataFrame, excess: pd.Series) -> pd.DataFrame:
    """Monthly Spearman IC of each (sign-adjusted) signal with the forward excess return."""
    df = X.copy()
    df["__ex__"] = excess.reindex(X.index).to_numpy()
    df = df[df["__ex__"].notna()]
    cols = [c for c in X.columns if c not in TIME_SERIES_FEATURES]

    def ics(g):
        r = g[cols + ["__ex__"]].rank()
        return r[cols].corrwith(r["__ex__"])
    ic = df.groupby(level=0).apply(ics)
    for c in cols:
        ic[c] *= EXPECTED_SIGN.get(c, 1)
    return ic


def decay_table(ic: pd.DataFrame, cfg, recent_months: int = 36, dead_t: float = 0.5) -> pd.DataFrame:
    rows = {}
    reg = cfg.signals.registry
    for c in ic.columns:
        s = ic[c].dropna()
        if len(s) < 12:
            continue
        year = reg.get(c, {}).get("year")
        t = lambda x: float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
        pre = s[s.index.year < year] if year else s.iloc[:0]
        post = s[s.index.year >= year] if year else s
        rec = s.iloc[-recent_months:]
        rows[c] = {"ic_mean": float(s.mean()), "ic_t": t(s), "ic_pre_pub": float(pre.mean()) if len(pre) else np.nan,
                   "ic_post_pub": float(post.mean()) if len(post) else np.nan,
                   f"ic_last_{recent_months}m": float(rec.mean()), f"t_last_{recent_months}m": t(rec),
                   "pub_year": year, "status": "dead?" if (t(rec) if len(rec) > 2 else 0) < dead_t else "alive"}
    return pd.DataFrame(rows).T


def rolling_ic(ic: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    return ic.rolling(window, min_periods=max(12, window // 2)).mean()
