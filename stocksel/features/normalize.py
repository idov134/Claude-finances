"""Cross-sectional normalization (per decision date, using only that date's eligible stocks)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_rank(x: pd.DataFrame, groups: pd.Series, min_group: int = 5) -> pd.DataFrame:
    """Per-date rank within each group (e.g. sector), scaled to [-0.5, 0.5].

    Groups with fewer than ``min_group`` valid names on a date fall back to the all-name rank,
    so a thin sector cannot produce extreme scores from two or three stocks.
    """
    g = groups.reindex(x.columns).fillna("__none__")
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    r_all = x.rank(axis=1)
    n_all = x.notna().sum(axis=1)
    glob = r_all.sub(1.0).div((n_all - 1).replace(0, np.nan), axis=0) - 0.5
    for key in pd.unique(g):
        cols = g.index[g == key]
        sub = x[cols]
        n = sub.notna().sum(axis=1)
        rk = sub.rank(axis=1).sub(1.0).div((n - 1).replace(0, np.nan), axis=0) - 0.5
        thin = n < min_group
        rk.loc[thin] = glob.loc[thin, cols]
        out[cols] = rk
    return out


def cross_sectional(df: pd.DataFrame, mask: pd.DataFrame | None = None, method: str = "rank",
                    winsor: float = 0.01, fill: float = 0.0, groups: pd.Series | None = None) -> pd.DataFrame:
    """Normalize each date's cross-section. With ``groups`` (sector labels) and method ``rank``,
    ranks are taken within each group: the feature then measures standing relative to sector peers."""
    x = df.astype(float)
    if mask is not None:
        x = x.where(mask.reindex_like(x).fillna(False).astype(bool))
    if method == "rank" and groups is not None:
        out = group_rank(x, groups)
    elif method == "rank":
        r = x.rank(axis=1)
        n = x.notna().sum(axis=1)
        out = r.sub(1.0).div((n - 1).replace(0, np.nan), axis=0) - 0.5   # spans [-0.5, 0.5] for any N
    elif method == "zscore":
        lo = x.quantile(winsor, axis=1)
        hi = x.quantile(1 - winsor, axis=1)
        x = x.clip(lower=lo, upper=hi, axis=0)
        out = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
    else:
        raise ValueError(method)
    if mask is not None:
        m = mask.reindex_like(out).fillna(False).astype(bool)
        out = out.where(~m | out.notna(), fill)  # eligible but missing -> neutral
        out = out.where(m)
    return out


def expanding_zscore(s: pd.Series, min_periods: int = 24) -> pd.Series:
    """Time-series z-score using only past and current observations."""
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return ((s - mu) / sd).clip(-4, 4)
