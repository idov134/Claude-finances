"""Cross-sectional normalization (per decision date, using only that date's eligible stocks)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional(df: pd.DataFrame, mask: pd.DataFrame | None = None, method: str = "rank",
                    winsor: float = 0.01, fill: float = 0.0) -> pd.DataFrame:
    x = df.astype(float)
    if mask is not None:
        x = x.where(mask.reindex_like(x).fillna(False).astype(bool))
    if method == "rank":
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
