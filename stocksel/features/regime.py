"""Market regime features (time series, same value for every stock on a date)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle
from .normalize import expanding_zscore


def regime_frame(bundle: DataBundle, dates: pd.DatetimeIndex) -> pd.DataFrame:
    bc = bundle.benchmark["close"]
    trend = bc / bc.rolling(210, min_periods=150).mean() - 1
    rv = bc.pct_change().rolling(21, min_periods=15).std() * np.sqrt(252)
    vix = bundle.vix.reindex(bundle.dates).ffill() if bundle.vix is not None else rv * 100
    df = pd.DataFrame({"trend": trend, "rvol": rv, "vix": vix}).loc[dates]
    return df


def risk_off(bundle: DataBundle, dates: pd.DatetimeIndex, vix_threshold: float) -> pd.Series:
    r = regime_frame(bundle, dates)
    return (r["trend"] < 0) & (r["vix"] > vix_threshold)


def regime_features(bundle: DataBundle, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    r = regime_frame(bundle, dates)
    tick = bundle.tickers
    out = {}
    for name, col in [("regime_trend", "trend"), ("regime_vol", "rvol"), ("regime_vix", "vix")]:
        z = expanding_zscore(r[col])
        out[name] = pd.DataFrame(np.repeat(z.to_numpy()[:, None], len(tick), axis=1), index=dates, columns=tick)
    return out
