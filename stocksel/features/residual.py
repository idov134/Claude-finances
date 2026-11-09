"""Residual (idiosyncratic) momentum — Blitz, Huij & Martens (2011); Gutierrez & Pirinsky (2007).

At each month-end decision date t, every stock's monthly returns over the trailing 36 months
(t-35..t, at least 24 valid) are regressed on the market (benchmark) return and the stock's
sector equal-weight return. Residual momentum is the sum of the residuals over the formation
window — months t-11..t-1 for 12-1, t-5..t-1 for 6-1 (the most recent month is skipped, as in
standard momentum) — divided by residual volatility x sqrt(months): a t-statistic-like score.

Everything uses month-end closes up to and including t, so the value at t is known at t's close.
Removing market and sector components strips the parts of past returns that come from beta and
industry moves; published evidence shows the residual version has a higher Sharpe ratio and
milder crashes than total-return momentum. On this repository's synthetic market the planted
momentum is idiosyncratic, so a gain there is partly by construction — see docs/algo_research_v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle


def _roll(a: np.ndarray, w: int) -> np.ndarray:
    """Rolling sum over the last ``w`` rows (inclusive), NaN treated as 0."""
    c = np.cumsum(np.nan_to_num(a), axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def monthly_returns(bundle: DataBundle, dates: pd.DatetimeIndex):
    c = bundle.close.reindex(dates)
    r = c / c.shift(1) - 1
    b = bundle.benchmark["close"].reindex(dates)
    m = (b / b.shift(1) - 1).to_numpy()
    return r, m


def residual_momentum(bundle: DataBundle, dates: pd.DatetimeIndex, window: int = 36, min_obs: int = 24,
                      formations: dict | None = None) -> dict[str, pd.DataFrame]:
    formations = formations or {"rmom_12_1": 11, "rmom_6_1": 5}
    dates = pd.DatetimeIndex(dates)
    r_df, m = monthly_returns(bundle, dates)
    R = r_df.to_numpy()
    T, N = R.shape
    sec = bundle.sectors.reindex(r_df.columns).fillna("Unknown").to_numpy()
    S = np.full((T, N), np.nan)
    for s in np.unique(sec):
        cols = sec == s
        valid_n = np.isfinite(R[:, cols]).sum(axis=1, keepdims=True)
        total = np.nansum(R[:, cols], axis=1, keepdims=True)
        # sector equal-weight return; months with no valid member stay NaN (no empty-slice warning)
        S[:, cols] = np.where(valid_n > 0, total / np.maximum(valid_n, 1), np.nan)
    M = np.repeat(m[:, None], N, axis=1)
    valid = np.isfinite(R) & np.isfinite(S) & np.isfinite(M)
    Rv, Mv, Sv = np.where(valid, R, 0.0), np.where(valid, M, 0.0), np.where(valid, S, 0.0)
    v = valid.astype(float)
    n = _roll(v, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mr, mm, ms = _roll(Rv, window) / n, _roll(Mv, window) / n, _roll(Sv, window) / n
        crm = _roll(Rv * Mv, window) / n - mr * mm
        crs = _roll(Rv * Sv, window) / n - mr * ms
        cmm = _roll(Mv * Mv, window) / n - mm * mm
        css = _roll(Sv * Sv, window) / n - ms * ms
        cms = _roll(Mv * Sv, window) / n - mm * ms
        crr = _roll(Rv * Rv, window) / n - mr * mr
        det = cmm * css - cms * cms
        bm = (crm * css - crs * cms) / det
        bs = (crs * cmm - crm * cms) / det
        alpha = mr - bm * mm - bs * ms
        resid_var = np.clip(crr - bm * crm - bs * crs, 0, None) * n / np.maximum(n - 3, 1)
        resid_sd = np.sqrt(resid_var)
    ok_fit = (n >= min_obs) & np.isfinite(det) & (np.abs(det) > 1e-14) & (resid_sd > 0)
    out = {}
    for name, k in formations.items():
        # formation months t-k .. t-1  ->  rolling k-sum ending at t-1
        def lag_sum(a):
            s = _roll(a, k)
            s = np.vstack([np.full((1, N), np.nan), s[:-1]])
            return s
        sr, sm, ss, nk = lag_sum(Rv), lag_sum(Mv), lag_sum(Sv), lag_sum(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            resid_sum = sr - nk * alpha - bm * sm - bs * ss
            score = resid_sum / (resid_sd * np.sqrt(nk))
        score[~ok_fit | (nk < max(3, int(0.8 * k)))] = np.nan
        out[name] = pd.DataFrame(score, index=dates, columns=r_df.columns)
    return out
