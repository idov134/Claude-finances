"""Daily long/flat rule engine: signal at the close, trade at the next open.

A position decided at the close of day t is executed at the open of t+1 and earns the
open-to-open return r_oo[t+1] = open[t+2] / open[t+1] - 1. Uninvested capital earns the T-bill
rate. Costs are charged per unit of weight traded at each open. Weights are target-to-target:
drift between trades is ignored, and buy-and-hold is treated the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def open_to_open(open_: pd.DataFrame | pd.Series):
    """r[d] = open[d+1] / open[d] - 1: the return of a position held from the open of d to the next open."""
    return open_.shift(-1) / open_ - 1


def simulate(signal: pd.DataFrame, base_weight: pd.DataFrame, r_oo: pd.DataFrame, rf_daily: pd.Series,
             cost_bps: float) -> pd.DataFrame:
    """Daily portfolio from close-of-day signals.

    signal       dates x assets, decided at the close of t: 1 long, 0 or NaN cash
    base_weight  dates x assets, the weight an asset gets when long (also decided at the close of t)
    r_oo         open-to-open returns (see ``open_to_open``)
    rf_daily     cash return per day
    Returns a frame indexed by execution day with: ret (net), gross, exposure, turnover, cost, entries.
    Days without any next-open price (the last day of data) are dropped.
    """
    w = signal.fillna(0.0) * base_weight.reindex_like(signal).fillna(0.0)
    w_exec = w.shift(1).fillna(0.0)                         # held from the open of d to the open of d+1
    prev = w_exec.shift(1).fillna(0.0)
    r = r_oo.reindex_like(w_exec)
    turnover = (w_exec - prev).abs().sum(axis=1)
    gross = (w_exec * r.fillna(0.0)).sum(axis=1)
    exposure = w_exec.sum(axis=1)
    rf = rf_daily.reindex(w_exec.index).ffill().fillna(0.0)
    cost = turnover * cost_bps / 1e4
    entries = ((w_exec > 0) & (prev <= 0)).sum(axis=1)
    out = pd.DataFrame({"ret": gross + (1.0 - exposure).clip(lower=0.0) * rf - cost, "gross": gross,
                        "exposure": exposure, "turnover": turnover, "cost": cost, "entries": entries})
    return out[r.notna().any(axis=1)]


def monthly(daily: pd.Series) -> pd.Series:
    """Compound daily returns into calendar months (indexed by month end)."""
    return (1.0 + daily).resample("ME").prod() - 1.0


def shift_stats(pos_exec: np.ndarray, r: np.ndarray, rf: np.ndarray, bench_m_ex: np.ndarray,
                month_starts: np.ndarray, cost: float, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Annualized OLS alpha (vs ``bench_m_ex``) and daily max drawdown of one asset's position series,
    circularly shifted by each k.

    Shifting keeps the rule's exposure and trading pattern but breaks its alignment with prices: the
    distribution over shifts is the "random timing with the same habits" null. A rule that holds
    less stock always has smaller drawdowns; only a drawdown milder than most shifts shows that the
    *timing* helped. k = 0 gives the actual rule, computed the same way.
    """
    P = np.stack([np.roll(pos_exec, int(k)) for k in shifts], axis=1)
    dP = np.abs(np.diff(P, axis=0, prepend=P[:1]))
    R = P * r[:, None] + (1.0 - P) * rf[:, None] - cost * dP
    L = np.log1p(R)
    M = np.expm1(np.add.reduceat(L, month_starts, axis=0))
    rf_m = np.expm1(np.add.reduceat(np.log1p(rf), month_starts))
    Y = M - rf_m[:, None]
    x = bench_m_ex
    xc = x - x.mean()
    beta = (xc[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()
    alpha = (Y.mean(axis=0) - beta * x.mean()) * 12.0
    eq = np.exp(np.cumsum(L, axis=0))
    dd = (eq / np.maximum(np.maximum.accumulate(eq, axis=0), 1.0) - 1.0).min(axis=0)
    return alpha, dd
