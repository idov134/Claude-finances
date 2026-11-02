"""Robustness: sub-periods, crisis regimes, pre/post-2005, sectors, size, parameter perturbations."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .metrics import max_drawdown, sharpe


def _stats(net, bench, rf):
    ex = net - bench
    return {"n": int(len(net)), "ret_ann": float(net.mean() * 12), "bench_ann": float(bench.mean() * 12),
            "excess_ann": float(ex.mean() * 12), "sharpe": sharpe(net - rf), "bench_sharpe": sharpe(bench - rf),
            "ir": sharpe(ex), "max_dd": max_drawdown(net), "bench_max_dd": max_drawdown(bench),
            "hit_rate_vs_bench": float((ex > 0).mean()) if len(ex) else np.nan}


def period_table(monthly: pd.DataFrame, periods: dict) -> pd.DataFrame:
    rows = {}
    for name, (s, e) in periods.items():
        m = monthly.loc[(monthly.index >= s) & (monthly.index <= e)]
        if len(m) >= 2:
            rows[name] = _stats(m["net"], m["bench"], m["rf"])
    return pd.DataFrame(rows).T


def yearly_table(monthly: pd.DataFrame) -> pd.DataFrame:
    g = monthly.groupby(monthly.index.year)
    return pd.DataFrame({y: {"ret": float((1 + m["net"]).prod() - 1), "bench": float((1 + m["bench"]).prod() - 1),
                             "excess": float((1 + m["net"]).prod() - (1 + m["bench"]).prod())} for y, m in g}).T


def split_table(monthly: pd.DataFrame, break_date: str) -> pd.DataFrame:
    b = pd.Timestamp(break_date)
    return period_table(monthly, {f"pre_{b.year}": (monthly.index.min(), b - pd.Timedelta(days=1)),
                                  f"post_{b.year}": (b, monthly.index.max())})


def market_state_table(monthly: pd.DataFrame, regime: pd.DataFrame, vix_threshold: float) -> pd.DataFrame:
    r = regime.reindex(monthly.index)
    out = {}
    for name, mask in {"bull (index > 10m SMA)": r["trend"] > 0, "bear (index < 10m SMA)": r["trend"] <= 0,
                       f"VIX > {vix_threshold:g}": r["vix"] > vix_threshold,
                       f"VIX <= {vix_threshold:g}": r["vix"] <= vix_threshold}.items():
        m = monthly[mask.fillna(False).to_numpy()]
        if len(m) >= 3:
            out[name] = _stats(m["net"], m["bench"], m["rf"])
    return pd.DataFrame(out).T


def sector_exclusion(run_bt, panel, sectors: pd.Series) -> pd.DataFrame:
    rows = {}
    for s in sorted(sectors.dropna().unique()):
        mask = panel.universe & ~pd.DataFrame(np.repeat((sectors.reindex(panel.universe.columns) == s).to_numpy()[None, :],
                                                         len(panel.universe), 0),
                                               index=panel.universe.index, columns=panel.universe.columns)
        bt = run_bt(universe_mask=mask)
        rows[f"ex-{s}"] = _stats(bt.monthly["net"], bt.monthly["bench"], bt.monthly["rf"])
    return pd.DataFrame(rows).T


def size_robustness(run_bt, panel) -> pd.DataFrame:
    mcap = panel.aux["mcap"].where(panel.universe)
    big = mcap.ge(mcap.median(axis=1), axis=0) & panel.universe
    out = {}
    for name, mask in {"large half only": big, "small half only": panel.universe & ~big}.items():
        bt = run_bt(universe_mask=mask)
        out[name] = _stats(bt.monthly["net"], bt.monthly["bench"], bt.monthly["rf"])
    return pd.DataFrame(out).T


def parameter_grid(run_bt, grid: dict) -> pd.DataFrame:
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        over = dict(zip(keys, combo))
        bt = run_bt(**{"prm_over": over})
        st = _stats(bt.monthly["net"], bt.monthly["bench"], bt.monthly["rf"])
        rows.append({**over, **st, "turnover_ann": float(bt.monthly["turnover"].mean() * 12)})
    return pd.DataFrame(rows)
