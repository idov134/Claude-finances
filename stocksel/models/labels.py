"""Labels: forward excess return from the next open to the open after the horizon.

* entry = open of the first trading day after the decision date (signal at close);
* exit  = entry open of the decision date ``horizon`` months later;
* delisted before exit => last close x (1 + delisting return) — never dropped;
* ``t1`` (label end time) is returned for purging/embargo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle, last_valid_positions


@dataclass
class Labels:
    fwd: pd.DataFrame        # stock forward returns (decision dates x tickers)
    bench: pd.Series         # benchmark forward return
    excess: pd.DataFrame     # fwd - benchmark (or - cross-sectional median)
    entry: pd.Series         # entry date per decision date
    exit: pd.Series          # exit date (t1) per decision date


def entry_exit(bundle: DataBundle, dates: pd.DatetimeIndex, horizon: int):
    pos = bundle.dates.searchsorted(dates, side="right")          # next trading day
    entry = np.where(pos < len(bundle.dates), pos, -1)
    exit_ = np.full(len(dates), -1)
    if len(dates) > horizon:
        exit_[:-horizon] = entry[horizon:]
    return entry, exit_


def forward_returns(bundle: DataBundle, dates: pd.DatetimeIndex, horizon: int = 1,
                    benchmark: str = "index", default_delist: float = -0.30,
                    universe: pd.DataFrame | None = None) -> Labels:
    dates = pd.DatetimeIndex(dates)
    entry, exit_ = entry_exit(bundle, dates, horizon)
    O = bundle.open.to_numpy()
    C = bundle.close.to_numpy()
    K, N = len(dates), O.shape[1]
    fwd = np.full((K, N), np.nan)
    lv = last_valid_positions(bundle.close)
    dr = bundle.delist_returns.reindex(bundle.tickers).fillna(default_delist).to_numpy()
    last_row = len(bundle.dates) - 1
    for k in range(K):
        e, x = entry[k], exit_[k]
        if e < 0 or x < 0:
            continue
        o_e = O[e]
        r = O[x] / o_e - 1
        # delisted between entry and exit (stopped trading for good before x)
        gone = np.isnan(O[x]) & np.isfinite(o_e) & (lv >= e) & (lv < x) & (lv < last_row)
        if gone.any():
            r[gone] = C[lv[gone], np.where(gone)[0]] * (1 + dr[gone]) / o_e[gone] - 1
        fwd[k] = r
    fwd = pd.DataFrame(fwd, index=dates, columns=bundle.tickers)
    bo = bundle.benchmark["open"].to_numpy()
    b = np.full(K, np.nan)
    ok = (entry >= 0) & (exit_ >= 0)
    b[ok] = bo[exit_[ok]] / bo[entry[ok]] - 1
    bench = pd.Series(b, index=dates)
    if benchmark == "xs_median":
        ref = fwd.where(universe) if universe is not None else fwd
        excess = fwd.sub(ref.median(axis=1), axis=0)
    else:
        excess = fwd.sub(bench, axis=0)
    to_date = lambda arr: pd.Series([bundle.dates[i] if i >= 0 else pd.NaT for i in arr], index=dates)
    return Labels(fwd=fwd, bench=bench, excess=excess, entry=to_date(entry), exit=to_date(exit_))


def label_panel(labels: Labels, index: pd.MultiIndex) -> pd.DataFrame:
    """Long-format labels aligned to a feature panel index."""
    d = index.get_level_values(0)
    t = index.get_level_values(1)
    ex = labels.excess.stack(future_stack=True)
    fw = labels.fwd.stack(future_stack=True)
    out = pd.DataFrame({"excess": ex.reindex(index).to_numpy(), "fwd": fw.reindex(index).to_numpy()}, index=index)
    out["y"] = (out["excess"] > 0).astype(float).where(out["excess"].notna())
    out["t1"] = labels.exit.reindex(d).to_numpy()
    out["entry"] = labels.entry.reindex(d).to_numpy()
    return out
