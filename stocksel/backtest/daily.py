"""Daily news strategy, tested separately with a harsher cost model.

Signal at the close of day d from articles first usable on day d; enter at the
open of d+1, hold ``holding_days`` open-to-open intervals; dollar-neutral long
top / short bottom sentiment among stocks with news (equal weight, 1x gross).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle
from .costs import CostModel


def daily_eligible(bundle: DataBundle, cfg) -> pd.DataFrame:
    u = cfg.universe
    adv = bundle.adv_usd(63)
    ok = (bundle.close >= u.min_price) & (bundle.market_cap >= u.min_market_cap) & (adv >= u.min_adv_usd)
    if u.get("require_index_membership", True):
        ok &= bundle.in_index()
    return ok.fillna(False)


def daily_news_strategy(bundle: DataBundle, ssum: pd.DataFrame, scnt: pd.DataFrame, cfg) -> pd.DataFrame:
    nd = cfg.news.daily
    H = int(nd.holding_days)
    cm = CostModel(cfg.costs, multiplier=nd.cost_multiplier)
    elig = daily_eligible(bundle, cfg).to_numpy()
    cnt = scnt.to_numpy()
    sig = np.where((cnt > 0) & elig, ssum.to_numpy() / np.where(cnt > 0, cnt, 1), np.nan)
    T, N = sig.shape
    W = np.zeros((T, N))
    frac = float(nd.top_frac)
    for t in range(T):
        row = sig[t]
        m = np.isfinite(row)
        if m.sum() < 6:
            continue
        vals = row[m]
        hi, lo = np.quantile(vals, 1 - max(frac, 0.2)), np.quantile(vals, min(frac, 0.2))
        longs = m & (row >= hi) & (row > 0)
        shorts = m & (row <= lo) & (row < 0)
        if longs.any() and shorts.any():
            W[t, longs] = 0.5 / longs.sum()
            W[t, shorts] = -0.5 / shorts.sum()
    # position held over open-to-open interval t (open t -> open t+1): cohorts formed at closes t-H..t-1
    Wc = pd.DataFrame(W).rolling(H, min_periods=1).sum().shift(1).fillna(0.0).to_numpy() / H
    op = bundle.open.to_numpy()
    oo = np.vstack([op[1:] / op[:-1] - 1, np.full((1, N), np.nan)])
    oo = np.nan_to_num(oo, nan=0.0)
    gross = (Wc * oo).sum(axis=1)
    dW = np.abs(np.diff(np.vstack([np.zeros((1, N)), Wc]), axis=0))
    capital = float(cfg.capital.initial)
    adv = bundle.adv_usd(63).to_numpy()
    vol = bundle.close.pct_change(fill_method=None).rolling(63, min_periods=21).std().to_numpy()
    rate = cm.rate(dW * capital, np.nan_to_num(adv, nan=1e6), vol)
    cost = (dW * rate).sum(axis=1)
    out = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                        "turnover": dW.sum(axis=1), "n_pos": (np.abs(Wc) > 0).sum(axis=1)}, index=bundle.dates)
    return out
