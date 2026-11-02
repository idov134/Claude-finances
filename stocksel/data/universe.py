"""Investable universe per decision date: membership, price, size and liquidity filters.

Strict by design — results that depend on micro-caps are not implementable at
scale and are therefore excluded *before* any modelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle, first_valid_positions


def eligible(bundle: DataBundle, cfg, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    u = cfg.universe
    d = pd.DatetimeIndex(decision_dates)
    close = bundle.close.loc[d]
    mcap = bundle.market_cap.loc[d]
    adv = bundle.adv_usd(63).loc[d]
    ok = close.notna() & (close >= u.min_price) & (mcap >= u.min_market_cap) & (adv >= u.min_adv_usd)
    if u.get("require_index_membership", True):
        ok &= bundle.in_index().loc[d]
    # history requirement (no look-ahead: counts trading days since first valid price)
    first = first_valid_positions(bundle.close)
    pos = bundle.dates.get_indexer(d)
    hist = pos[:, None] - first[None, :]
    ok &= pd.DataFrame((first[None, :] >= 0) & (hist >= u.min_history_days), index=d, columns=bundle.tickers)
    pct = float(u.get("exclude_bottom_mcap_pct", 0.0))
    if pct > 0:
        m = mcap.where(ok)
        ranks = m.rank(axis=1, pct=True)
        ok &= ranks > pct
    return ok.fillna(False).astype(bool)
