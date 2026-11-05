"""Fundamental / earnings / analyst signals, joined point-in-time by publication time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle
from ..data.pit import asof_wide

FLOW = ["gross_profit", "net_income", "revenue"]
STOCK = ["total_assets", "book_equity"]


def _ttm(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["ticker", "period_end", "publish_date"]).drop_duplicates(["ticker", "period_end"], keep="first")
    g = f.groupby("ticker", sort=False)
    for col in FLOW:
        f[col + "_ttm"] = g[col].transform(lambda s: s.rolling(4, min_periods=4).sum())
    # A later-period filing may arrive before a delayed earlier quarter. The
    # aggregate cannot be available until every constituent quarter is public.
    # Delay the entire joined row conservatively, including balance-sheet fields,
    # rather than letting the time join expose future quarterly flows.
    publication_dates = [f["publish_date"]] + [g["publish_date"].shift(i) for i in range(1, 4)]
    f["publish_date"] = pd.concat(publication_dates, axis=1).max(axis=1)
    return f


def fundamental_features(bundle: DataBundle, cfg, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    tz, close_t = cfg.market.timezone, cfg.market.close_time
    tick = bundle.tickers
    out: dict[str, pd.DataFrame] = {}
    nan = pd.DataFrame(np.nan, index=dates, columns=tick)
    mcap = bundle.market_cap.loc[dates]
    px = bundle.close.loc[dates]

    if bundle.fundamentals is not None and len(bundle.fundamentals) and bundle.meta.get("pit_fundamentals", False):
        f = _ttm(bundle.fundamentals.copy())
        cols = [c + "_ttm" for c in FLOW] + STOCK
        j = asof_wide(f, "publish_date", cols, dates, tick, tz, close_t,
                      lag_days=int(cfg.data.pit_lag_days), return_avail=True)
        age = -(j["avail"].astype("datetime64[ns]").sub(pd.Series(dates, index=dates).astype("datetime64[ns]"), axis=0))
        stale = age > pd.Timedelta(days=270)
        for c in cols:
            j[c] = j[c].where(~stale)
        be = j["book_equity"].where(j["book_equity"] > 0)
        out["value_bm"] = be / mcap
        out["value_ep"] = j["net_income_ttm"] / mcap
        out["quality_gpa"] = j["gross_profit_ttm"] / j["total_assets"].where(j["total_assets"] > 0)
        out["quality_roe"] = j["net_income_ttm"] / be
    else:
        for k in ("value_bm", "value_ep", "quality_gpa", "quality_roe"):
            out[k] = nan.copy()

    if bundle.earnings is not None and len(bundle.earnings):
        e = bundle.earnings.copy()
        e["surprise"] = e["eps_actual"] - e["eps_consensus"]
        if e["eps_consensus"].isna().all():  # seasonal random walk fallback
            e = e.sort_values(["ticker", "announce_ts"])
            d4 = e.groupby("ticker")["eps_actual"].diff(4)
            e["surprise"] = d4 / d4.groupby(e["ticker"]).transform(lambda s: s.rolling(8, min_periods=4).std())
        j = asof_wide(e, "announce_ts", ["surprise"], dates, tick, tz, close_t, return_avail=True)
        dts = pd.Series(decision_ts(dates, close_t), index=dates).astype("datetime64[ns]")
        days = -(j["avail"].astype("datetime64[ns]").sub(dts, axis=0)) / pd.Timedelta(days=1)
        sue = j["surprise"] / px
        sue = sue.where(days <= 120)
        out["sue"] = sue
        out["pead"] = sue * np.exp(-days / 45.0)
    else:
        out["sue"] = nan.copy(); out["pead"] = nan.copy()

    if bundle.estimates is not None and len(bundle.estimates):
        j = asof_wide(bundle.estimates, "date", ["fy1_consensus"], dates, tick, tz, close_t, lag_days=1)
        fy1 = j["fy1_consensus"]
        out["revisions"] = fy1 / fy1.shift(3) - 1
    else:
        out["revisions"] = nan.copy()
    return out


def decision_ts(dates, close_time):
    from ..data.pit import decision_timestamps
    return decision_timestamps(dates, close_time)
