"""Provider interface + helpers to assemble a canonical ``DataBundle`` from long tables."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from ..bundle import DataBundle


class DataProvider(ABC):
    name: str = "base"
    survivorship_free: bool = False
    pit_fundamentals: bool = False

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def load(self) -> DataBundle:  # pragma: no cover - interface
        ...

    def api_key(self) -> str:
        env = self.cfg.data.api_keys_env.get(self.name)
        key = os.environ.get(env or "", "")
        if not key:
            raise RuntimeError(f"{self.name}: set the API key in environment variable {env!r} (never in config.yaml)")
        return key

    def meta(self, **extra) -> dict:
        return {"source": self.name, "survivorship_free": self.survivorship_free,
                "pit_fundamentals": self.pit_fundamentals, "sector_pit": False,
                "timezone": self.cfg.market.timezone, **extra}


def membership_from_intervals(intervals: pd.DataFrame, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """intervals: ticker, start, end (end NaN = still a member) -> bool wide frame."""
    m = pd.DataFrame(False, index=dates, columns=pd.Index(tickers))
    for row in intervals.itertuples(index=False):
        if row.ticker not in m.columns:
            continue
        end = pd.Timestamp(row.end) if pd.notna(row.end) else dates[-1]
        m.loc[(dates >= pd.Timestamp(row.start)) & (dates <= end), row.ticker] = True
    return m


def assemble(prices: pd.DataFrame, benchmark: pd.DataFrame, meta: dict, sectors: Optional[pd.Series] = None,
             membership: Optional[pd.DataFrame] = None, delist_returns: Optional[pd.Series] = None,
             **events) -> DataBundle:
    """prices: long frame with date,ticker,open,high,low,close,volume,market_cap (adjusted)."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    wide = {f: prices.pivot_table(index="date", columns="ticker", values=f, aggfunc="last")
            for f in ("open", "high", "low", "close", "volume", "market_cap")}
    dates = wide["close"].index.union(pd.to_datetime(benchmark.index)).sort_values()
    bench = benchmark.copy()
    bench.index = pd.to_datetime(bench.index).tz_localize(None).normalize()
    dates = wide["close"].index.intersection(bench.index)
    tickers = wide["close"].columns
    for f in wide:
        wide[f] = wide[f].reindex(index=dates, columns=tickers).astype(float)
    if sectors is None:
        sectors = pd.Series("Unknown", index=tickers)
    if membership is not None:
        membership = membership.reindex(index=dates, columns=tickers).fillna(False).astype(bool)
    b = DataBundle(**wide, sectors=sectors.reindex(tickers).fillna("Unknown"),
                   benchmark=bench.reindex(dates)[["open", "close"]].ffill(),
                   membership=membership,
                   delist_returns=delist_returns if delist_returns is not None else pd.Series(dtype=float),
                   meta=meta, **events)
    b.validate()
    return b
