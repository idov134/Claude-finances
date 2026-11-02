"""Local CSV provider (canonical format). Use this to plug in any vendor export.

Expected files in ``cfg.data.csv_dir`` (only prices.csv and benchmark.csv are required):
  prices.csv        date,ticker,open,high,low,close,volume,market_cap   (split+dividend adjusted)
  benchmark.csv     date,open,close                                     (total-return index)
  sectors.csv       ticker,sector
  membership.csv    ticker,start,end                                    (point-in-time index membership)
  delistings.csv    ticker,delist_return
  fundamentals.csv  ticker,period_end,publish_date,total_assets,book_equity,gross_profit,net_income,revenue,eps
  earnings.csv      ticker,announce_ts,eps_actual,eps_consensus        (announce_ts in UTC ISO format)
  estimates.csv     date,ticker,fy1_consensus
  news.csv          ticker,timestamp,headline,body                      (timestamp in UTC ISO format)
  vix.csv           date,vix
  rf.csv            date,rf                                             (annualized, decimal)
Flags: put ``survivorship_free.flag`` / ``pit_fundamentals.flag`` in the folder only if true.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataProvider, assemble, membership_from_intervals


class CSVProvider(DataProvider):
    name = "csv"

    def load(self):
        d = Path(self.cfg.data.csv_dir)
        rd = lambda f, **k: pd.read_csv(d / f, **k) if (d / f).exists() else None
        prices = rd("prices.csv")
        bench = rd("benchmark.csv", index_col="date", parse_dates=True)
        if prices is None or bench is None:
            raise FileNotFoundError(f"{d}/prices.csv and {d}/benchmark.csv are required")
        self.survivorship_free = (d / "survivorship_free.flag").exists()
        self.pit_fundamentals = (d / "pit_fundamentals.flag").exists()
        sectors = rd("sectors.csv")
        sectors = sectors.set_index("ticker")["sector"] if sectors is not None else None
        dl = rd("delistings.csv")
        dl = dl.set_index("ticker")["delist_return"] if dl is not None else None
        events = {}
        for name, f, dcols in [("fundamentals", "fundamentals.csv", ["period_end", "publish_date"]),
                               ("earnings", "earnings.csv", []), ("estimates", "estimates.csv", ["date"]),
                               ("news", "news.csv", [])]:
            df = rd(f, parse_dates=dcols or False)
            if df is not None and name == "earnings":
                df["announce_ts"] = pd.to_datetime(df["announce_ts"], utc=True)
            if df is not None and name == "news":
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if df is not None and name == "fundamentals" and not self.pit_fundamentals:
                df = None  # refuse non-PIT fundamentals in backtests
            events[name] = df
        vix = rd("vix.csv", index_col="date", parse_dates=True)
        rf = rd("rf.csv", index_col="date", parse_dates=True)
        b = assemble(prices, bench, self.meta(), sectors=sectors, delist_returns=dl,
                     vix=None if vix is None else vix["vix"], risk_free=None if rf is None else rf["rf"], **events)
        memb = rd("membership.csv")
        if memb is not None:
            b.membership = membership_from_intervals(memb, b.dates, b.tickers)
        return b
