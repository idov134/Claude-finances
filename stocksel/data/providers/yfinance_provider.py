"""yfinance provider — convenient but NOT suitable for unbiased backtests.

Limitations (reported as red flags in every report that uses it):
* no delisted tickers -> survivorship bias;
* no historical index membership (needs ``data.membership_file``);
* fundamentals are current/restated, not point-in-time -> disabled;
* market cap approximated from current shares outstanding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import DataProvider, assemble, membership_from_intervals


class YFinanceProvider(DataProvider):
    name = "yfinance"
    survivorship_free = False
    pit_fundamentals = False

    def _tickers(self) -> list[str]:
        f = self.cfg.data.get("tickers_file")
        if f and Path(f).exists():
            return [t.strip() for t in Path(f).read_text().split() if t.strip()]
        mf = self.cfg.data.get("membership_file")
        if mf and Path(mf).exists():
            return sorted(pd.read_csv(mf)["ticker"].unique())
        raise RuntimeError("yfinance needs data.tickers_file or data.membership_file (it has no constituent history)")

    def load(self):
        import yfinance as yf

        tickers = self._tickers()
        start, end = self.cfg.data.start, self.cfg.data.end
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker",
                          progress=False, threads=True)
        rows = []
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].dropna(how="all").reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["ticker"] = t
            rows.append(df)
        prices = pd.concat(rows, ignore_index=True)
        shares, sectors = {}, {}
        for t in tickers:
            try:
                info = yf.Ticker(t).get_info()
                shares[t] = info.get("sharesOutstanding", np.nan)
                sectors[t] = info.get("sector", "Unknown")
            except Exception:  # network/info failures are common; keep going
                shares[t], sectors[t] = np.nan, "Unknown"
        prices["market_cap"] = prices["close"] * prices["ticker"].map(shares)
        bt = self.cfg.market.benchmark
        b = yf.download(bt, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(b.columns, pd.MultiIndex):
            b.columns = b.columns.get_level_values(0)
        bench = b.rename(columns=str.lower)[["open", "close"]]
        vix = yf.download("^VIX", start=start, end=end, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        meta = self.meta(warning="yfinance: survivorship-biased, no PIT fundamentals")
        bundle = assemble(prices, bench, meta, sectors=pd.Series(sectors), vix=vix["Close"].rename("vix"))
        mf = self.cfg.data.get("membership_file")
        if mf and Path(mf).exists():
            bundle.membership = membership_from_intervals(pd.read_csv(mf), bundle.dates, bundle.tickers)
        return bundle
