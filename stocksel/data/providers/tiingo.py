"""Tiingo provider (adjusted EOD prices; news API returns descriptions only). Untested against the live API here."""
from __future__ import annotations

import pandas as pd
import requests

from .base import DataProvider, assemble

API = "https://api.tiingo.com"


class TiingoProvider(DataProvider):
    name = "tiingo"
    survivorship_free = False   # partial delisted coverage; treat as biased unless verified
    pit_fundamentals = False

    def _get(self, path, **params):
        params["token"] = self.api_key()
        r = requests.get(API + path, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def load(self):
        start, end = self.cfg.data.start, self.cfg.data.end
        tf = self.cfg.data.get("tickers_file")
        if not tf:
            raise RuntimeError("tiingo: provide data.tickers_file")
        rows = []
        for t in open(tf).read().split():
            for bar in self._get(f"/tiingo/daily/{t}/prices", startDate=start, endDate=end):
                rows.append({"date": bar["date"][:10], "ticker": t, "open": bar["adjOpen"], "high": bar["adjHigh"],
                             "low": bar["adjLow"], "close": bar["adjClose"], "volume": bar["adjVolume"],
                             "market_cap": float("nan")})
        prices = pd.DataFrame(rows)
        bt = self.cfg.market.benchmark
        b = pd.DataFrame(self._get(f"/tiingo/daily/{bt}/prices", startDate=start, endDate=end))
        bench = pd.DataFrame({"open": b["adjOpen"].values, "close": b["adjClose"].values},
                             index=pd.to_datetime(b["date"].str[:10]))
        return assemble(prices, bench, self.meta(warning="tiingo: no market cap; size filter inactive"))
