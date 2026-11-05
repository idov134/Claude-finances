"""Polygon.io provider (prices incl. delisted tickers, filings-dated financials, news summaries).

News from Polygon carries a description, not the full article body — the
article-level LLM requirement needs a full-text vendor; scores computed on
summaries are flagged as such. Untested against the live API here.
"""
from __future__ import annotations

import pandas as pd
import requests

from .base import DataProvider, assemble

API = "https://api.polygon.io"


class PolygonProvider(DataProvider):
    name = "polygon"
    # This adapter consumes a user-supplied ticker list and prices/current
    # shares only. Vendor capabilities are not evidence that this particular
    # load contains delisted names, PIT membership or as-reported financials.
    survivorship_free = False
    pit_fundamentals = False

    def _get(self, path, **params):
        params["apiKey"] = self.api_key()
        out, url = [], API + path
        while url:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            out.extend(js.get("results", []))
            url, params = js.get("next_url"), {"apiKey": params["apiKey"]}
        return out

    def load(self):
        start, end = self.cfg.data.start, self.cfg.data.end
        tf = self.cfg.data.get("tickers_file")
        if not tf:
            raise RuntimeError("polygon: provide data.tickers_file (and data.membership_file for PIT membership)")
        tickers = open(tf).read().split()
        rows = []
        for t in tickers:
            for bar in self._get(f"/v2/aggs/ticker/{t}/range/1/day/{start}/{end}", adjusted="true", limit=50000):
                rows.append({"date": pd.to_datetime(bar["t"], unit="ms"), "ticker": t, "open": bar["o"],
                             "high": bar["h"], "low": bar["l"], "close": bar["c"], "volume": bar["v"]})
        prices = pd.DataFrame(rows)
        shares = {}
        for t in tickers:
            info = self._get(f"/v3/reference/tickers/{t}")
            shares[t] = (info[0] if isinstance(info, list) and info else {}).get("share_class_shares_outstanding")
        prices["market_cap"] = prices["close"] * prices["ticker"].map(shares)
        bt = self.cfg.market.benchmark
        b = pd.DataFrame(self._get(f"/v2/aggs/ticker/{bt}/range/1/day/{start}/{end}", adjusted="true", limit=50000))
        bench = pd.DataFrame({"open": b["o"].values, "close": b["c"].values}, index=pd.to_datetime(b["t"], unit="ms"))
        return assemble(prices, bench, self.meta(warning=(
            "market cap from current share count; delisted coverage and PIT membership "
            "unverified; no fundamentals loaded")))
