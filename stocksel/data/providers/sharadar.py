"""Sharadar (Nasdaq Data Link) provider — the recommended source for US equities.

Why: SEP includes delisted tickers (survivorship-free), SF1 dimension ARQ is
as-reported with ``datekey`` = SEC filing date (point-in-time), SP500 holds the
historical constituent events, ACTIONS holds delisting reasons.

NOTE: written against the public Tables API documentation; it has not been run
against the live API from this repository's CI (no key / no network).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .base import DataProvider, assemble

BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json"
FAILURE_ACTIONS = {"delisted", "bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting"}


class SharadarProvider(DataProvider):
    name = "sharadar"
    survivorship_free = True
    pit_fundamentals = True

    def _table(self, table_name: str, **params) -> pd.DataFrame:
        params = {k.replace("__", "."): v for k, v in params.items()}
        params["api_key"] = self.api_key()
        frames, cursor = [], None
        while True:
            if cursor:
                params["qopts.cursor_id"] = cursor
            for attempt in range(4):
                r = requests.get(BASE.format(table=table_name), params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                break
            js = r.json()["datatable"]
            cols = [c["name"] for c in js["columns"]]
            frames.append(pd.DataFrame(js["data"], columns=cols))
            cursor = r.json().get("meta", {}).get("next_cursor_id")
            if not cursor:
                break
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load(self):
        start, end = self.cfg.data.start, self.cfg.data.end
        sp = self._table("SP500")
        sp["date"] = pd.to_datetime(sp["date"])
        tickers = sorted(sp["ticker"].unique())
        chunks = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
        sep = pd.concat([self._table("SEP", ticker=",".join(c), date__gte=start, date__lte=end) for c in chunks])
        daily = pd.concat([self._table("DAILY", ticker=",".join(c), date__gte=start, date__lte=end,
                                       qopts__columns="ticker,date,marketcap") for c in chunks])
        sep["date"] = pd.to_datetime(sep["date"]); daily["date"] = pd.to_datetime(daily["date"])
        adj = sep["closeadj"] / sep["close"]
        prices = pd.DataFrame({"date": sep["date"], "ticker": sep["ticker"], "open": sep["open"] * adj,
                               "high": sep["high"] * adj, "low": sep["low"] * adj, "close": sep["closeadj"],
                               "volume": sep["volume"] / adj})
        prices = prices.merge(daily.assign(market_cap=daily["marketcap"] * 1e6)[["date", "ticker", "market_cap"]],
                              on=["date", "ticker"], how="left")
        tick = self._table("TICKERS", table="SEP")
        sectors = tick.drop_duplicates("ticker").set_index("ticker")["sector"]
        sf1 = pd.concat([self._table("SF1", ticker=",".join(c), dimension="ARQ") for c in chunks])
        fundamentals = pd.DataFrame({
            "ticker": sf1["ticker"], "period_end": pd.to_datetime(sf1["reportperiod"]),
            "publish_date": pd.to_datetime(sf1["datekey"]), "total_assets": sf1["assets"],
            "book_equity": sf1["equity"], "gross_profit": sf1["gp"], "net_income": sf1["netinc"],
            "revenue": sf1["revenue"], "eps": sf1["eps"]})
        acts = self._table("ACTIONS", date__gte=start)
        dl = acts[acts["action"].isin(FAILURE_ACTIONS)]
        delist = pd.Series(self.cfg.data.default_delisting_return, index=dl["ticker"].unique())
        # benchmark: SPY total return from SFP (fund prices)
        spy = self._table("SFP", ticker=self.cfg.market.benchmark, date__gte=start, date__lte=end)
        spy["date"] = pd.to_datetime(spy["date"])
        a = spy["closeadj"] / spy["close"]
        bench = pd.DataFrame({"open": (spy["open"] * a).values, "close": spy["closeadj"].values},
                             index=spy["date"]).sort_index()
        b = assemble(prices, bench, self.meta(), sectors=sectors, delist_returns=delist,
                     fundamentals=fundamentals)
        b.membership = _sp500_membership(sp, b.dates, b.tickers)
        return b


def _sp500_membership(sp: pd.DataFrame, dates, tickers) -> pd.DataFrame:
    m = pd.DataFrame(False, index=dates, columns=tickers)
    ev = sp[sp["action"].isin(["added", "removed"])].sort_values("date")
    for t, g in ev.groupby("ticker"):
        if t not in m.columns:
            continue
        state_dates = list(zip(g["date"], g["action"]))
        cur_start = None
        for d, a in state_dates:
            if a == "added":
                cur_start = d
            elif a == "removed":
                s = cur_start if cur_start is not None else dates[0]
                m.loc[(dates >= s) & (dates < d), t] = True
                cur_start = None
        if cur_start is not None:
            m.loc[dates >= cur_start, t] = True
    # constituents with no events in range (members throughout)
    cur = sp[sp["action"] == "current"]["ticker"]
    for t in set(cur) - set(ev["ticker"]):
        if t in m.columns:
            m[t] = True
    return m
