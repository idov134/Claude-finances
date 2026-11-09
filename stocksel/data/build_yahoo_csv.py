"""Build the canonical CSV dataset (see providers/csv_provider.py) from free public sources.

    python -m stocksel.data.build_yahoo_csv --out data_csv --start 1999-01-01

Sources
* historical S&P 500 membership intervals: github.com/fja05680/sp500 (`sp500_ticker_start_end.csv`);
* current GICS sectors: github.com/datasets/s-and-p-500-companies (current members only);
* daily OHLCV, split/dividend adjustment, share counts, SPY, ^VIX, ^IRX: Yahoo Finance via yfinance.

Known biases (written into the dataset's `DATASET.md` and flagged by every report):
* **survivorship**: Yahoo no longer serves most delisted tickers, so many former members that
  failed or were acquired are missing -> backtests are biased upward (an upper bound);
* market cap = split-adjusted close x share count expressed in today's split units (each report
  is multiplied by every later split ratio); Yahoo's share history starts ~2015 and is back-filled
  before that, so buybacks/issuance before 2015 are ignored (approximate size and value weights);
* sectors are current GICS for current members, "Unknown" for former members;
* delisting returns unknown -> 0 (merger-like) for names whose prices stop early;
* no point-in-time fundamentals, earnings, estimates or news -> those signals are unavailable.
The data is for local research only; Yahoo's terms do not allow redistribution, so never commit it.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MEMBERSHIP_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
SECTORS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def yahoo_symbol(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def _download(tickers, start, end, retries=4):
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker",
                             progress=False, threads=True)
            if df is not None and len(df):
                return df
        except Exception:  # network hiccups / rate limits
            pass
        time.sleep(2 ** (attempt + 1))
    return pd.DataFrame()


def split_adjusted_shares(shares_raw: pd.Series, splits: pd.Series | None) -> pd.Series:
    """Express every share-count report in today's (post-all-splits) units.

    Yahoo's ``Close`` is split-adjusted, but its share-count history is as reported at the time.
    A report dated r is multiplied by every split ratio *after r*; forward-filling the adjusted
    reports then stays consistent with the split-adjusted price even while a report lags a split.
    """
    if splits is None or len(splits) == 0:
        return shares_raw
    sp = splits[splits > 0].copy()
    sp.index = pd.to_datetime(sp.index).tz_localize(None).normalize() if getattr(sp.index, "tz", None) \
        else pd.to_datetime(sp.index).normalize()
    sp = sp.groupby(level=0).prod().sort_index()
    after = np.r_[np.cumprod(sp.to_numpy()[::-1])[::-1], 1.0]          # product of ratios from k onward
    pos = sp.index.searchsorted(shares_raw.index, side="right")        # first split strictly after r
    adj = shares_raw * after[pos]
    # Filings lag: a report dated shortly after a split can still carry the pre-split count. In
    # today's units it then sits one split-ratio below the last pre-split report; scale it up.
    for when, ratio in sp.items():
        before = adj[adj.index <= when]
        if not len(before) or ratio == 1:
            continue
        ref = before.iloc[-1]
        window = (adj.index > when) & (adj.index <= when + pd.Timedelta(days=120))
        stale = window & (np.abs(adj * ratio / ref - 1) < 0.15)
        adj[stale] = adj[stale] * ratio
    return adj


def _shares(ticker: str, start: str):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        s = tk.get_shares_full(start=start)
    except Exception:
        s = None
    if s is None or len(s) == 0:
        return None
    try:
        splits = tk.splits
    except Exception:
        splits = None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.groupby(level=0).last().astype(float)
    s = split_adjusted_shares(s, splits)
    # drop spurious prints (after split adjustment, so genuine splits are not mistaken for outliers)
    med = s.rolling(9, center=True, min_periods=1).median()
    s = s[(s / med).between(1 / 3, 3)]
    return s.astype(float)


def build(out: Path, start: str, end: str | None, batch: int = 40, sleep: float = 3.0,
          limit: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    mem = pd.read_csv(io.StringIO(requests.get(MEMBERSHIP_URL, timeout=60).text))
    mem["ticker"] = mem["ticker"].map(yahoo_symbol)
    mem = mem.rename(columns={"start_date": "start", "end_date": "end"})
    mem = mem[(mem["end"].isna()) | (pd.to_datetime(mem["end"]) >= pd.Timestamp(start))]
    mem[["ticker", "start", "end"]].to_csv(out / "membership.csv", index=False)
    tickers = sorted(mem["ticker"].unique())
    if limit:  # smoke tests only
        tickers = tickers[:limit]
        mem = mem[mem["ticker"].isin(tickers)]
    print(f"[build] {len(tickers)} tickers ever in the S&P 500 since {start}", flush=True)

    rows, have = [], []

    def take(raw, chunk):
        for t in chunk:
            if t in have or raw.empty or t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].dropna(subset=["Close"])
            if len(df) < 60:
                continue
            adj = (df["Adj Close"] / df["Close"]).astype(float)
            rows.append(pd.DataFrame({"date": df.index.tz_localize(None) if df.index.tz else df.index,
                                      "ticker": t, "open": df["Open"] * adj, "high": df["High"] * adj,
                                      "low": df["Low"] * adj, "close": df["Adj Close"],
                                      "volume": df["Volume"] / adj, "close_raw": df["Close"]}))
            have.append(t)

    for k in range(0, len(tickers), batch):
        chunk = tickers[k:k + batch]
        take(_download(chunk, start, end), chunk)
        print(f"[build] prices {min(k + batch, len(tickers))}/{len(tickers)} (with data: {len(have)})", flush=True)
        time.sleep(sleep)
    # Yahoo throttles bursts and then answers "possibly delisted": retry the misses slowly, twice.
    for rnd in range(2):
        missing = [t for t in tickers if t not in have]
        for k in range(0, len(missing), 15):
            chunk = missing[k:k + 15]
            take(_download(chunk, start, end), chunk)
            time.sleep(4)
        print(f"[build] retry round {rnd + 1}: with data {len(have)} / {len(tickers)}", flush=True)
    prices = pd.concat(rows, ignore_index=True)

    caps, n_shares = [], 0
    for i, t in enumerate(have):
        sh = _shares(t, "2005-01-01")
        p = prices.loc[prices["ticker"] == t, ["date", "close_raw"]].set_index("date")["close_raw"]
        if sh is not None and len(sh):
            n_shares += 1
            s = sh.reindex(p.index.union(sh.index)).ffill().bfill().reindex(p.index)
            caps.append(pd.Series((p * s).to_numpy(), index=prices.index[prices["ticker"] == t]))
        if (i + 1) % 100 == 0:
            print(f"[build] shares {i + 1}/{len(have)}", flush=True)
    prices["market_cap"] = pd.concat(caps).reindex(prices.index) if caps else np.nan
    prices.drop(columns=["close_raw"]).to_csv(out / "prices.csv", index=False, float_format="%.6g")

    bench = _download(["SPY", "^VIX", "^IRX"], start, end)
    spy = bench["SPY"].dropna(subset=["Close"])
    a = spy["Adj Close"] / spy["Close"]
    pd.DataFrame({"open": spy["Open"] * a, "close": spy["Adj Close"]}, index=spy.index.rename("date")).to_csv(
        out / "benchmark.csv")
    bench["^VIX"]["Close"].dropna().rename("vix").rename_axis("date").to_csv(out / "vix.csv")
    (bench["^IRX"]["Close"].dropna() / 100).rename("rf").rename_axis("date").to_csv(out / "rf.csv")

    sec = pd.read_csv(io.StringIO(requests.get(SECTORS_URL, timeout=60).text))
    sec = pd.DataFrame({"ticker": sec["Symbol"].map(yahoo_symbol), "sector": sec["GICS Sector"]})
    sec = pd.concat([sec, pd.DataFrame({"ticker": sorted(set(have) - set(sec["ticker"])), "sector": "Unknown"})])
    sec[sec["ticker"].isin(have)].to_csv(out / "sectors.csv", index=False)

    last = prices.groupby("ticker")["date"].max()
    final_day = prices["date"].max()
    early = last[last < final_day - pd.Timedelta(days=10)].index
    pd.DataFrame({"ticker": early, "delist_return": 0.0}).to_csv(out / "delistings.csv", index=False)

    # coverage: share of index members (per membership file) with a price on each month-end
    me = pd.Series(sorted(prices["date"].unique()))
    me = me.groupby([me.dt.year, me.dt.month]).max().to_numpy()
    have_set = prices.groupby("date")["ticker"].apply(set)
    cov = []
    for d in me:
        d = pd.Timestamp(d)
        members = mem[(pd.to_datetime(mem["start"]) <= d) & (mem["end"].isna() | (pd.to_datetime(mem["end"]) >= d))]
        n = members["ticker"].nunique()
        got = len(set(members["ticker"]) & have_set.get(d, set()))
        cov.append({"date": str(d.date()), "members": int(n), "with_prices": int(got)})
    cov = pd.DataFrame(cov)
    info = {"built": time.strftime("%Y-%m-%d %H:%M"), "start": start, "end": end,
            "tickers_ever_member": len(tickers), "tickers_with_prices": len(have),
            "tickers_with_share_history": n_shares, "rows": int(len(prices)),
            "coverage_mean": float((cov["with_prices"] / cov["members"]).mean()),
            "coverage_first_year": float((cov["with_prices"] / cov["members"]).iloc[:12].mean()),
            "coverage_last_year": float((cov["with_prices"] / cov["members"]).iloc[-12:].mean()),
            "early_ending_tickers": int(len(early))}
    cov.to_csv(out / "coverage.csv", index=False)
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    (out / "DATASET.md").write_text(__doc__)
    print(f"[build] done: {json.dumps(info)}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stocksel.data.build_yahoo_csv")
    ap.add_argument("--out", default="data_csv")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N tickers only (smoke test)")
    args = ap.parse_args(argv)
    build(Path(args.out), args.start, args.end, limit=args.limit)


if __name__ == "__main__":
    main()
