"""Daily ETF prices (no survivorship bias) and one consistent sector source for the stock data.

    python -m stocksel.data.build_etf_csv --data data_csv

Writes into the local CSV folder (never committed: Yahoo data may not be redistributed):
  etf_prices.csv     date, ticker, open, high, low, close, volume   (split- and dividend-adjusted)
  sectors_yahoo.csv  ticker, sector, industry                       (Yahoo's classification, every ticker)

ETFs: SPY (S&P 500), QQQ (Nasdaq-100), and the nine original Select Sector SPDRs, all listed since
Dec 1998 (QQQ: Mar 1999). An ETF's price history includes every company it ever held, so a rule
tested on it is free of the survivorship bias of the stock data.

sectors.csv holds current GICS sectors for current members only ("Unknown" for the 219 former
members). For sector baskets this would keep only today's members, which are the winners. Yahoo's
sector field is used here for every ticker that still trades. It is still a current, not a
historical, classification.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ETFS = ["SPY", "QQQ", "XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


def download_etfs(start: str = "1998-12-01", end: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ETFS, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False,
                      threads=True)
    rows = []
    for t in ETFS:
        df = raw[t].dropna(subset=["Close"])
        adj = (df["Adj Close"] / df["Close"]).astype(float)
        idx = df.index.tz_localize(None) if df.index.tz else df.index
        rows.append(pd.DataFrame({"date": idx, "ticker": t, "open": (df["Open"] * adj).to_numpy(),
                                  "high": (df["High"] * adj).to_numpy(), "low": (df["Low"] * adj).to_numpy(),
                                  "close": df["Adj Close"].to_numpy(), "volume": (df["Volume"] / adj).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def yahoo_sectors(tickers: list[str], sleep: float = 0.3, tries: int = 3) -> pd.DataFrame:
    import yfinance as yf
    out = []
    for i, t in enumerate(tickers):
        sec = ind = None
        for k in range(tries):
            try:
                info = yf.Ticker(t).info or {}
                sec, ind = info.get("sector"), info.get("industry")
                break
            except Exception:  # throttling or a missing quote: retry slowly, then record Unknown
                time.sleep(2 ** (k + 1))
        out.append({"ticker": t, "sector": sec or "Unknown", "industry": ind or "Unknown"})
        time.sleep(sleep)
        if (i + 1) % 100 == 0:
            print(f"[etf] sectors {i + 1}/{len(tickers)}", flush=True)
    return pd.DataFrame(out)


def build(data_dir: Path, sectors: bool = True) -> dict:
    etf = download_etfs()
    etf.to_csv(data_dir / "etf_prices.csv", index=False, date_format="%Y-%m-%d", float_format="%.6g")
    info = {"built": time.strftime("%Y-%m-%d %H:%M"),
            "etfs": {t: [str(g["date"].min().date()), str(g["date"].max().date()), int(len(g))]
                     for t, g in etf.groupby("ticker")}}
    if sectors:
        tick = sorted(pd.read_csv(data_dir / "sectors.csv")["ticker"].unique())
        s = yahoo_sectors(tick)
        s.to_csv(data_dir / "sectors_yahoo.csv", index=False)
        info["yahoo_sectors"] = s["sector"].value_counts().to_dict()
    (data_dir / "etf_info.json").write_text(json.dumps(info, indent=2))
    print(f"[etf] done: {json.dumps(info)}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stocksel.data.build_etf_csv")
    ap.add_argument("--data", default="data_csv")
    ap.add_argument("--no-sectors", action="store_true")
    a = ap.parse_args(argv)
    build(Path(a.data), sectors=not a.no_sectors)


if __name__ == "__main__":
    main()
