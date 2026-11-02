"""Canonical in-memory data container.

All price-like fields are *wide* DataFrames (index = trading dates, columns =
tickers), split/dividend adjusted. Event-like data (fundamentals, earnings,
estimates, news) are *long* DataFrames with an explicit availability timestamp
(``publish_date`` / ``announce_ts`` / ``timestamp``) so point-in-time joins are
possible.

``DataBundle.truncate(end)`` is the physical firewall used for the holdout and
for leakage tests: it removes every row that became known after ``end``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close", "volume", "market_cap")


@dataclass
class DataBundle:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    market_cap: pd.DataFrame
    sectors: pd.Series                       # ticker -> sector (usually NOT point-in-time; flagged)
    benchmark: pd.DataFrame                  # columns: open, close (total-return index)
    membership: Optional[pd.DataFrame] = None   # bool wide frame: index membership on each date
    delist_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fundamentals: Optional[pd.DataFrame] = None  # ticker, period_end, publish_date, fields...
    earnings: Optional[pd.DataFrame] = None      # ticker, announce_ts, eps_actual, eps_consensus
    estimates: Optional[pd.DataFrame] = None     # ticker, date, fy1_consensus
    news: Optional[pd.DataFrame] = None          # ticker, timestamp (UTC), headline, body
    vix: Optional[pd.Series] = None
    risk_free: Optional[pd.Series] = None        # annualized rate, daily index
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    def validate(self) -> None:
        idx, cols = self.close.index, self.close.columns
        for name in PRICE_FIELDS:
            df = getattr(self, name)
            if not df.index.equals(idx) or not df.columns.equals(cols):
                raise ValueError(f"{name} is not aligned with close")
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError("dates must be sorted and unique")
        if not self.benchmark.index.equals(idx):
            raise ValueError("benchmark must share the trading calendar")
        if self.membership is not None and not self.membership.index.equals(idx):
            raise ValueError("membership must share the trading calendar")

    def listed(self) -> pd.DataFrame:
        """True where the stock has a valid close on that date."""
        return self.close.notna()

    def in_index(self) -> pd.DataFrame:
        if self.membership is None:
            return self.listed()
        return self.membership.reindex_like(self.close).fillna(False).astype(bool) & self.listed()

    def rf_daily(self) -> pd.Series:
        if self.risk_free is None:
            return pd.Series(0.0, index=self.dates)
        return (self.risk_free.reindex(self.dates).ffill().fillna(0.0)) / 252.0

    # --------------------------------------------------------------- firewall
    def truncate(self, end: pd.Timestamp | str) -> "DataBundle":
        """Return a copy containing only information known at or before ``end``.

        Prices: rows with date <= end. Events: rows whose availability time is
        <= end (publish_date, announce_ts, estimate date, news timestamp).
        Tickers with no data at all before ``end`` are dropped.
        """
        end = pd.Timestamp(end)
        keep = self.dates <= end
        close = self.close.loc[keep]
        alive = close.columns[close.notna().any()]

        def _w(df: pd.DataFrame | None) -> pd.DataFrame | None:
            return None if df is None else df.loc[keep, alive]

        def _ev(df: pd.DataFrame | None, col: str) -> pd.DataFrame | None:
            if df is None:
                return None
            ts = pd.to_datetime(df[col])
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
            return df.loc[(ts <= end + pd.Timedelta(hours=23, minutes=59)) & df["ticker"].isin(alive)].copy()

        # Delisting returns are only "known" if the delisting happened before `end`.
        lv = last_valid_positions(self.close)
        last_valid = pd.Series([self.dates[i] if i >= 0 else pd.NaT for i in lv], index=self.close.columns)
        dr = self.delist_returns.reindex(alive)
        dr = dr[last_valid.reindex(dr.index) < end] if len(dr) else dr

        news = None
        if self.news is not None:
            ts = pd.to_datetime(self.news["timestamp"], utc=True)
            cutoff = end.tz_localize(self.meta.get("timezone", "America/New_York")) + pd.Timedelta(hours=23, minutes=59)
            news = self.news.loc[(ts <= cutoff) & self.news["ticker"].isin(alive)].copy()

        return replace(
            self,
            open=_w(self.open), high=_w(self.high), low=_w(self.low), close=close.loc[:, alive],
            volume=_w(self.volume), market_cap=_w(self.market_cap),
            membership=_w(self.membership),
            benchmark=self.benchmark.loc[keep],
            sectors=self.sectors.reindex(alive),
            delist_returns=dr.dropna(),
            fundamentals=_ev(self.fundamentals, "publish_date"),
            earnings=_ev(self.earnings, "announce_ts"),
            estimates=_ev(self.estimates, "date"),
            news=news,
            vix=None if self.vix is None else self.vix.loc[self.vix.index <= end],
            risk_free=None if self.risk_free is None else self.risk_free.loc[self.risk_free.index <= end],
            meta={**self.meta, "truncated_at": str(end.date())},
        )

    # -------------------------------------------------------------- helpers
    def daily_returns(self) -> pd.DataFrame:
        """Close-to-close returns; the delisting return is booked on the last listed day + 1."""
        return self.close.pct_change(fill_method=None)

    def month_end_dates(self) -> pd.DatetimeIndex:
        """Last trading day of each calendar month (decision dates)."""
        s = pd.Series(self.dates, index=self.dates)
        return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).max().values)

    def next_trading_day(self, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        pos = self.dates.searchsorted(d, side="right")
        return self.dates[pos] if pos < len(self.dates) else None

    def adv_usd(self, window: int = 63) -> pd.DataFrame:
        dv = self.close * self.volume
        return dv.rolling(window, min_periods=max(5, window // 3)).median()

    def survivorship_warning(self) -> Optional[str]:
        if not self.meta.get("survivorship_free", False):
            return ("Data source is NOT survivorship-bias free (delisted stocks missing). "
                    "Backtest returns are biased upward.")
        return None


def first_valid_positions(df: pd.DataFrame) -> np.ndarray:
    arr = df.notna().to_numpy()
    has = arr.any(axis=0)
    first = arr.argmax(axis=0)
    return np.where(has, first, -1)


def last_valid_positions(df: pd.DataFrame) -> np.ndarray:
    arr = df.notna().to_numpy()
    has = arr.any(axis=0)
    last = arr.shape[0] - 1 - arr[::-1].argmax(axis=0)
    return np.where(has, last, -1)
