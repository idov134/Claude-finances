"""Async quote sources with explicit provenance and conservative freshness.

Quotes do not retrain the monthly model. Alpaca's default IEX feed covers one
exchange rather than the consolidated US market; a subscription is necessary
for SIP. Yahoo is treated as delayed (15 minutes), never real-time. Synthetic
replay preserves historical event timestamps, even when received today.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from time import monotonic
from typing import Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("quote timestamps must include a timezone")
    return value.astimezone(UTC)


@lru_cache(maxsize=1)
def _calendar():
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start="1990-01-01", end="2035-12-31")


def market_status(now: datetime | None = None) -> dict:
    """NYSE regular session; includes exchange holidays, DST and early closes."""
    now = _utc(now or utc_now())
    out = {"exchange": "XNYS", "state": "unknown", "checked_at": now.isoformat(),
           "session_open": None, "session_close": None}
    try:
        cal = _calendar()
        day = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        if not cal.is_session(day):
            return {**out, "state": "closed"}
        opening, closing = _utc(cal.session_open(day)), _utc(cal.session_close(day))
        return {**out, "state": "open" if opening <= now < closing else "closed",
                "session_open": opening.isoformat(), "session_close": closing.isoformat()}
    except (ImportError, ValueError, KeyError):
        # Missing/out-of-range calendar must never turn a stale quote green.
        return out


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float
    data_timestamp: datetime
    received_at: datetime
    provider: str
    documented_delay_seconds: int
    simulated: bool = False
    connected: bool = True

    def __post_init__(self):
        if not self.ticker or not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("a quote requires a ticker and a finite positive price")
        if self.documented_delay_seconds < 0:
            raise ValueError("provider delay cannot be negative")
        object.__setattr__(self, "data_timestamp", _utc(self.data_timestamp))
        object.__setattr__(self, "received_at", _utc(self.received_at))

    def to_dict(self, now: datetime | None = None) -> dict:
        now = _utc(now or utc_now())
        return {"ticker": self.ticker, "price": float(self.price),
                "data_timestamp": self.data_timestamp.isoformat(),
                "received_at": self.received_at.isoformat(), "provider": self.provider,
                "documented_delay_seconds": self.documented_delay_seconds,
                "simulated": self.simulated, "connected": self.connected,
                "age_seconds": max(0.0, (now - self.data_timestamp).total_seconds()),
                "receive_age_seconds": max(0.0, (now - self.received_at).total_seconds()),
                "market_status": market_status(now), "freshness": freshness(self, now)}


def freshness(quote: Quote, now: datetime | None = None) -> dict:
    now = _utc(now or utc_now())
    age = (now - quote.data_timestamp).total_seconds()
    receive_age = (now - quote.received_at).total_seconds()
    if age < -5 or receive_age < -5 or quote.data_timestamp > quote.received_at + timedelta(seconds=5):
        return {"state": "red", "label": "Invalid future timestamp", "icon": "!", "live": False}
    if quote.simulated:
        return {"state": "amber", "label": "Simulated historical replay", "icon": "◷", "live": False}
    if not quote.connected:
        return {"state": "red", "label": "Feed disconnected; cached quote", "icon": "!", "live": False}
    if age > quote.documented_delay_seconds + 120 or receive_age > 120:
        return {"state": "red", "label": "Stale quote", "icon": "!", "live": False}
    status = market_status(now)["state"]
    if status != "open":
        return {"state": "amber", "label": "Market closed" if status == "closed" else "Market hours unknown",
                "icon": "◷", "live": False}
    if quote.documented_delay_seconds:
        return {"state": "amber", "label": "Delayed quote", "icon": "◷", "live": False}
    if age > 60 or receive_age > 60:
        return {"state": "amber", "label": "Aging quote", "icon": "◷", "live": False}
    return {"state": "green", "label": "Live quote", "icon": "●", "live": True}


def global_freshness(quotes: list[Quote], now: datetime | None = None) -> dict:
    """Worst observed state; an empty feed is unavailable, never green."""
    if not quotes:
        return {"state": "red", "label": "No quotes available", "icon": "!", "live": False}
    states = [freshness(q, now) for q in quotes]
    rank = {"green": 0, "amber": 1, "red": 2}
    return max(states, key=lambda s: rank[s["state"]])


class Feed(Protocol):
    async def poll(self, tickers: list[str]) -> list[Quote]: ...
    async def close(self) -> None: ...


class SyntheticFeed:
    """Advance one historical trading bar per poll; never invent current prices."""
    def __init__(self, bundle, replay_start=None, auto_advance=True):
        self.bundle = bundle
        self.auto_advance = auto_advance
        if replay_start is None:
            self._cursor = max(0, len(bundle.dates) - 21)
        else:
            import pandas as pd
            self._cursor = min(int(bundle.dates.searchsorted(pd.Timestamp(replay_start))), len(bundle.dates) - 1)

    async def poll(self, tickers: list[str]) -> list[Quote]:
        if self.bundle.close.empty:
            return []
        date = self.bundle.dates[self._cursor]
        # Daily bars are close prices; a date-only timestamp is not the event time.
        session = market_status(datetime.combine(date.date(), datetime.min.time(), ZoneInfo("America/New_York"))
                                + timedelta(hours=12))
        stamp = (_utc(session["session_close"]) if session["session_close"] else
                 datetime.combine(date.date(), datetime.min.time(), ZoneInfo("America/New_York"))
                 .replace(hour=16).astimezone(UTC))
        now = utc_now()
        prices = self.bundle.close.iloc[self._cursor]
        out = [Quote(t, float(prices[t]), stamp, now, "synthetic-replay", 0, True)
               for t in tickers if t in prices and math.isfinite(prices[t]) and prices[t] > 0]
        if self.auto_advance:
            self._cursor = min(self._cursor + 1, len(self.bundle.dates) - 1)
        return out

    def advance(self):
        if self._cursor >= len(self.bundle.dates) - 1:
            raise ValueError("Replay ended; no unseen session remains in this demo")
        self._cursor += 1
        return self.bundle.dates[self._cursor]

    async def close(self) -> None:
        pass


class YahooPollingFeed:
    """Offload blocking downloads; missing symbols stay missing, never fabricated.

    Yahoo's feed varies by venue and entitlement. The 900-second delay is a
    conservative product assumption, not a promise of fixed 15-minute latency.
    Actual source timestamps decide freshness.
    """
    def __init__(self, delay_seconds: int = 900, batch_size: int = 100, min_poll_seconds: float = 60):
        self.delay_seconds = max(900, delay_seconds)
        self.batch_size = max(1, min(batch_size, 100))
        self.min_poll_seconds = max(1, min_poll_seconds)
        self._next_poll = 0.0
        self._retry_seconds = self.min_poll_seconds
        self._quotes: dict[str, Quote] = {}
        self.last_error: str | None = None

    def _download(self, tickers):
        import yfinance as yf
        import pandas as pd
        df = yf.download(tickers, period="1d", interval="1m", auto_adjust=False,
                         progress=False, threads=False, timeout=15, group_by="ticker")
        out = []
        if df.empty:
            return out
        for ticker in tickers:
            try:
                series = (df[ticker]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]).dropna()
                if series.empty:
                    continue
                ts = series.index[-1]
                # yfinance intraday indexes are normally tz-aware. Reject ambiguity.
                if ts.tzinfo is None:
                    log.warning("Yahoo returned a timezone-naive timestamp for %s; quote omitted", ticker)
                    continue
                out.append(Quote(ticker, float(series.iloc[-1]), _utc(ts), utc_now(), "yfinance-delayed",
                                 self.delay_seconds))
            except (KeyError, TypeError, ValueError):
                log.warning("Yahoo returned no usable quote for %s", ticker)
        return out

    async def poll(self, tickers: list[str]) -> list[Quote]:
        if monotonic() < self._next_poll:
            return [replace(self._quotes[t], connected=self.last_error is None)
                    for t in tickers if t in self._quotes]
        try:
            out = []
            for i in range(0, len(tickers), self.batch_size):
                out.extend(await asyncio.to_thread(self._download, tickers[i:i + self.batch_size]))
            if not out and tickers:
                raise RuntimeError("Yahoo returned no usable quotes")
            self._quotes.update({q.ticker: q for q in out})
            self._retry_seconds = self.min_poll_seconds
            self.last_error = None
        except Exception as exc:
            self.last_error = f"Yahoo polling failed ({type(exc).__name__}); retaining timestamped cached quotes"
            log.warning(self.last_error)
            self._retry_seconds = min(900, self._retry_seconds * 2)
        self._next_poll = monotonic() + self._retry_seconds
        return [replace(self._quotes[t], connected=self.last_error is None)
                for t in tickers if t in self._quotes]

    async def close(self) -> None:
        pass


class AlpacaFeed:
    """Alpaca v2 trades, reconnecting with capped exponential backoff.

    Credentials are loaded only from environment at connection time. IEX is the
    default, limited venue feed; selecting SIP requires paid data entitlement.
    This class never sends broker orders and does not silently downgrade feeds.
    """
    def __init__(self, feed: str = "iex"):
        if feed not in {"iex", "sip"}:
            raise ValueError("Alpaca feed must be iex or sip")
        self.feed = feed
        self.coverage = "IEX exchange only" if feed == "iex" else "US consolidated SIP (entitlement required)"
        self._symbols: set[str] = set()
        self._subscribed: set[str] = set()
        self._quotes: dict[str, Quote] = {}
        self._task: asyncio.Task | None = None
        self._ws = None
        self._send_lock = asyncio.Lock()
        self.connected = False
        self.last_error: str | None = None

    async def _subscribe(self):
        async with self._send_lock:
            if self._ws is not None:
                added = self._symbols - self._subscribed
                if added:
                    await self._ws.send(json.dumps({"action": "subscribe", "trades": sorted(added)}))
                    self._subscribed |= added

    def _ingest(self, item: dict) -> None:
        if item.get("T") != "t":
            return
        try:
            quote = Quote(str(item["S"]), float(item["p"]), _utc(item["t"]), utc_now(),
                          f"alpaca-{self.feed}", 0)
            old = self._quotes.get(quote.ticker)
            if old is None or old.data_timestamp <= quote.data_timestamp:
                self._quotes[quote.ticker] = quote
        except (KeyError, TypeError, ValueError):
            log.warning("Malformed Alpaca trade omitted")

    async def _run(self):
        import websockets
        delay = 1
        while True:
            try:
                # websockets >=15 supports the inherited HTTP(S) proxy and TLS CA.
                async with websockets.connect(f"wss://stream.data.alpaca.markets/v2/{self.feed}",
                                              open_timeout=15, ping_interval=20, ping_timeout=20,
                                              close_timeout=5) as ws:
                    self._ws = ws
                    self._subscribed = set()
                    await ws.send(json.dumps({"action": "auth", "key": os.environ["APCA_API_KEY_ID"],
                                              "secret": os.environ["APCA_API_SECRET_KEY"]}))
                    async for raw in ws:
                        items = json.loads(raw)
                        for item in items if isinstance(items, list) else [items]:
                            if item.get("T") == "error":
                                # Provider error messages may reflect input. Log a code only.
                                code = item.get("code", "unknown")
                                self.last_error = f"Alpaca stream error code {code}"
                                raise RuntimeError(self.last_error)
                            if item.get("T") == "success" and item.get("msg") == "authenticated":
                                self.connected = True
                                self.last_error = None
                                delay = 1
                                await self._subscribe()
                            self._ingest(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = self.last_error or f"Alpaca connection failed ({type(exc).__name__})"
                log.warning("%s; reconnect in %ss", self.last_error, delay)
            finally:
                self._ws = None
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(60, delay * 2)

    async def poll(self, tickers: list[str]) -> list[Quote]:
        missing = [name for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY") if not os.environ.get(name)]
        if missing:
            raise ValueError("Missing Alpaca environment variables: " + ", ".join(missing))
        self._symbols |= set(tickers)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="alpaca-market-data")
        elif self.connected:
            try:
                await self._subscribe()
            except Exception:
                self.connected = False
        return [replace(self._quotes[t], connected=self.connected) for t in tickers if t in self._quotes]

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.connected = False


def make_feed(name: str, bundle=None, *, replay_start=None) -> Feed:
    if name == "synthetic":
        if bundle is None:
            raise ValueError("SyntheticFeed needs a DataBundle")
        return SyntheticFeed(bundle, replay_start=replay_start)
    if name in {"yfinance", "yahoo"}:
        return YahooPollingFeed()
    if name == "alpaca":
        return AlpacaFeed(os.environ.get("ALPACA_DATA_FEED", "iex"))
    raise ValueError(f"Unsupported live feed: {name}")
