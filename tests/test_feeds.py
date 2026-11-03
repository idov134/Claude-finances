"""Offline feed contracts: event time, stale cache, real-session awareness."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from stocksel.live.feeds import (AlpacaFeed, Quote, SyntheticFeed, YahooPollingFeed,
                                freshness, global_freshness, make_feed, market_status)

UTC = timezone.utc
OPEN = datetime(2026, 11, 3, 16, 0, tzinfo=UTC)


def quote(**overrides):
    args = dict(ticker="AAPL", price=100.0, data_timestamp=OPEN - timedelta(seconds=5),
                received_at=OPEN, provider="test", documented_delay_seconds=0)
    return Quote(**(args | overrides))


def test_quote_provenance_and_freshness():
    q = quote()
    result = q.to_dict(OPEN)
    assert result["age_seconds"] == 5
    assert result["receive_age_seconds"] == 0
    assert result["market_status"]["state"] == "open"
    assert result["freshness"]["live"]
    assert result["data_timestamp"] != result["received_at"]


@pytest.mark.parametrize("changes,label", [
    ({"data_timestamp": OPEN - timedelta(minutes=5)}, "Stale quote"),
    ({"received_at": OPEN - timedelta(minutes=5), "data_timestamp": OPEN - timedelta(minutes=5)}, "Stale quote"),
    ({"data_timestamp": OPEN + timedelta(seconds=20)}, "Invalid future timestamp"),
    ({"received_at": OPEN + timedelta(seconds=20)}, "Invalid future timestamp"),
])
def test_reject_stale_or_future(changes, label):
    result = freshness(quote(**changes), OPEN)
    assert result["state"] == "red"
    assert result["label"] == label
    assert not result["live"]


def test_documented_delay_is_never_live():
    q = quote(data_timestamp=OPEN - timedelta(minutes=15), documented_delay_seconds=900)
    assert freshness(q, OPEN)["label"] == "Delayed quote"
    assert not freshness(q, OPEN)["live"]
    assert freshness(q, OPEN + timedelta(minutes=3))["state"] == "red"


def test_disconnected_recent_quote_is_not_live():
    result = freshness(quote(connected=False), OPEN)
    assert result["state"] == "red"
    assert result["label"] == "Feed disconnected; cached quote"
    assert not result["live"]


def test_synthetic_never_live_even_if_received_now():
    result = freshness(quote(data_timestamp=OPEN - timedelta(days=365), simulated=True), OPEN)
    assert result["state"] == "amber"
    assert result["label"] == "Simulated historical replay"
    assert not result["live"]


def test_market_holidays_weekends_and_early_closes():
    assert market_status(datetime(2026, 11, 26, 16, tzinfo=UTC))["state"] == "closed"  # Thanksgiving
    assert market_status(datetime(2026, 11, 7, 16, tzinfo=UTC))["state"] == "closed"  # Saturday
    early = market_status(datetime(2026, 11, 27, 17, tzinfo=UTC))
    assert early["state"] == "open"
    assert early["session_close"] == "2026-11-27T18:00:00+00:00"
    assert market_status(datetime(2026, 11, 27, 18, tzinfo=UTC))["state"] == "closed"


def test_market_dst_and_closed_quote():
    assert market_status(datetime(2026, 7, 7, 13, 30, tzinfo=UTC))["state"] == "open"
    closed = datetime(2026, 11, 3, 22, tzinfo=UTC)
    q = quote(data_timestamp=closed, received_at=closed)
    assert freshness(q, closed)["label"] == "Market closed"
    assert not freshness(q, closed)["live"]


def test_worst_quote_sets_global_status_and_empty_feed_is_red():
    assert global_freshness([], OPEN)["state"] == "red"
    assert global_freshness([quote(), quote(data_timestamp=OPEN - timedelta(hours=2))], OPEN)["state"] == "red"


@pytest.mark.parametrize("changes", [{"price": 0}, {"price": float("nan")},
                                       {"data_timestamp": datetime(2026, 11, 3)},
                                       {"documented_delay_seconds": -1}])
def test_malformed_quotes_rejected(changes):
    with pytest.raises(ValueError):
        quote(**changes)


def test_synthetic_replay_advances_and_preserves_event_time():
    class Bundle:
        dates = pd.DatetimeIndex(["2025-11-26", "2025-11-28"])
        close = pd.DataFrame({"AAPL": [100.0, 110.0], "BAD": [float("nan"), float("nan")]}, index=dates)
    feed = SyntheticFeed(Bundle(), replay_start="2025-11-26")

    async def run():
        first = await feed.poll(["AAPL", "BAD", "MISSING"])
        second = await feed.poll(["AAPL"])
        last = await feed.poll(["AAPL"])
        await feed.close()
        return first, second, last

    first, second, last = asyncio.run(run())
    assert [q.ticker for q in first] == ["AAPL"]
    assert first[0].price == 100
    assert second[0].price == last[0].price == 110
    assert first[0].data_timestamp == datetime(2025, 11, 26, 21, tzinfo=UTC)
    assert second[0].data_timestamp == datetime(2025, 11, 28, 18, tzinfo=UTC)  # Early close
    assert first[0].data_timestamp < first[0].received_at


def test_yahoo_cache_does_not_refresh_timestamp_and_backs_off(monkeypatch):
    feed = YahooPollingFeed()
    q = quote()
    calls = []
    clock = [100.0]
    monkeypatch.setattr("stocksel.live.feeds.monotonic", lambda: clock[0])

    def download(tickers):
        calls.append(tickers)
        if len(calls) > 1:
            raise RuntimeError("rate limit")
        return [q]

    monkeypatch.setattr(feed, "_download", download)

    async def run():
        assert await feed.poll(["AAPL"]) == [q]
        clock[0] = 110
        assert await feed.poll(["AAPL"]) == [q]
        assert len(calls) == 1
        clock[0] = 161
        cached = (await feed.poll(["AAPL"]))[0]
        assert cached.received_at == q.received_at
        assert cached.data_timestamp == q.data_timestamp
        assert not cached.connected
        assert feed.last_error
        assert feed._next_poll == 281
        await feed.close()

    asyncio.run(run())
    assert len(calls) == 2
    assert q.received_at == OPEN


def test_alpaca_ingest_ignores_out_of_order_or_bad_messages():
    feed = AlpacaFeed()
    feed._ingest({"T": "t", "S": "AAPL", "p": 200, "t": OPEN.isoformat()})
    feed._ingest({"T": "t", "S": "AAPL", "p": 100, "t": (OPEN - timedelta(seconds=2)).isoformat()})
    feed._ingest({"T": "t", "S": "AAPL", "p": -1, "t": OPEN.isoformat()})
    feed._ingest({"T": "success", "msg": "connected"})
    assert feed._quotes["AAPL"].price == 200
    assert feed._quotes["AAPL"].provider == "alpaca-iex"
    assert feed.coverage == "IEX exchange only"


def test_alpaca_missing_credentials_reports_names_only(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="APCA_API_KEY_ID, APCA_API_SECRET_KEY"):
        asyncio.run(AlpacaFeed().poll(["AAPL"]))


def test_alpaca_auth_subscribe_trade_and_clean_shutdown(monkeypatch):
    """Exercise the real stream loop with an in-memory websocket transport."""
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    feed = AlpacaFeed()
    sent = []

    async def run():
        queue = asyncio.Queue()
        ingested = asyncio.Event()
        original = feed._ingest

        def ingest(item):
            original(item)
            if item.get("T") == "t":
                ingested.set()

        monkeypatch.setattr(feed, "_ingest", ingest)

        class Socket:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def send(self, message):
                sent.append(json.loads(message))

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await queue.get()

        def connect(url, **kwargs):
            assert url == "wss://stream.data.alpaca.markets/v2/iex"
            return Socket()

        monkeypatch.setattr("websockets.connect", connect)
        assert await feed.poll(["AAPL"]) == []
        await queue.put(json.dumps([{"T": "success", "msg": "authenticated"},
                                    {"T": "t", "S": "AAPL", "p": 123.4, "t": OPEN.isoformat()}]))
        await asyncio.wait_for(ingested.wait(), timeout=2)
        assert feed.connected
        assert (await feed.poll(["AAPL", "MSFT"]))[0].price == 123.4
        assert sent[0] == {"action": "auth", "key": "test-key", "secret": "test-secret"}
        assert sent[1] == {"action": "subscribe", "trades": ["AAPL"]}
        assert sent[2] == {"action": "subscribe", "trades": ["MSFT"]}
        await feed.close()
        assert not feed.connected
        assert feed._task is None

    asyncio.run(run())


def test_factory_has_explicit_providers():
    assert isinstance(make_feed("alpaca"), AlpacaFeed)
    assert isinstance(make_feed("yfinance"), YahooPollingFeed)
    with pytest.raises(ValueError):
        make_feed("synthetic")
    with pytest.raises(ValueError):
        make_feed("made-up")
