"""Temporal and connection invariants at the scanner/feed boundary."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from stocksel.live.feeds import Quote, make_feed
from stocksel.web.service import MarketService

UTC = timezone.utc
NOW = datetime(2026, 11, 3, 16, tzinfo=UTC)


def _scanner(cfg, asof, bundle, spec, include_all):
    ticker = bundle.close.iloc[-1].dropna().index[0]
    return pd.DataFrame([{"ticker": ticker, "position_pct": 1.0, "stop": None}]), {
        "asof": asof, "n_candidates": 1,
    }


def test_synthetic_replay_strictly_follows_frozen_signal(cfg, bundle, monkeypatch):
    service = MarketService(cfg, bundle=bundle, scanner=_scanner)
    monkeypatch.setattr(service, "_read_portfolio", lambda: {"positions": [], "history": []})

    async def run():
        await service.start()
        deadline = asyncio.get_running_loop().time() + 5
        while service._last_poll is None and service.status != "error":
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("service did not poll its replay feed")
            await asyncio.sleep(0.01)
        assert service.status == "ready", service.error
        assert service.quotes
        signal_date = pd.Timestamp(service.meta["asof"]).date()
        for q in service.quotes.values():
            assert q.data_timestamp.date() > signal_date
            assert q.simulated
        # The holdout's final two years were physically removed before fitting.
        assert service.bundle.dates[-1] < bundle.dates[-1] - pd.DateOffset(months=23)
        await service.close()

    asyncio.run(run())


def _snapshot_service(cfg, monkeypatch, *, connected=True, last_error=None, error=None, missing=False):
    service = MarketService(cfg, feed_name="alpaca", feed=SimpleNamespace(connected=connected,
                             last_error=last_error, coverage="IEX exchange only"))
    service.status = "ready"
    service.feed_error = error
    service.rows = [{"ticker": "AAPL", "position_pct": 1, "signal_asof": "2026-10-30"}]
    if missing:
        service.rows.append({"ticker": "MSFT", "position_pct": 1, "signal_asof": "2026-10-30"})
    service.quotes = {"AAPL": Quote("AAPL", 123, NOW - timedelta(seconds=2), NOW, "alpaca-iex", 0)}
    monkeypatch.setattr("stocksel.web.service.utc_now", lambda: NOW)
    return service


def test_disconnected_service_never_labels_cached_quote_live(cfg, monkeypatch):
    service = _snapshot_service(cfg, monkeypatch, connected=False)
    result = service.snapshot()
    assert result["freshness"]["state"] == "red"
    assert not result["recommendations"][0]["quote"]["freshness"]["live"]
    assert result["feed"]["status"] == "waiting"


def test_provider_error_changes_feed_and_per_quote_status(cfg, monkeypatch):
    service = _snapshot_service(cfg, monkeypatch, last_error="Alpaca stream error code 406")
    result = service.snapshot()
    assert result["feed"]["status"] == "error"
    assert result["feed"]["error"] == "Alpaca stream error code 406"
    assert result["quotes"][0]["freshness"]["state"] == "red"


def test_missing_symbol_prevents_global_green(cfg, monkeypatch):
    service = _snapshot_service(cfg, monkeypatch, missing=True)
    result = service.snapshot()
    assert result["freshness"]["state"] == "red"
    assert result["feed"]["missing_symbols"] == ["MSFT"]
    assert result["feed"]["received_symbols"] == 1


def test_synthetic_factory_accepts_replay_start(bundle):
    decision = bundle.month_end_dates()[-2]
    feed = make_feed("synthetic", bundle, replay_start=decision + pd.Timedelta(days=1))
    assert bundle.dates[feed._cursor] > decision
