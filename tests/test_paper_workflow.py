"""Research-to-paper API acceptance without a model or vendor dependency."""
import asyncio
from dataclasses import replace
from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stocksel.data.bundle import DataBundle
from stocksel.live.feeds import SyntheticFeed, utc_now
from stocksel.live.paper import PaperLedger
from stocksel.web.app import create_app
from stocksel.web.service import MarketService


@pytest.fixture
def service(cfg, tmp_path):
    cfg = cfg.copy_with({"live.ledger_dir": str(tmp_path / "ledger")})
    dates = pd.bdate_range("2023-10-01", periods=50)
    p = pd.DataFrame(100., index=dates, columns=["AAA", "BBB"])
    b = DataBundle(open=p.copy(), high=p+1, low=p-1, close=p.copy(), volume=p*10000,
                   market_cap=p*1e8, sectors=pd.Series({"AAA": "Tech", "BBB": "Health"}),
                   benchmark=pd.DataFrame({"open": 100., "close": 100.}, index=dates),
                   meta={"source": "synthetic"})
    feed = SyntheticFeed(b, replay_start=dates[30], auto_advance=False)
    s = MarketService(cfg, bundle=b, feed=feed)
    s.rows = [{"ticker": "AAA", "position_pct": 2., "stop": 80., "signal_asof": str(dates[29].date()),
               "rank": 1, "confidence": "Low", "calibrated_prob": .54}]
    s.meta = {"asof": str(dates[29].date())}
    s.status = "ready"
    s.quotes = {q.ticker: q for q in asyncio.run(feed.poll(["AAA"]))}
    async def start():
        pass
    s.start = start
    return s


HEADERS = {"X-Stocksel-Request": "paper"}


def test_preview_submit_retry_fill_restart_and_export(service):
    with TestClient(create_app(service=service)) as c:
        detail = c.get("/api/stocks/AAA").json()
        assert detail["history"][-1]["date"] == service.quotes["AAA"].data_timestamp.date().isoformat()
        assert "probability_event" in detail["evidence"]
        p = c.post("/api/paper/preview", headers=HEADERS,
                   json={"ticker": "AAA", "position_pct": 2, "thesis": "Test evidence", "counter_thesis": "Could fail"})
        assert p.status_code == 200, p.text
        preview = p.json()
        assert preview["estimated_cost"] > 0
        payload = {"preview_id": preview["preview_id"], "idempotency_key": "first-decision"}
        first = c.post("/api/paper/submit", headers=HEADERS, json=payload)
        assert first.status_code == 200, first.text
        assert not first.json()["duplicate"]
        assert c.post("/api/paper/submit", headers=HEADERS, json=payload).json()["duplicate"]
        ledger = PaperLedger(service.cfg)
        assert not ledger.state["positions"]  # no same-close fill
        assert len(ledger.state["pending"]) == 1
        advance = c.post("/api/paper/advance", headers=HEADERS, json={})
        assert advance.status_code == 200, advance.text
        state = c.get("/api/paper/export").json()
        assert len(state["trades"]) == 1
        assert state["trades"][0]["date"] > preview["asof"]
        assert state["cash"] >= 0 and state["fees_paid"] > 0
        assert state["journal"][0]["thesis"] == "Test evidence"
        assert c.post("/api/paper/submit", headers=HEADERS, json=payload).json()["duplicate"]
        assert len(PaperLedger(service.cfg).state["trades"]) == 1
        # Lost receipt still retries safely via the durable ledger submission.
        for file in (ledger.dir / "previews").glob("receipt-*"):
            file.unlink()
        assert c.post("/api/paper/submit", headers=HEADERS, json=payload).json()["duplicate"]


def test_paper_guards_and_unpreviewed_mutation(service):
    with TestClient(create_app(service=service)) as c:
        body = {"ticker": "AAA", "position_pct": 2}
        assert c.post("/api/paper/preview", json=body).status_code == 403
        assert c.post("/api/paper/preview", headers={**HEADERS, "Origin": "https://evil.example"}, json=body).status_code == 403
        assert c.get("/api/snapshot", headers={"Host": "evil.example"}).status_code == 400
        assert c.post("/api/paper/preview", headers=HEADERS, json={**body, "position_pct": 10}).status_code == 409
        assert c.post("/api/paper/submit", headers=HEADERS,
                      json={"preview_id": "a"*32, "idempotency_key": "unknown-test"}).status_code == 409
        service.quotes["AAA"] = replace(service.quotes["AAA"], received_at=utc_now()-timedelta(minutes=10))
        assert c.post("/api/paper/preview", headers=HEADERS, json=body).status_code == 409


def test_manual_replay_quotes_do_not_advance_or_refit(service):
    signal = service.rows[0]["signal_asof"]
    before = service.feed._cursor
    asyncio.run(service.feed.poll(["AAA"]))
    asyncio.run(service.feed.poll(["AAA"]))
    assert service.feed._cursor == before
    assert service.rows[0]["signal_asof"] == signal


def test_changed_portfolio_invalidates_preview(service):
    with TestClient(create_app(service=service)) as c:
        p = c.post("/api/paper/preview", headers=HEADERS, json={"ticker": "AAA", "position_pct": 2}).json()
        PaperLedger(service.cfg).save()
        result = c.post("/api/paper/submit", headers=HEADERS,
                        json={"preview_id": p["preview_id"], "idempotency_key": "revision-test"})
        assert result.status_code == 409
        assert not PaperLedger(service.cfg).state["pending"]


def test_quote_order_and_future_timestamp_cannot_poison_snapshot(service):
    quote = service.quotes["AAA"]
    service._accept_quotes([replace(quote, price=1, data_timestamp=quote.data_timestamp-timedelta(days=1))])
    assert service.quotes["AAA"].price == 100
    service._accept_quotes([replace(quote, price=2, data_timestamp=utc_now()+timedelta(days=1))])
    assert service.quotes["AAA"].price == 100
    service._accept_quotes([replace(quote, price=101, data_timestamp=quote.data_timestamp+timedelta(days=1))])
    assert service.quotes["AAA"].price == 101


def test_unsupported_currency_has_visible_block(service):
    service.cfg = service.cfg.copy_with({"capital.currency": "ILS"})
    with pytest.raises(ValueError, match="USD only"):
        service._bootstrap()
