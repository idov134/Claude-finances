"""API, paper valuation and promotion-gate regression checks (no provider keys)."""
import asyncio
import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from stocksel.live.feeds import Quote
from stocksel.validation.readiness import holdout_passed, system_validated
from stocksel.web.app import create_app
from stocksel.web.service import MarketService


def _evidence(cfg):
    meta = {"source": "csv", "survivorship_free": True, "pit_fundamentals": True}
    spec = {"config_hash": cfg.hash(), "dsr": .99, "pbo": .1,
            "holdout_passed": True, "validation_protocol": "real_pit_holdout_v1",
            "research_gate": {"full_validation": True, "sanity_passed": True, "blocking_flags": []}}
    result = {"meta": meta, "spec": spec, "final_metrics": {"n_periods": 24, "excess_ci_low": .01},
              "dsr": {"dsr": .99}, "lock": {"contaminated": False}, "red_flags": []}
    return meta, spec, result


def test_promotion_requires_real_full_research_and_holdout(cfg):
    meta, spec, result = _evidence(cfg)
    assert holdout_passed(result, cfg)
    assert system_validated(spec, cfg, meta)
    for changes in ({"source": "synthetic"}, {"survivorship_free": False}, {"pit_fundamentals": False}):
        assert not system_validated(spec, cfg, {**meta, **changes})
        assert not holdout_passed({**result, "meta": {**meta, **changes}}, cfg)
    for key, value in (("full_validation", False), ("sanity_passed", False), ("blocking_flags", [{"level": "serious"}])):
        altered = copy.deepcopy(spec)
        altered["research_gate"][key] = value
        assert not system_validated(altered, cfg, meta)
        assert not holdout_passed({**result, "spec": altered}, cfg)
    assert not holdout_passed({**result, "lock": {"contaminated": True}}, cfg)
    assert not holdout_passed({**result, "final_metrics": {"n_periods": 24, "excess_ci_low": -.01}}, cfg)
    assert not system_validated({**spec, "config_hash": "different"}, cfg, meta)


class FakeService:
    started = False
    closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    def snapshot(self):
        return {"status": "ready", "recommendations": [{"ticker": "AAPL", "confidence": "Low"}]}


def test_dashboard_routes_and_read_only_api():
    service = FakeService()
    with TestClient(create_app(service=service)) as client:
        assert service.started
        assert client.get("/").status_code == 200
        assert "Research" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").json()["ready"]
        assert client.get("/api/snapshot").json()["recommendations"][0]["ticker"] == "AAPL"
        assert client.get("/api/stocks/aapl").json()["confidence"] == "Low"
        assert client.get("/api/stocks/missing").status_code == 404
        assert client.post("/api/snapshot", json={"buy": "AAPL"}).status_code == 405
        assert client.post("/api/orders", json={"buy": "AAPL"}).status_code == 404
    assert service.closed


def test_older_replay_quote_cannot_regress_paper_mark(cfg, bundle):
    path = __import__("pathlib").Path(cfg.live.ledger_dir) / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ticker = bundle.tickers[0]
    state = {"start": "2010-01-04", "last_update": "2010-03-01", "cash": 99000,
             "positions": {ticker: {"shares": 10, "entry": "2010-01-05", "entry_px": 90, "last": 100, "stop": 70}},
             "equity": []}
    path.write_text(json.dumps(state))
    service = MarketService(cfg, bundle=bundle)
    q = Quote(ticker, 50, datetime(2010, 2, 1, 21, tzinfo=timezone.utc), datetime.now(timezone.utc), "replay", 0, True)
    service.quotes[ticker] = q
    portfolio = service._read_portfolio()
    assert portfolio["equity"] == 100000
    assert portfolio["positions"][0]["valuation_source"] == "ledger last mark"
    assert ticker in service._position_tickers  # poll held names even outside universe
    q = Quote(ticker, 110, datetime(2010, 3, 2, 21, tzinfo=timezone.utc), datetime.now(timezone.utc), "replay", 0, True)
    service.quotes[ticker] = q
    assert service._read_portfolio()["equity"] == 100100
    # Raw exchange trades may use a different split/share basis.
    service.quotes[ticker] = Quote(ticker, 55, q.data_timestamp, q.received_at, "alpaca-iex", 0)
    assert service._read_portfolio()["equity"] == 100000
    path.unlink()


def test_provider_exception_text_not_exposed(cfg):
    service = MarketService(cfg)
    def broken():
        raise ValueError("https://vendor.invalid/?api_key=NEVER_EXPOSE")
    service._bootstrap = broken
    asyncio.run(service._run())
    assert service.status == "error"
    assert "NEVER_EXPOSE" not in service.snapshot()["error"]


def test_scan_scores_all_candidates_with_low_synthetic_confidence(cfg, bundle):
    from stocksel.live.recommend import recommend
    from stocksel.config import vol_target
    spec = {"members": ["logit"], "horizon": 1, "features": cfg.features.sets["core"],
            "keep": .5, "vol_target": vol_target(cfg), "dsr": 1, "pbo": 0, "holdout_passed": True}
    rows, meta = recommend(cfg, asof="2009-11-30", bundle=bundle, spec=spec, include_all=True)
    assert len(rows) == meta["n_candidates"]
    assert rows.position_pct.sum() <= 100.01
    assert (rows.position_pct <= 100 * cfg.portfolio.max_position).all()
    assert rows.confidence.str.startswith("Low").all()
    assert not meta["system_validated"]


def test_stop_alerts_monitor_holdings_without_replay_lookahead(cfg, bundle):
    b = copy.deepcopy(bundle)
    ticker = b.tickers[0]
    b.high[ticker], b.low[ticker], b.close[ticker] = 101., 99., 100.
    # A later huge high must not lift the trailing stop before it happens.
    b.high.loc["2010-04-01":, ticker] = 1000.
    now = datetime(2010, 3, 2, 21, tzinfo=timezone.utc)
    service = MarketService(cfg, bundle=b)
    service.quotes[ticker] = Quote(ticker, 95, now, now, "replay", 0, True)
    service._portfolio = {"positions": [{"ticker": ticker, "entry": "2010-01-04",
        "entry_px": 100, "stop": 96, "valuation_timestamp": now}]}
    service._check_risk()
    assert "stop:" + ticker in service.alerts
    assert "trailing:" + ticker not in service.alerts
    service.quotes[ticker] = Quote(ticker, 85, now, now, "replay", 0, True)
    service._check_risk()
    assert "trailing:" + ticker in service.alerts
    service._portfolio = {"positions": []}
    service.alerts.clear()
    service._check_risk()
    assert not service.alerts
