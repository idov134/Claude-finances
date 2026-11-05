"""Hand-checkable ledger arithmetic, chronology, recovery and idempotency."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from stocksel.config import load_config
from stocksel.data.bundle import DataBundle
from stocksel.live.paper import PaperLedger


@pytest.fixture
def accounting_cfg(tmp_path):
    return load_config(overrides={"live.ledger_dir": str(tmp_path / "paper"),
        "capital.initial": 1000., "costs.commission_bps": 100.,
        "costs.min_half_spread_bps": 0., "costs.spread_k": 0.,
        "costs.slippage_bps": 0., "costs.impact_k": 0.})


def market(n=12):
    dates = pd.bdate_range("2026-01-05", periods=n)
    prices = pd.DataFrame(10., index=dates, columns=["A", "B"])
    return DataBundle(open=prices.copy(), high=prices + 1, low=prices - 1, close=prices.copy(),
        volume=prices * 1_000_000, market_cap=prices * 1e9,
        sectors=pd.Series({"A": "Tech", "B": "Finance"}),
        benchmark=pd.DataFrame({"open": 100., "close": 100.}, index=dates),
        meta={"source": "synthetic", "price_basis": "canonical_adjusted"})


def recs(a=0, b=0, stop=None):
    return pd.DataFrame([{"ticker": "A", "position_pct": a, "stop": stop},
                         {"ticker": "B", "position_pct": b, "stop": None}])


def test_hand_checked_buy_sale_cost_basis_pnl(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(50), str(b.dates[0].date()), idempotency_key="buy")
    led.update(b.truncate(b.dates[1]))
    # 50 adjusted units at $10, $5 fee, cash $495, equity $995.
    assert led.state["cash"] == pytest.approx(495)
    assert led.state["positions"]["A"]["cost_basis"] == pytest.approx(505)
    assert led.status()["unrealized_pnl"] == pytest.approx(-5)
    b.open.loc[b.dates[2]:, "A"] = 12.
    b.close.loc[b.dates[2]:, "A"] = 12.
    led.submit(recs(), str(b.dates[1].date()), idempotency_key="sell")
    led.update(b.truncate(b.dates[2]))
    # Gross proceeds 600, sale fee 6, realized 600 - 6 - 505 = 89.
    assert led.state["cash"] == pytest.approx(1089)
    assert led.status()["realized_pnl"] == pytest.approx(89)
    assert led.status()["fees_paid"] == pytest.approx(11)
    assert led.status()["equity"] == pytest.approx(1089)
    assert not led.state["positions"]
    assert led.status()["cumulative_return"] == pytest.approx(.089)


def test_full_capital_purchase_reserves_fees(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(100), str(b.dates[0].date()))
    led.update(b.truncate(b.dates[1]))
    assert 0 <= led.state["cash"] < 1e-6
    assert led.state["positions"]["A"]["shares"] == pytest.approx(1000 / 1.01 / 10)
    assert led.state["positions"]["A"]["cost_basis"] == pytest.approx(1000)


def test_missing_sale_remains_pending_and_funds_later_buy(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(100), str(b.dates[0].date()))
    led.update(b.truncate(b.dates[1]))
    led.submit(recs(0, 90), str(b.dates[1].date()))
    b.open.loc[b.dates[2], "A"] = np.nan
    led.update(b.truncate(b.dates[2]))
    assert led.state["pending"]
    assert "A" in led.state["positions"] and "B" not in led.state["positions"]
    led.update(b.truncate(b.dates[3]))
    assert not led.state["pending"]
    assert "A" not in led.state["positions"] and "B" in led.state["positions"]
    trades = [t for t in led.state["trades"] if t["date"] == str(b.dates[3].date())]
    assert [t["ticker"] for t in trades] == ["A", "B"]
    assert led.state["cash"] >= 0


def test_actual_reductions_before_lower_weight_buys(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(100), str(b.dates[0].date()))
    led.update(b.truncate(b.dates[1]))
    led.submit(recs(60, 30), str(b.dates[1].date()))
    led.update(b.truncate(b.dates[2]))
    assert [t["ticker"] for t in led.state["trades"][-2:]] == ["A", "B"]
    assert led.state["positions"]["B"]["shares"] > 0


def test_idempotency_restart_conflict_and_concurrent_retries(accounting_cfg):
    b = market()
    date = str(b.dates[0].date())
    def submit(_):
        return PaperLedger(accounting_cfg).submit(recs(50), date, "same-key", {"thesis": "test"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(submit, range(4)))
    assert len({r["order_id"] for r in receipts}) == 1
    assert sum(not r["duplicate"] for r in receipts) == 1
    led = PaperLedger(accounting_cfg)
    assert len(led.state["pending"]) == len(led.state["journal"]) == 1
    assert led.submission("same-key")["journal"]["thesis"] == "test"
    with pytest.raises(ValueError, match="different decision"):
        led.submit(recs(60), date, "same-key", {"thesis": "test"})
    led.update(b.truncate(b.dates[1]))
    before = len(led.state["trades"])
    led.update(b.truncate(b.dates[1]))
    assert len(led.state["trades"]) == before
    assert PaperLedger(accounting_cfg).submit(recs(50), date, "same-key", {"thesis": "test"})["duplicate"]


def test_execution_cost_does_not_use_future_close_volume(accounting_cfg, tmp_path):
    cfg = accounting_cfg.copy_with({"costs.impact_k": .5, "costs.spread_k": 25.,
                                    "costs.min_half_spread_bps": 2.})
    b = market(80)
    decision, execution = b.dates[65], b.dates[66]
    altered = replace(b, close=b.close.copy(), volume=b.volume.copy())
    altered.close.loc[execution:, "A"] = 100.
    altered.volume.loc[execution:, "A"] = 1.
    costs = []
    for i, data in enumerate((b, altered)):
        c = cfg.copy_with({"live.ledger_dir": str(tmp_path / str(i))})
        led = PaperLedger(c)
        led.submit(recs(50), str(decision.date()))
        led.update(data.truncate(execution))
        costs.append(led.state["trades"][0]["cost"])
    assert costs[0] == costs[1]


def test_stop_sells_only_its_position(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(40, 40, stop=9), str(b.dates[0].date()))
    led.update(b.truncate(b.dates[1]))
    b.close.loc[b.dates[2]:, "A"] = 8.
    b.open.loc[b.dates[3]:, "A"] = 8.
    led.update(b.truncate(b.dates[3]))
    assert "A" not in led.state["positions"]
    assert led.state["positions"]["B"]["shares"] == 40.
    assert len([t for t in led.state["trades"] if t["ticker"] == "B"]) == 1


def test_legacy_migration_preserves_bytes_and_unknown_basis(accounting_cfg):
    led = PaperLedger(accounting_cfg)
    old = {"start": "2026-01-05", "last_update": "2026-01-06", "cash": 500.,
           "positions": {"A": {"shares": 50., "entry": "2026-01-06", "entry_px": 10., "last": 10., "stop": None}},
           "pending": [], "equity": [], "trades": []}
    raw = json.dumps(old)
    led.path.write_text(raw)
    migrated = PaperLedger(accounting_cfg)
    assert migrated.state["positions"]["A"]["cost_basis"] is None
    assert migrated.state["realized_pnl"] is None
    migrated.save()
    assert migrated.backup_path.read_text() == raw
    assert json.loads(migrated.path.read_text())["schema_version"] == 2
    assert migrated.state["positions"]["A"]["shares"] == 50


def test_atomic_save_rejects_nan_and_stale_writer(accounting_cfg):
    led = PaperLedger(accounting_cfg)
    led.save()
    old = led.path.read_text()
    led.state["cash"] = float("nan")
    with pytest.raises(ValueError):
        led.save()
    assert led.path.read_text() == old
    a, b = PaperLedger(accounting_cfg), PaperLedger(accounting_cfg)
    a.save()
    with pytest.raises(ValueError, match="concurrently"):
        b.save()


def test_raw_prices_and_revised_adjustment_history_blocked(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(50), str(b.dates[0].date()))
    raw = replace(b, meta={"source": "test", "price_basis": "raw"})
    with pytest.raises(ValueError, match="raw corporate actions"):
        led.update(raw)
    led.update(b.truncate(b.dates[1]))
    b.close.loc[b.dates[1], "A"] = 5.
    with pytest.raises(ValueError, match="Historical marks changed"):
        led.update(b.truncate(b.dates[2]))


def test_additional_purchase_uses_weighted_basis(accounting_cfg):
    b = market()
    led = PaperLedger(accounting_cfg)
    led.submit(recs(20), str(b.dates[0].date()))
    led.update(b.truncate(b.dates[1]))
    b.open.loc[b.dates[2]:, "A"] = 20.
    b.close.loc[b.dates[2]:, "A"] = 20.
    led.submit(recs(50), str(b.dates[1].date()))
    led.update(b.truncate(b.dates[2]))
    pos = led.state["positions"]["A"]
    buys = [t for t in led.state["trades"] if t["shares"] > 0]
    basis = sum(t["shares"] * t["px"] + t["cost"] for t in buys)
    assert pos["cost_basis"] == pytest.approx(basis)
    assert pos["entry_px"] == pytest.approx(sum(t["shares"] * t["px"] for t in buys) / pos["shares"])


def test_invalid_inputs_do_not_change_ledger(accounting_cfg):
    led = PaperLedger(accounting_cfg)
    with pytest.raises(ValueError, match="exposure"):
        led.submit(recs(70, 70), "2026-01-05")
    with pytest.raises(ValueError, match="finite"):
        led.submit(recs(float("nan")), "2026-01-05")
    assert not led.path.exists()


def test_known_synthetic_delisting_settles_once_after_last_price(accounting_cfg):
    b = market()
    b.delist_returns = pd.Series({"A": -.30})
    for field in ("open", "high", "low", "close", "volume", "market_cap"):
        getattr(b, field).loc[b.dates[3]:, "A"] = np.nan
    led = PaperLedger(accounting_cfg)
    led.submit(recs(50), str(b.dates[0].date()), "delisting-buy")
    led.update(b.truncate(b.dates[2]))
    assert led.state["positions"]["A"]["shares"] == 50
    assert not any(o["kind"] == "delisting" for o in led.state["orders"])
    led.submit(recs(), str(b.dates[2].date()), "pending-sale")
    led.update(b.truncate(b.dates[3]))
    # $495 cash + 50 units × $10 last close × (1-.30) = $845.
    assert led.state["cash"] == pytest.approx(845.)
    assert led.state["realized_pnl"] == pytest.approx(-155.)
    assert not led.state["positions"] and not led.state["pending"]
    led.update(b.truncate(b.dates[4]))
    assert led.state["cash"] == pytest.approx(845.)
    assert len([o for o in led.state["orders"] if o["kind"] == "delisting"]) == 1


def test_undated_real_delisting_fails_without_destroying_ledger(accounting_cfg):
    b = market()
    b.meta["source"] = "csv"
    b.delist_returns = pd.Series({"A": -.30})
    for field in ("open", "high", "low", "close", "volume", "market_cap"):
        getattr(b, field).loc[b.dates[3]:, "A"] = np.nan
    led = PaperLedger(accounting_cfg)
    led.submit(recs(50), str(b.dates[0].date()))
    led.update(b.truncate(b.dates[2]))
    saved = led.path.read_text()
    with pytest.raises(ValueError, match="dated real-provider settlement"):
        led.update(b.truncate(b.dates[3]))
    assert led.path.read_text() == saved


def test_incomplete_legacy_sell_history_preserves_unknown_pnl(accounting_cfg):
    led = PaperLedger(accounting_cfg)
    old = {"start": "2026-01-05", "last_update": "2026-01-09", "cash": 500.,
           "positions": {}, "pending": [], "equity": [], "trades": [
               {"ticker": "A", "shares": -5., "px": 10., "cost": 1.},
               {"ticker": "B", "shares": 10., "px": 10., "cost": 1.},
               {"ticker": "B", "shares": -10., "px": 12., "cost": 1.}]}
    led.path.write_text(json.dumps(old))
    migrated = PaperLedger(accounting_cfg)
    assert migrated.state["realized_pnl"] is None
    assert migrated.state["initial_capital"] is None
    assert migrated.state["trades"] == old["trades"]
