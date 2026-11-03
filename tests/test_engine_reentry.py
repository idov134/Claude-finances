"""Execution-time information, persistent risk exits and optional restart policy."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stocksel.backtest.engine import run_daily
from stocksel.data.bundle import DataBundle
from stocksel.validation.metrics import max_drawdown


def market(n=65):
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(100., index=dates, columns=["A"])
    bench = pd.DataFrame({"open": 100., "close": 100.}, index=dates)
    return DataBundle(open=close.copy(), high=close + 1, low=close - 1, close=close,
                      volume=close * 100_000, market_cap=close * 100_000_000,
                      sectors=pd.Series({"A": "Tech"}), benchmark=bench,
                      risk_free=pd.Series(0., index=dates))


def crash_market():
    b = market()
    b.close.iloc[31:, 0] = 70.
    b.open.iloc[32:, 0] = 70.
    b.high = b.close + 1
    b.low = b.close - 1
    b.benchmark.loc[b.dates[31:], "close"] = 70 + np.arange(len(b.dates) - 31) * 3
    b.benchmark.loc[b.dates[32:], "open"] = b.benchmark.close.iloc[31:-1].to_numpy()
    targets = {b.dates[i]: pd.Series({"A": .9}) for i in (29, 35, 41, 47)}
    return b, targets


def test_first_period_loss_is_in_drawdown():
    assert max_drawdown([-.20, .01]) == pytest.approx(-.20)
    assert max_drawdown([-1., .1]) == -1


def test_opening_cost_does_not_use_execution_day_close_or_volume(cfg):
    b = market()
    d, entry = b.dates[29], b.dates[30]
    altered = replace(b, close=b.close.copy(), volume=b.volume.copy())
    altered.close.loc[entry, "A"] = 300.
    altered.volume.loc[entry, "A"] = 1.
    kwargs = {"use_stops": False, "kill_switch": False}
    a = run_daily(b, {d: pd.Series({"A": .5})}, cfg, **kwargs)
    c = run_daily(altered, {d: pd.Series({"A": .5})}, cfg, **kwargs)
    assert a.daily.loc[entry, "cost"] > 0
    assert a.daily.loc[entry, "cost"] == c.daily.loc[entry, "cost"]


def test_monthly_retains_first_opening_cost_and_benchmark_move(cfg):
    b = market(n=31)
    b.benchmark.loc[b.dates[-1], "close"] = 102.
    r = run_daily(b, {b.dates[29]: pd.Series({"A": .5})}, cfg,
                  use_stops=False, kill_switch=False)
    m = r.monthly()
    assert len(m) == 1
    assert m.net.iloc[0] == pytest.approx(-r.daily.cost.iloc[0] / cfg.capital.initial)
    assert m.bench.iloc[0] == pytest.approx(.02)


def test_permanent_halt_is_default(cfg):
    b, targets = crash_market()
    r = run_daily(b, targets, cfg, use_stops=False, kill_switch=True)
    assert len(r.kill_events) == 1
    assert r.daily.loc[b.dates[32]:, "killed"].all()
    assert (r.daily.loc[b.dates[32]:, "exposure"] == 0).all()
    assert not r.reentry_events


def test_reentry_waits_for_cooldown_and_scheduled_rebalance(cfg):
    b, targets = crash_market()
    cfg = cfg.copy_with({"risk.kill_switch.reentry": {
        "enabled": True, "cooldown_days": 10, "trend_days": 3, "recovery_fraction": .05}})
    r = run_daily(b, targets, cfg, use_stops=False, kill_switch=True)
    assert r.kill_events[0]["date"] == b.dates[31]
    assert r.daily.loc[b.dates[36], "exposure"] == 0  # recovered, but too soon
    assert r.reentry_events[0]["date"] == b.dates[42]
    assert r.daily.loc[b.dates[42], "exposure"] > 0
    assert not r.daily.loc[b.dates[42], "killed"]


def test_reentry_cannot_use_execution_day_benchmark_close(cfg):
    b, targets = crash_market()
    cfg = cfg.copy_with({"risk.kill_switch.reentry": {
        "enabled": True, "cooldown_days": 3, "trend_days": 3, "recovery_fraction": .05}})
    altered = replace(b, benchmark=b.benchmark.copy())
    altered.benchmark.loc[b.dates[36]:, "close"] = 1.
    a = run_daily(b, targets, cfg, use_stops=False, kill_switch=True)
    c = run_daily(altered, targets, cfg, use_stops=False, kill_switch=True)
    assert a.reentry_events[0]["date"] == c.reentry_events[0]["date"] == b.dates[36]
    assert a.daily.loc[b.dates[36], "cost"] == c.daily.loc[b.dates[36], "cost"]


def test_pending_liquidation_survives_missing_open(cfg):
    b, targets = crash_market()
    b.open.iloc[32, 0] = np.nan
    r = run_daily(b, targets, cfg, use_stops=False, kill_switch=True)
    assert r.daily.loc[b.dates[32], "n_pos"] == 1
    assert r.daily.loc[b.dates[33], "n_pos"] == 0
    exits = r.trades[r.trades.reason == "kill_switch"]
    assert len(exits) == 1 and exits.iloc[0].exit_date == b.dates[33]


def test_invalid_reentry_rule_is_rejected(cfg):
    b = market()
    cfg = cfg.copy_with({"risk.kill_switch.reentry": {"enabled": True, "cooldown_days": 0}})
    with pytest.raises(ValueError, match="reentry requires"):
        run_daily(b, {b.dates[29]: pd.Series({"A": .5})}, cfg)
