"""Hand-checkable research diagnostics; no holdout lock is opened."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stocksel.backtest.costs import CostModel
from stocksel.experiments.comparison import (economics, fee_reserved_weights,
    forecast_diagnostics, longest_drawdown_months, research_only, rank_diagnostics,
    executable_targets)


def test_fee_reserve_prevents_implicit_leverage():
    target = pd.Series({"A": 1.})
    cm = CostModel(SimpleNamespace(commission_bps=100., min_half_spread_bps=0., spread_k=0.,
                                  slippage_bps=0., impact_k=0.))
    w, fees, turnover = fee_reserved_weights(target, pd.Series(dtype=float), pd.Series({"A": 1e9}),
                                            pd.Series({"A": .01}), 100., cm)
    assert w.sum() + fees <= 1 + 1e-12
    assert 0 < w.A <= 1 / 1.01  # A conservative residual cash reserve is allowed.
    assert fees == pytest.approx(w.A * .01)
    assert turnover == pytest.approx(w.A)


def test_economics_does_not_invent_positive_edge():
    assert economics(-.01, -.02, 2., 600.)["total_break_even_cost_bps_per_traded_dollar"] is None
    assert economics(.01, 0., 2., 600.)["capital_usd_to_cover_fixed_cost_if_point_estimate_persisted"] is None
    case = economics(.02, .01, 2., 600.)
    assert case["total_break_even_cost_bps_per_traded_dollar"] == 100
    assert case["additional_break_even_cost_bps_per_traded_dollar"] == 50
    assert case["capital_usd_to_cover_fixed_cost_if_point_estimate_persisted"] == 60000


def test_missing_entry_open_is_unfilled_without_replacement_selection():
    desired = pd.Series({"NEW": .2, "OLD": .1, "VALID": .2})
    previous = pd.Series({"OLD": .15})
    prices = pd.Series({"NEW": np.nan, "OLD": np.nan, "VALID": 100.})
    actual, count = executable_targets(desired, previous, prices)
    assert "NEW" not in actual
    assert actual.OLD == .15
    assert actual.VALID == .2
    assert count == 2
    # Entry-only handling does not consult any later return or use cash to
    # replace a missing name with the successful opening name.
    assert actual.sum() == .35


def test_drawdown_duration_includes_first_loss_and_unrecovered_tail():
    assert longest_drawdown_months(pd.Series([-.1, .05, .2, -.1, -.1, .01])) == 3


def test_probability_event_calibration_and_rank_are_separate():
    index = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-28"]), list("ABCDEFGHIJ")], names=["date", "ticker"])
    scores = pd.Series(np.tile(np.arange(10), 2), index=index)
    labels = pd.DataFrame({"y": (scores > 4).astype(float), "excess": scores / 100 - .04, "fwd": scores / 100}, index=index)
    pred = pd.DataFrame({"prob": .5, "score": scores}, index=index)
    forecast = forecast_diagnostics(pred, labels)
    assert forecast["brier"] == .25
    assert forecast["constant_half_probability"]["brier"] == .25
    assert len(forecast["calibration"]) == 10
    assert sum(x["n"] for x in forecast["calibration"]) == 20
    assert any(x["observed_fraction"] is None for x in forecast["calibration"])
    ranks = rank_diagnostics(scores, labels)
    assert ranks["ic_mean"] == pytest.approx(1.)
    assert ranks["quintiles_low_to_high"][-1]["mean_monthly_return"] > ranks["quintiles_low_to_high"][0]["mean_monthly_return"]


def test_research_firewall_removes_final_two_years(cfg, bundle):
    research, boundary = research_only(bundle, cfg)
    assert research.dates[-1] < boundary
    assert research.dates[-1] < bundle.dates[-1] - pd.DateOffset(months=23)
    assert boundary <= pd.Timestamp("2024-01-01")
    assert "truncated_at" in research.meta


def test_delisting_proceeds_become_cash_without_a_second_sale(cfg, monkeypatch):
    import stocksel.experiments.comparison as comparison
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    entries = pd.to_datetime(["2020-02-03", "2020-03-02"])
    exits = pd.to_datetime(["2020-03-02", "2020-04-01"])
    index = pd.MultiIndex.from_product([dates, ["DELISTED"]], names=["date", "ticker"])
    predictions = pd.DataFrame({"prob": .6, "score": .6}, index=index)
    factors = pd.Series(0., index=index)
    labels = SimpleNamespace(bench=pd.Series(0., index=dates),
        entry=pd.Series(entries, index=dates), exit=pd.Series(exits, index=dates),
        fwd=pd.DataFrame({"DELISTED": [-.5, np.nan]}, index=dates))
    bundle = SimpleNamespace(open=pd.DataFrame({"DELISTED": [100., np.nan, np.nan]},
        index=pd.to_datetime(["2020-02-03", "2020-03-02", "2020-04-01"])),
        sectors=pd.Series({"DELISTED": "Tech"}), risk_free=pd.Series(0., index=dates))
    panel = SimpleNamespace(aux={"adv": pd.DataFrame(1e9, index=dates, columns=["DELISTED"]),
                                 "vol_d": pd.DataFrame(.02, index=dates, columns=["DELISTED"])})
    monkeypatch.setattr(comparison, "market_vol", lambda b, ds: pd.Series(.1, index=ds))
    monkeypatch.setattr(comparison, "candidates_at", lambda *args: pd.DataFrame(index=["DELISTED"]))
    observed_previous = []
    def targets(rule, cands, scores, previous, cfg, sigma):
        observed_previous.append(previous.copy())
        return pd.Series({"DELISTED": .5}) if len(observed_previous) == 1 else pd.Series(dtype=float)
    monkeypatch.setattr(comparison, "allocated_weights", targets)
    cfg = cfg.copy_with({"costs.commission_bps": 0., "costs.min_half_spread_bps": 0.,
                         "costs.spread_k": 0., "costs.slippage_bps": 0., "costs.impact_k": 0.})
    result = comparison.backtest("current_logit", 1, predictions, factors, panel, labels, bundle, cfg)
    assert result.net.iloc[0] == -.25  # 50% position loses 50%, including terminal settlement.
    assert result.equity.iloc[-1] == cfg.capital.initial * .75
    assert observed_previous[1].empty  # Proceeds are cash, not phantom shares.
    assert result.turnover.iloc[1] == 0  # No second sale/charge of the settled security.
