"""Tests for the algorithm-research v2 building blocks: corrected monthly engine, residual momentum,
sector-neutral ranks, tails target, tuned LightGBM, score smoothing and the relative mandate."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stocksel.backtest.costs import CostModel
from stocksel.backtest.monthly import executable_targets, fee_reserved_weights, monthly_backtest
from stocksel.features.normalize import cross_sectional, group_rank
from stocksel.features.residual import residual_momentum
from stocksel.models.labels import forward_returns, label_panel
from stocksel.models.train import fit_calibrated, smooth_predictions
from stocksel.portfolio.sizing import SizingParams, target_weights


# ------------------------------------------------------------------ corrected monthly engine
def test_fee_reserve_keeps_cash_nonnegative(cfg):
    cm = CostModel(cfg.costs, 3.0)
    tgt = pd.Series(0.05, index=[f"T{i}" for i in range(20)])  # fully invested
    w, comp, turnover = fee_reserved_weights(tgt, pd.Series(dtype=float), pd.Series(2e6, index=tgt.index),
                                             pd.Series(0.03, index=tgt.index), 1e6, cm)
    assert w.sum() + sum(comp.values()) <= 1 + 1e-9 and w.sum() < 1


def test_unavailable_open_is_unfilled_not_zero_return():
    wanted = pd.Series({"A": 0.5, "B": 0.5})
    prev = pd.Series({"B": 0.3, "C": 0.2})
    px = pd.Series({"A": np.nan, "B": np.nan, "C": 10.0})
    w, unfilled = executable_targets(wanted, prev, px)
    assert "A" not in w.index           # new buy without an open stays cash
    assert w["B"] == 0.3                # existing holding keeps its drifted weight
    assert "C" not in w.index           # tradable name dropped from the target is sold
    assert unfilled == 2


def test_monthly_engine_accounting(panel, bundle, cfg):
    lab = forward_returns(bundle, panel.dates, 1)
    X = panel.features(["mom_12_1"])
    preds = pd.DataFrame({"prob": 0.6, "score": X["mom_12_1"], "er": 0.0})
    preds = preds[preds.index.get_level_values(0) >= panel.dates[30]]
    prm = SizingParams.from_cfg(cfg, 1.0, 10.0, mandate="relative", min_prob=0.0)
    bt = monthly_backtest(preds, panel, lab, bundle, cfg, prm)
    m = bt.monthly
    assert (m["exposure"] + m["cost"] <= 1 + 1e-9).all()           # never levered by fees
    assert m["exposure"].median() > 0.95                            # relative mandate is fully invested
    # net return identity
    assert np.allclose(m["net"], m["gross"] - m["cost"] * (1 + m["rf"]))


# ------------------------------------------------------------------ residual momentum
def test_residual_momentum_ignores_future(bundle):
    dates = bundle.month_end_dates()
    cut = dates[80]
    a = residual_momentum(bundle, dates)
    close = bundle.close.copy()
    fut = bundle.dates > cut
    close.loc[fut] *= np.random.default_rng(0).uniform(0.5, 1.5, close.loc[fut].shape)
    bench = bundle.benchmark.copy()
    bench.loc[fut] *= 1.3
    b = residual_momentum(replace(bundle, close=close, benchmark=bench), dates)
    for k in a:
        pd.testing.assert_frame_equal(a[k].loc[:cut], b[k].loc[:cut])


def test_residual_momentum_removes_beta_and_finds_drift():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    T, N = len(dates), 40
    m = rng.normal(0.01, 0.05, T)
    beta = np.linspace(0.5, 1.5, N)
    # Blitz et al. regress with an intercept over 36 months, so residual momentum measures *recent*
    # idiosyncratic outperformance relative to the stock's own norm: plant drift in the last year only.
    drift = np.where(np.arange(N) % 2 == 0, 0.015, -0.015)
    recent = (np.arange(T) >= T - 13)[:, None]
    r = beta[None, :] * m[:, None] + recent * drift[None, :] + rng.normal(0, 0.02, (T, N))
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=[f"T{i}" for i in range(N)])
    bench = pd.DataFrame({"open": 100 * np.cumprod(1 + m), "close": 100 * np.cumprod(1 + m)}, index=dates)

    class B:  # minimal stand-in exposing what residual_momentum reads
        pass
    b = B()
    b.close, b.benchmark = close, bench
    b.sectors = pd.Series(["S1"] * (N // 2) + ["S2"] * (N // 2), index=close.columns)
    out = residual_momentum(b, dates)["rmom_12_1"].iloc[-1]
    assert out[::2].mean() > 0 > out[1::2].mean()
    assert abs(np.corrcoef(out.to_numpy(), beta)[0, 1]) < 0.4   # not a beta bet


# ------------------------------------------------------------------ sector-neutral ranks
def test_group_rank_within_sector():
    x = pd.DataFrame([[1, 2, 3, 10, 20, 30, 5]], columns=list("abcdefg"), dtype=float)
    g = pd.Series(["s1", "s1", "s1", "s2", "s2", "s2", "s3"], index=x.columns)
    r = group_rank(x, g, min_group=3)
    assert r.loc[0, "a"] == -0.5 and r.loc[0, "c"] == 0.5 and r.loc[0, "d"] == -0.5 and r.loc[0, "f"] == 0.5
    assert -0.5 <= r.loc[0, "g"] <= 0.5        # thin sector falls back to the all-name rank


def test_neutralized_features_change_and_stay_bounded(bundle, cfg):
    from stocksel.features.pipeline import build_features
    p = build_features(bundle, cfg.copy_with({"features.neutralize": "sector"}))
    assert p.X["mom_12_1"].between(-0.5, 0.5).all()
    sec = bundle.sectors.reindex(p.X.index.get_level_values(1)).to_numpy()
    d = p.X.index.get_level_values(0)
    g = pd.DataFrame({"v": p.X["value_bm"].to_numpy(), "s": sec, "d": d}).groupby(["d", "s"])["v"]
    assert g.mean().abs().median() < 0.05      # each sector is centred
    with pytest.raises(ValueError):
        build_features(bundle, cfg.copy_with({"features.neutralize": "industry"}))


# ------------------------------------------------------------------ tails target / tuned LightGBM
def test_tails_target_trains_on_extremes_only(panel, bundle, cfg):
    lab = forward_returns(bundle, panel.dates, 1)
    L = label_panel(lab, panel.X.index)
    X = panel.features(["mom_12_1", "vol_252", "sue"])
    ok = L["y"].notna().to_numpy()
    full = fit_calibrated("logit", X[ok], L[ok], cfg)
    tails = fit_calibrated("logit", X[ok], L[ok], cfg.copy_with({"labels.target": "tails", "labels.tail_frac": 0.3}))
    assert 0.5 * full.n_train < tails.n_train < 0.7 * full.n_train
    p = tails.predict(X[ok].to_numpy()[:200])["prob"]
    assert ((p > 0) & (p < 1)).all()


def test_tuned_lgbm_selects_from_grid(panel, bundle, cfg):
    lab = forward_returns(bundle, panel.dates, 1)
    L = label_panel(lab, panel.X.index)
    X = panel.features(["mom_12_1", "vol_252", "sue", "idio_vol"])
    ok = L["y"].notna().to_numpy()
    fm = fit_calibrated("lgbm_tuned", X[ok], L[ok], cfg.copy_with({
        "models.params.lgbm_tuned.grid": {"num_leaves": [3, 7]},
        "models.params.lgbm_tuned.fixed": {"n_estimators": 30, "learning_rate": 0.05, "min_child_samples": 100}}))
    assert fm.model.best_params_["num_leaves"] in (3, 7)


# ------------------------------------------------------------------ smoothing / mandate
def test_smoothing_uses_past_only():
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-31", periods=4, freq="ME"), ["A", "B"]])
    p = pd.DataFrame({"prob": [0.6, 0.4, 0.4, 0.6, 0.6, 0.4, 0.5, 0.5], "score": 0.0, "er": 0.0}, index=idx)
    s = smooth_predictions(p, 0.5)
    assert s.loc[idx[0], "prob"] == 0.6 and np.isclose(s.loc[idx[2], "prob"], 0.5)
    p2 = p.copy()
    p2.iloc[-2:, 0] = 0.9                         # change the last month only
    s2 = smooth_predictions(p2, 0.5)
    pd.testing.assert_frame_equal(s.iloc[:-2], s2.iloc[:-2])
    assert smooth_predictions(p, 0.0) is p


def test_relative_mandate_fully_invested_with_caps():
    rng = np.random.default_rng(0)
    n = 80
    c = pd.DataFrame({"prob": rng.uniform(0.45, 0.6, n), "score": rng.standard_normal(n),
                      "mcap": np.exp(rng.normal(23, 1, n)), "sector": rng.choice(list("ABCDE"), n),
                      "idio_vol": rng.uniform(0.15, 0.4, n), "beta": rng.uniform(0.7, 1.3, n)},
                     index=[f"T{i}" for i in range(n)])
    w, _ = target_weights(c, pd.Series(dtype=float), SizingParams(top_n=40, mandate="relative", vol_target=0.05), 0.2)
    assert np.isclose(w.sum(), 1.0) and w.max() <= 0.05 + 1e-9
    wa, _ = target_weights(c, pd.Series(dtype=float), SizingParams(top_n=40, vol_target=0.05), 0.2)
    assert wa.sum() < 0.8                          # the absolute mandate de-risks
    with pytest.raises(ValueError):
        target_weights(c, pd.Series(dtype=float), SizingParams(mandate="other"), 0.2)


# ------------------------------------------------------------------ real-data builder
def test_split_adjusted_shares_matches_split_adjusted_prices():
    from stocksel.data.build_yahoo_csv import split_adjusted_shares
    # AAPL-like: 4.44B shares reported before a 4:1 split on 2020-08-31, 17B after (raw counts)
    raw = pd.Series([4.6e9, 4.44e9, 4.44e9, 17.0e9],
                    index=pd.to_datetime(["2019-06-30", "2020-06-30", "2020-09-15", "2020-12-31"]))
    splits = pd.Series([7.0, 4.0], index=pd.to_datetime(["2014-06-09", "2020-08-31"]))
    adj = split_adjusted_shares(raw, splits)
    assert np.allclose(adj.iloc[:3], raw.iloc[:3] * 4)    # pre-split reports (even one lagging the split) x4
    assert adj.iloc[3] == 17.0e9                           # post-split report already in today's units
    fresh = pd.Series([4.44e9, 17.6e9], index=pd.to_datetime(["2020-06-30", "2020-09-15"]))
    assert split_adjusted_shares(fresh, splits).iloc[1] == 17.6e9   # a genuine post-split count is kept
    assert split_adjusted_shares(raw, None).equals(raw)
