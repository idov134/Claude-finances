import numpy as np
import pandas as pd
import pytest

from stocksel.backtest.costs import CostModel
from stocksel.config import ConfigError, load_config
from stocksel.portfolio.risk import deviation_test, drawdown_breach
from stocksel.portfolio.sizing import SizingParams, kelly_cap, target_weights


def _cands(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = [f"T{i}" for i in range(n)]
    return pd.DataFrame({"prob": rng.uniform(0.5, 0.62, n), "score": rng.standard_normal(n),
                         "mcap": np.exp(rng.normal(23, 1, n)), "sector": rng.choice(list("ABCDE"), n),
                         "idio_vol": rng.uniform(0.15, 0.4, n), "beta": rng.uniform(0.7, 1.3, n)}, index=idx)


def test_caps_respected():
    prm = SizingParams(top_n=40, max_position=0.05, max_sector=0.3, vol_target=1.0, keep=0.5)
    w, d = target_weights(_cands(), pd.Series(dtype=float), prm, 0.16)
    c = _cands()
    assert w.max() <= 0.05 + 1e-9
    assert w.groupby(c.loc[w.index, "sector"]).sum().max() <= 0.3 + 1e-9
    assert w.sum() <= 1.0 + 1e-9


def test_vol_target_scales_down():
    c = _cands()
    hi = target_weights(c, pd.Series(dtype=float), SizingParams(vol_target=5.0), 0.16)[0].sum()
    lo = target_weights(c, pd.Series(dtype=float), SizingParams(vol_target=0.05), 0.16)[0].sum()
    assert lo < hi


def test_kelly_haircut_and_sign():
    assert kelly_cap(0.5, 0.3, 0.25) == 0
    assert kelly_cap(0.55, 0.3, 0.25) > kelly_cap(0.525, 0.3, 0.25) > 0
    assert kelly_cap(0.45, 0.3, 0.25) < 0


def test_buffer_keeps_existing_holdings():
    c = _cands()
    c["prob"] = 0.6
    rank = c["score"].rank(ascending=False)
    held = rank[(rank > 40) & (rank <= 60)].index[:3]
    prev = pd.Series(0.02, index=held)
    w_buf, _ = target_weights(c, prev, SizingParams(top_n=40, buffer=True, exit_rank_mult=1.6, vol_target=5), 0.16)
    w_nob, _ = target_weights(c, prev, SizingParams(top_n=40, buffer=False, vol_target=5), 0.16)
    assert set(held) <= set(w_buf.index) and not (set(held) & set(w_nob.index))


def test_cost_model_monotone():
    cfg = load_config("config.yaml")
    cm = CostModel(cfg.costs)
    small = cm.rate(np.array([1e4]), np.array([1e8]), np.array([0.02]))[0]
    big = cm.rate(np.array([1e7]), np.array([1e8]), np.array([0.02]))[0]
    illiquid = cm.rate(np.array([1e4]), np.array([1e6]), np.array([0.02]))[0]
    assert big > small and illiquid > small
    assert CostModel(cfg.costs, 2.0).rate(np.array([1e4]), np.array([1e8]), np.array([0.02]))[0] == pytest.approx(2 * small)


def test_kill_switch_rules():
    eq = pd.Series([100, 120, 90.0])
    assert drawdown_breach(eq, 0.2).kill and not drawdown_breach(eq, 0.3).kill
    bt = pd.Series(np.random.default_rng(0).normal(0.01, 0.03, 200))
    assert deviation_test(pd.Series([-0.06, -0.05, -0.07, -0.04]), bt).kill
    assert not deviation_test(pd.Series([0.01, 0.02]), bt).kill  # too few months


@pytest.mark.parametrize("over", [{"validation.holdout_months": 12}, {"portfolio.kelly_fraction": 0.5},
                                  {"universe.min_market_cap": 5.0e7}, {"portfolio.max_gross": 1.5}])
def test_config_guards(over):
    with pytest.raises(ConfigError):
        load_config("config.yaml", over)
