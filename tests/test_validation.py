from math import comb

import numpy as np
import pandas as pd
import pytest

from stocksel.validation.metrics import (deflated_sharpe, expected_max_sharpe, max_drawdown, psr, sharpe)
from stocksel.validation.pbo import pbo
from stocksel.validation.splits import CombinatorialPurgedCV, PurgedKFold, walk_forward_schedule


def test_sharpe_and_drawdown():
    r = np.array([0.01, -0.02, 0.03, 0.01])
    assert np.isclose(sharpe(r, ppy=12), r.mean() / r.std(ddof=1) * np.sqrt(12))
    assert np.isclose(max_drawdown(pd.Series([0.1, -0.5, 0.2])), -0.5)


def test_dsr_penalizes_more_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.04, 240)
    ts = rng.normal(0, 0.05, 200)
    d10 = deflated_sharpe(r, 10, ts)["dsr"]
    d1000 = deflated_sharpe(r, 1000, ts)["dsr"]
    assert d1000 < d10 <= psr(r)
    assert expected_max_sharpe(1000, 0.0025) > expected_max_sharpe(10, 0.0025) > 0


def test_pbo_noise_vs_signal():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(0, 0.04, (240, 40)))
    p_noise = pbo(noise, 12, 2000)["pbo"]
    assert 0.3 < p_noise < 0.7
    sig = noise.copy()
    sig[0] = rng.normal(0.02, 0.04, 240)
    assert pbo(sig, 12, 2000)["pbo"] < 0.1


def _toy(n_dates=60, per=5, h=3):
    d = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    start = pd.Series(np.repeat(d, per))
    t1 = start + pd.DateOffset(months=h)
    return start, t1


def test_purged_kfold_no_overlap_and_embargo():
    start, t1 = _toy()
    cv = PurgedKFold(5, embargo_pct=0.05)
    for tr, te in cv.split(start, t1):
        t0, tmax = start[te].min(), t1[te].max()
        s, e = start[tr], t1[tr]
        assert not ((s <= tmax) & (e >= t0)).any()
        emb = (start.max() - start.min()) * 0.05
        assert not ((s > tmax) & (s <= tmax + emb)).any()


def test_cpcv_paths():
    start, t1 = _toy(h=1)
    cv = CombinatorialPurgedCV(6, 2, 0.0)
    splits = list(cv.split(start, t1))
    assert len(splits) == comb(6, 2) == 15 and cv.n_paths == 5
    for path in cv.paths():
        assert sorted(g for g, _ in path) == list(range(6))
    for groups, tr, te in splits:
        assert len(set(tr) & set(te)) == 0


def test_walk_forward_schedule_monotone():
    d = pd.date_range("2000-01-31", periods=100, freq="ME")
    s = walk_forward_schedule(d, 36, 60, 12)
    assert s[0][1] == d[36] and all(a[1] < b[1] for a, b in zip(s, s[1:]))
    assert all(start >= R - pd.DateOffset(months=61) for start, R, _ in s)


def test_alpha_returns_remove_market_exposure():
    from stocksel.validation.metrics import alpha_returns
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.007, 0.045, 240))
    no_skill = 0.6 * b + rng.normal(0, 0.01, 240)               # pure beta: positive Sharpe, no alpha
    skill = 0.6 * b + 0.004 + rng.normal(0, 0.01, 240)
    assert sharpe(no_skill) > 0.2                                  # positive Sharpe from beta alone
    assert abs(sharpe(alpha_returns(no_skill, b))) < 0.5 < sharpe(alpha_returns(skill, b))
