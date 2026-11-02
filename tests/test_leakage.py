"""Unit tests for information leakage. Each test perturbs the future and checks the past is unchanged,
or checks the exact availability rule of a data type."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stocksel.data.pit import asof_wide
from stocksel.errors import HoldoutError, LeakageError
from stocksel.features.fundamental import fundamental_features
from stocksel.features.news import (ClaudeScorer, LexiconScorer, assert_cutoff_before, daily_sentiment,
                                    effective_day_positions)
from stocksel.features.normalize import cross_sectional
from stocksel.features.price import price_features
from stocksel.models.labels import forward_returns


def _perturb_after(bundle, d, seed=0):
    rng = np.random.default_rng(seed)
    fut = bundle.dates > d
    out = {}
    for f in ("open", "high", "low", "close", "volume", "market_cap"):
        df = getattr(bundle, f).copy()
        df.loc[fut] = df.loc[fut] * rng.uniform(0.5, 1.5, size=df.loc[fut].shape)
        out[f] = df
    bench = bundle.benchmark.copy()
    bench.loc[fut] *= 1.7
    return replace(bundle, benchmark=bench, **out)


def test_price_features_do_not_use_future(bundle):
    dates = bundle.month_end_dates()
    cut = dates[60]
    a = price_features(bundle, dates)
    b = price_features(_perturb_after(bundle, cut), dates)
    for k in a:
        pd.testing.assert_frame_equal(a[k].loc[:cut], b[k].loc[:cut], check_exact=False, rtol=1e-12)


def test_fundamentals_respect_publication_date(bundle, cfg):
    dates = bundle.month_end_dates()
    cut = dates[60]
    f = bundle.fundamentals.copy()
    future = f["publish_date"] >= cut
    f.loc[future, ["book_equity", "net_income", "gross_profit", "total_assets", "revenue"]] *= 25.0
    b2 = replace(bundle, fundamentals=f)
    a = fundamental_features(bundle, cfg, dates[:61])
    b = fundamental_features(b2, cfg, dates[:61])
    for k in ("value_bm", "value_ep", "quality_gpa", "quality_roe"):
        pd.testing.assert_frame_equal(a[k], b[k])


def test_filing_published_on_decision_day_is_not_used_that_day():
    ev = pd.DataFrame({"ticker": ["A"], "publish_date": [pd.Timestamp("2020-03-10")], "x": [1.0]})
    dates = pd.DatetimeIndex(["2020-03-09", "2020-03-10", "2020-03-11"])
    w = asof_wide(ev, "publish_date", ["x"], dates, ["A"], lag_days=1)["x"]
    assert np.isnan(w.loc["2020-03-10", "A"]) and w.loc["2020-03-11", "A"] == 1.0


def test_after_close_earnings_used_next_day_only():
    ts = pd.Timestamp("2020-03-10 16:30", tz="America/New_York").tz_convert("UTC")
    ev = pd.DataFrame({"ticker": ["A"], "announce_ts": [ts], "x": [1.0]})
    dates = pd.DatetimeIndex(["2020-03-10", "2020-03-11"])
    w = asof_wide(ev, "announce_ts", ["x"], dates, ["A"])["x"]
    assert np.isnan(w.loc["2020-03-10", "A"]) and w.loc["2020-03-11", "A"] == 1.0


@pytest.mark.parametrize("hhmm,expected", [("15:59", 0), ("16:00", 1), ("16:01", 1), ("09:00", 0)])
def test_news_timestamp_alignment(hhmm, expected):
    dates = pd.DatetimeIndex(["2020-03-10", "2020-03-11"])
    ts = pd.Series([pd.Timestamp(f"2020-03-10 {hhmm}", tz="America/New_York").tz_convert("UTC")])
    assert effective_day_positions(ts, dates, "America/New_York", "16:00")[0] == expected


def test_llm_cutoff_guard_and_filter(bundle, cfg):
    s = LexiconScorer()
    s.knowledge_cutoff = pd.Timestamp("2008-01-01")
    with pytest.raises(LeakageError):
        assert_cutoff_before(s, "2007-06-01")
    assert_cutoff_before(s, "2008-02-01")
    ssum, scnt, info = daily_sentiment(bundle, s, cfg)
    assert info["n_dropped_cutoff"] > 0
    assert scnt.loc[:"2008-01-01"].to_numpy().sum() == 0
    assert scnt.loc["2008-01-03":].to_numpy().sum() > 0


def test_unknown_llm_cutoff_defaults_to_today(tmp_path):
    sc = ClaudeScorer(knowledge_cutoff=None, cache_file=str(tmp_path / "c.jsonl"))
    assert sc.knowledge_cutoff >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    with pytest.raises(LeakageError):
        assert_cutoff_before(sc, "2015-01-01")


def test_labels_start_at_next_open_and_ignore_past(bundle):
    dates = bundle.month_end_dates()[10:20]
    lab = forward_returns(bundle, dates, 1)
    d = dates[0]
    e = bundle.dates[bundle.dates.searchsorted(d, side="right")]
    assert lab.entry.iloc[0] == e and lab.exit.iloc[0] > e
    t = lab.fwd.iloc[0].dropna().index[0]
    x = lab.exit.iloc[0]
    assert np.isclose(lab.fwd.iloc[0][t], bundle.open.loc[x, t] / bundle.open.loc[e, t] - 1)
    # changing the close on the decision day must not change the label
    close = bundle.close.copy()
    close.loc[d] *= 3
    lab2 = forward_returns(replace(bundle, close=close), dates, 1)
    pd.testing.assert_frame_equal(lab.fwd, lab2.fwd)


def test_delisted_stocks_keep_their_label(bundle):
    dates = bundle.month_end_dates()
    lab = forward_returns(bundle, dates, 1)
    lv = bundle.close.apply(lambda s: s.last_valid_index())
    delisted = [t for t in bundle.delist_returns.index if lv[t] < bundle.dates[-30]]
    assert delisted, "fixture should contain delistings"
    found = 0
    for t in delisted:
        k = lab.exit.searchsorted(lv[t])  # first decision whose exit is after delisting
        k = min(max(k - 0, 0), len(dates) - 1)
        row = lab.entry.index[(lab.entry <= lv[t]) & (lab.exit > lv[t])]
        if len(row):
            assert np.isfinite(lab.fwd.loc[row[0], t])
            found += 1
    assert found > 0


def test_truncate_is_a_firewall(bundle):
    end = pd.Timestamp("2007-06-29")
    t = bundle.truncate(end)
    assert t.dates.max() <= end
    assert pd.to_datetime(t.fundamentals["publish_date"]).max() <= end
    assert pd.to_datetime(t.estimates["date"]).max() <= end
    assert pd.to_datetime(t.news["timestamp"], utc=True).max() <= end.tz_localize("America/New_York") + pd.Timedelta(days=1)
    lv = bundle.close.apply(lambda s: s.last_valid_index())
    assert all(lv[x] < end for x in t.delist_returns.index)


def test_cross_sectional_normalization_is_per_date():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((3, 50)))
    a = cross_sectional(df)
    df2 = df.copy()
    df2.iloc[2] *= 100
    b = cross_sectional(df2)
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])
    assert a.max().max() <= 0.5 and a.min().min() >= -0.5


def test_walk_forward_training_labels_end_before_retrain(panel, bundle, cfg, monkeypatch):
    import stocksel.models.train as tr
    from stocksel.models.labels import label_panel
    lab = forward_returns(bundle, panel.dates, 1)
    L = label_panel(lab, panel.X.index)
    seen = []
    real = tr.fit_calibrated

    def spy(name, X, L_, cfg_, seed=0):
        seen.append((X.index.get_level_values(0).max(), pd.to_datetime(L_["t1"]).max()))
        return real(name, X, L_, cfg_, seed)
    monkeypatch.setattr(tr, "fit_calibrated", spy)
    wf = tr.walk_forward(panel.features(["mom_12_1", "vol_252"]), L, ["logit"], cfg)
    assert len(seen) == len(wf.schedule) > 0
    for (max_d, max_t1), R in zip(seen, wf.schedule):
        assert max_d < R and max_t1 <= R
    # predictions only for dates at/after the first retrain date
    assert wf.preds.index.get_level_values(0).min() >= wf.schedule[0]


def test_engine_trades_at_next_open(bundle, cfg):
    from stocksel.backtest.engine import run_daily
    d = bundle.month_end_dates()[30]
    t = bundle.close.loc[d].dropna().index[0]
    res = run_daily(bundle, {d: pd.Series({t: 0.5})}, cfg, use_stops=False, kill_switch=False)
    first = res.trades.iloc[0] if len(res.trades) else None
    nxt = bundle.dates[bundle.dates.searchsorted(d, side="right")]
    assert first is not None and first["entry_date"] == nxt
    assert np.isclose(first["entry_px"], bundle.open.loc[nxt, t])


def test_holdout_lock_and_guard(tmp_path, bundle):
    from stocksel.validation.holdout import HoldoutLock, assert_research_data, holdout_start
    hs = holdout_start(bundle.dates, 24)
    with pytest.raises(HoldoutError):
        assert_research_data(bundle, hs)
    assert_research_data(bundle.truncate(hs - pd.Timedelta(days=1)), hs)
    lock = HoldoutLock(tmp_path)
    lock.open("abc", {"x": 1})
    with pytest.raises(HoldoutError):
        lock.open("abc", {"x": 1})
    rec = lock.open("abc", {"x": 1}, force=True)
    assert rec["contaminated"] and rec["n_opens"] == 2
    with pytest.raises(HoldoutError):
        holdout_start(bundle.dates, 12)
