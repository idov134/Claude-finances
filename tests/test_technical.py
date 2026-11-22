import numpy as np
import pandas as pd
import pytest

from stocksel.backtest.rules import monthly, open_to_open, shift_stats, simulate
from stocksel.features.technical import (RULES, Indicators, Market, bollinger, cross_sectional_features, hold,
                                         month_end_mask, rsi, rule_positions, sector_breadth,
                                         turn_of_month_window)
from stocksel.validation.metrics import alpha_tstat


def _prices(n=900, k=3, seed=0, start="2001-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = rng.normal(0.0003, 0.015, size=(n, k))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"S{i}" for i in range(k)])
    open_ = close.shift(1) * np.exp(rng.normal(0, 0.003, size=(n, k)))
    open_.iloc[0] = close.iloc[0]
    return open_, close


def _market(idx, seed=1):
    rng = np.random.default_rng(seed)
    vix = pd.Series(20 + 8 * np.sin(np.arange(len(idx)) / 40) + rng.normal(0, 2, len(idx)), index=idx)
    breadth = pd.Series(rng.integers(0, 10, len(idx)).astype(float), index=idx)
    breadth.iloc[:150] = np.nan
    return Market(vix=vix, breadth=breadth)


def test_rsi_matches_wilder_recursion():
    x = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
                   46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
    n = 14
    d = x.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = [up.iloc[1]], [dn.iloc[1]]
    for i in range(2, len(x)):
        au.append(au[-1] + (up.iloc[i] - au[-1]) / n)
        ad.append(ad[-1] + (dn.iloc[i] - ad[-1]) / n)
    expected = 100 - 100 / (1 + np.array(au) / np.array(ad))
    got = rsi(x.to_frame("a"), n)["a"]
    assert got.iloc[:n].isna().all()
    assert np.allclose(got.iloc[n:].to_numpy(), expected[n - 1:], atol=1e-10)
    up_only = rsi(pd.DataFrame({"a": np.arange(1.0, 40.0)}), 14)["a"].dropna()
    assert (up_only == 100).all()


def test_bollinger_uses_population_std():
    x = pd.DataFrame({"a": np.r_[np.arange(1.0, 30.0), 10.0]})
    mid, up, lo = bollinger(x, 20, 2.0)
    w = x["a"].iloc[-20:].to_numpy()
    assert np.isclose(mid["a"].iloc[-1], w.mean())
    assert np.isclose(up["a"].iloc[-1], w.mean() + 2 * w.std(ddof=0))
    assert np.isclose(lo["a"].iloc[-1], w.mean() - 2 * w.std(ddof=0))


def test_hold_state_machine():
    idx = pd.RangeIndex(8)
    entry = pd.DataFrame({"a": [0, 1, 0, 0, 1, 0, 1, 0]}, index=idx).astype(bool)
    exit_ = pd.DataFrame({"a": [1, 0, 0, 1, 0, 1, 1, 0]}, index=idx).astype(bool)
    valid = pd.DataFrame({"a": [False] + [True] * 7}, index=idx)
    pos = hold(entry, exit_, valid)["a"]
    # day 6 has both: the entry wins
    assert pos.iloc[0] != pos.iloc[0]          # NaN while invalid
    assert pos.iloc[1:].tolist() == [1, 1, 0, 1, 0, 1, 1]


def test_calendar_helpers():
    idx = pd.bdate_range("2024-01-24", "2024-02-09")
    me = month_end_mask(idx)
    assert me[pd.Timestamp("2024-01-31")] and me.sum() == 1          # the final (partial) month is not an end
    tom = turn_of_month_window(idx)
    days = [d.strftime("%m-%d") for d in idx[tom.to_numpy()]]
    # the first partial month (Jan 24-26 counted as its first trading days) aside, the window is exact
    assert [d for d in days if d >= "01-27"] == ["01-31", "02-01", "02-02", "02-05"]
    close = pd.DataFrame({"a": 1.0}, index=idx)
    pos = rule_positions("TOM", Indicators(close), Market(pd.Series(dtype=float), pd.Series(dtype=float)))["a"]
    assert pos[pd.Timestamp("2024-01-30")] == 1.0 and pos[pd.Timestamp("2024-02-05")] == 0.0
    assert pos[pd.Timestamp("2024-02-02")] == 1.0                    # tomorrow (Feb 5) is the 3rd day


@pytest.mark.parametrize("name", RULES)
def test_rules_do_not_look_ahead(name):
    _, close = _prices(n=700, k=3)
    mkt = _market(close.index)
    cut = 450
    base = rule_positions(name, Indicators(close), mkt)
    changed = close.copy()
    rng = np.random.default_rng(9)
    changed.iloc[cut + 1:] *= np.exp(np.cumsum(rng.normal(0, 0.05, size=changed.iloc[cut + 1:].shape), axis=0))
    mkt2 = Market(vix=mkt.vix.copy(), breadth=mkt.breadth.copy())
    mkt2.vix.iloc[cut + 1:] = 99.0
    mkt2.breadth.iloc[cut + 1:] = 0.0
    alt = rule_positions(name, Indicators(changed), mkt2)
    a, b = base.iloc[:cut + 1], alt.iloc[:cut + 1]
    assert ((a == b) | (a.isna() & b.isna())).all().all(), name
    assert set(np.unique(base.dropna().to_numpy())) <= {0.0, 1.0}


def test_engine_next_open_execution_costs_and_cash():
    idx = pd.bdate_range("2020-01-01", periods=6)
    open_ = pd.DataFrame({"a": [100, 101, 103, 102, 104, 105.0]}, index=idx)
    r = open_to_open(open_)
    sig = pd.DataFrame({"a": [0, 1, 1, 0, 0, 0.0]}, index=idx)      # decided at closes of day 1 and 2
    rf = pd.Series(0.0001, index=idx)
    out = simulate(sig, pd.DataFrame(1.0, index=idx, columns=["a"]), r, rf, cost_bps=10)
    assert len(out) == 5                                             # last day has no next open
    # long during days 2 and 3: from the open of day 2 (103) to the open of day 4 (104)
    assert out["exposure"].tolist() == [0, 0, 1, 1, 0]
    assert np.isclose(out["ret"].iloc[2], 102 / 103 - 1 - 0.001)    # entry cost at the open of day 2
    assert np.isclose(out["ret"].iloc[3], 104 / 102 - 1)
    assert np.isclose(out["ret"].iloc[4], 0.0001 - 0.001)           # exit cost, then cash
    assert np.isclose(out["ret"].iloc[0], 0.0001)
    bh = simulate(pd.DataFrame(1.0, index=idx, columns=["a"]), pd.DataFrame(1.0, index=idx, columns=["a"]), r, rf, 0)
    assert np.allclose(bh["ret"].iloc[1:], r["a"].iloc[1:5])


def test_engine_equal_weight_basket():
    open_, _ = _prices(n=60, k=4)
    r = open_to_open(open_)
    sig = pd.DataFrame(1.0, index=open_.index, columns=open_.columns)
    w = sig / 4
    out = simulate(sig, w, r, pd.Series(0.0, index=open_.index), 0)
    assert np.allclose(out["ret"].iloc[1:], r.mean(axis=1).iloc[1:len(out)])


def test_alpha_tstat_recovers_alpha():
    rng = np.random.default_rng(3)
    b = rng.normal(0.006, 0.04, 400)
    y = 0.004 + 0.6 * b + rng.normal(0, 0.01, 400)
    a = alpha_tstat(y, b)
    assert abs(a["alpha"] - 0.048) < 0.015 and abs(a["beta"] - 0.6) < 0.05 and a["t_alpha"] > 5
    z = 0.6 * b + rng.normal(0, 0.01, 400)
    assert abs(alpha_tstat(z, b)["t_alpha"]) < 3


def test_shift_alpha_k0_matches_engine():
    open_, close = _prices(n=800, k=1, seed=4)
    idx = close.index
    r = open_to_open(open_)
    rf = pd.Series(0.00008, index=idx)
    pos = rule_positions("SMA150", Indicators(close), _market(idx)).fillna(0)
    lo = idx[200]
    s = simulate(pos, pd.DataFrame(1.0, index=idx, columns=pos.columns), r, rf, 5).loc[lo:]
    b = simulate(pd.DataFrame(1.0, index=idx, columns=pos.columns), pd.DataFrame(1.0, index=idx, columns=pos.columns),
                 r, rf, 5).loc[lo:]
    rfm = monthly(rf.loc[s.index])
    want = alpha_tstat(monthly(s["ret"]) - rfm, monthly(b["ret"]) - rfm)["alpha"]
    pe = pos["S0"].shift(1).fillna(0).loc[s.index].to_numpy()
    months = s.index.to_period("M")
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    bm = (monthly(b["ret"]) - rfm).to_numpy()
    got, dd = shift_stats(pe, r["S0"].loc[s.index].to_numpy(), rf.loc[s.index].to_numpy(), bm, starts, 5e-4,
                          np.array([0, 50]))
    # the engine charges the entry trade on the window's first day only if it happens inside the window
    assert abs(got[0] - want) < 2e-3
    from stocksel.validation.metrics import max_drawdown
    assert abs(dd[0] - max_drawdown(s["ret"])) < 2e-3 and dd[1] != dd[0]


def test_sector_breadth_and_features():
    _, close = _prices(n=400, k=9, seed=5)
    b = sector_breadth(close, list(close.columns), 150)
    assert b.iloc[:149].isna().all() and b.dropna().between(0, 9).all()
    f = cross_sectional_features(Indicators(close))
    assert set(f) == {"rsi14", "bb_pctb", "dist_sma150", "dist_sma200", "macd_hist", "prox_52w_high"}
    assert (f["prox_52w_high"].dropna() <= 1 + 1e-12).all().all()


def test_technical_study_runs_end_to_end_on_synthetic_markets(tmp_path, monkeypatch):
    import stocksel.experiments.technical_rules as tr
    monkeypatch.setitem(tr.BOOT, "n", 20)
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("1999-01-04", "2014-06-30")
    n = len(idx)

    def walk(cols, vol):
        r = rng.normal(0.0003, vol, size=(n, len(cols)))
        c = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
        o = c.shift(1) * np.exp(rng.normal(0, 0.002, size=c.shape))
        o.iloc[0] = c.iloc[0]
        return o, c

    eo, ec = walk(tr.ETFS, 0.012)
    tick = [f"T{i:02d}" for i in range(36)]
    so, sc = walk(tick, 0.02)
    sc.iloc[:400, :4] = np.nan                     # late listings
    so.iloc[:400, :4] = np.nan
    mcap = sc * rng.uniform(1e9, 5e10, size=len(tick))
    member = pd.DataFrame(True, index=idx, columns=tick)
    member.iloc[:1500, 30:] = False                # joins the index later
    sectors = pd.Series(["Technology"] * 12 + ["Energy"] * 12 + ["Healthcare"] * 12, index=tick)
    inp = tr.Inputs(etf_open=eo, etf_close=ec, stock_open=so, stock_close=sc, stock_mcap=mcap, member=member,
                    sectors=sectors, rf=pd.Series(0.02, index=idx),
                    vix=pd.Series(22 + 12 * np.sin(np.arange(n) / 50), index=idx),
                    prev_alpha_sharpes=np.array([0.01, -0.02, 0.03]), n_previous=3)
    res = tr.run(inp, tmp_path, n_shifts={"etf": 20, "stock": 3})
    T = pd.read_csv(tmp_path / "trials.csv")
    assert len(T) == 16 * (11 + 4) and res["n_trials_charged"] == len(T) + 3
    assert T["disc_months"].min() > 140 and T["conf_months"].min() >= 17
    assert T.loc[T["arm"] == "etf", "shift_p"].between(0, 1).all()
    assert T.loc[T["arm"] == "etf", "dd_shift_p"].between(0, 1).all()
    assert {"alpha_gross", "t_alpha_gross"} <= {c[5:] for c in T.columns if c.startswith("full_")}
    assert T["full_dsr_alpha"].between(0, 1).all()
    # buy-and-hold-like rules on random walks: exposure between 0 and 1, alpha finite
    assert T["full_exposure"].between(0, 1).all() and np.isfinite(T["full_alpha"]).all()
    assert set(res["global"]) == {"etf", "stock"} and len(res["cross_sectional_ic"]) == 4 * 6
    assert (tmp_path / "registry.jsonl").exists() and (tmp_path / "returns.csv").exists()


def test_replication_helpers_and_smoke(tmp_path, monkeypatch):
    import stocksel.experiments.technical_replication as rp
    import stocksel.experiments.technical_rules as tr
    monkeypatch.setitem(tr.BOOT, "n", 20)
    # rotation: equal split capped at 25%, cash otherwise
    idx = pd.bdate_range("2020-01-01", periods=3)
    pos = pd.DataFrame([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], index=idx, columns=list("abcde"), dtype=float)
    sig, w = rp.rotation(pos, pos.notna(), 0.25)
    assert w.iloc[0].tolist() == [0.25, 0, 0, 0, 0] and np.allclose(w.iloc[1], 0.2) and w.iloc[2].sum() == 0
    # Newey-West mean t: a constant positive series with noise is significant, pure noise is not
    rng = np.random.default_rng(2)
    assert rp.mean_t(pd.Series(0.01 + rng.normal(0, 0.01, 300)))["t"] > 5
    assert abs(rp.mean_t(pd.Series(rng.normal(0, 0.01, 300)))["t"]) < 3
    # end-to-end on random walks
    didx = pd.bdate_range("1999-01-04", "2014-06-30")
    cols = rp.INTL[:5]
    r = rng.normal(0.0002, 0.012, size=(len(didx), len(cols)))
    ec = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=didx, columns=cols)
    eo = ec.shift(1) * np.exp(rng.normal(0, 0.002, size=ec.shape))
    eo.iloc[0] = ec.iloc[0]
    spy = pd.Series(rng.normal(0.0003, 0.01, len(didx)), index=didx)
    res = rp.run(eo, ec, pd.Series(0.02, index=didx), spy, tmp_path, eo, ec, n_shifts=20)
    assert set(res["H1_dip_buying"]) >= {"verdict", "BB_MR_TREND", "RSI2"}
    assert set(res["H2_trend_drawdown"]) == {"SMA200", "GOLDEN", "FABER10"}
    assert set(res["H3_rotation"]["RSI2"]) == {"10bps", "20bps", "usable"}
    T = pd.read_csv(tmp_path / "trials.csv")
    assert len(T) == 5 * len(cols) and T["dd_shift_p"].between(0, 1).all()
