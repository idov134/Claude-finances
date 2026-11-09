"""Synthetic market with *known* ground truth.

Why: the research environment may not have market data access, and — more
importantly — a simulator with planted, realistic-size anomalies lets us test
that the pipeline (a) recovers a weak true signal, (b) finds nothing when
``alpha_scale=0`` ("null world"), and (c) is sensitive to deliberate leaks.

What is simulated (daily, business-day calendar):
* market factor with regime-dependent volatility and hard-coded stress windows
  (2000-02 dot-com bust, 2008 GFC, 2020 COVID crash/rebound, 2022 bear);
* 11 sector factors; heterogeneous betas and idiosyncratic volatilities;
* IPOs and delistings (failures with negative delisting returns, mergers with a
  premium) — the full set is retained => survivorship-free by construction;
* planted anomalies of realistic magnitude (cross-sectional IC ~ 0.02-0.05):
  persistent quality, value, momentum drift, low-volatility, 1-month reversal,
  PEAD, analyst revisions that lead earnings surprises, and short-horizon news
  drift (stronger for negative news, decaying after 2012);
* quarterly fundamentals published with a random 25-70 day lag (so
  point-in-time handling matters), earnings announcements with timestamps
  before the open / after the close, monthly analyst consensus, and timestamped
  news articles with full bodies;
* VIX proxy, risk-free rate path, a cap-weighted benchmark index.

Every planted alpha is multiplied by ``alpha_scale`` and by ``post_2005_decay``
after 2005-01-01. None of this says anything about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle

SECTORS = ["Tech", "Health", "Financials", "ConsDisc", "ConsStaples", "Industrials",
           "Energy", "Materials", "Utilities", "RealEstate", "Comm"]

POS_WORDS = ["beat", "strong", "growth", "record", "upgrade", "surge", "profit", "exceeded",
             "robust", "outperform", "expansion", "gain", "improved", "momentum", "raised",
             "breakthrough", "win", "accelerating", "favorable", "optimistic"]
NEG_WORDS = ["miss", "weak", "decline", "loss", "downgrade", "plunge", "lawsuit", "shortfall",
             "impairment", "recall", "investigation", "default", "cut", "warning", "slump",
             "layoffs", "fraud", "deteriorating", "adverse", "bankruptcy"]
NEUTRAL = ["the company", "management", "the quarter", "the board", "investors", "the sector",
           "guidance", "operations", "the market", "analysts", "revenue", "the outlook"]

# (start, end, vol multiplier, extra annual drift of the market)
STRESS_WINDOWS = [
    ("2000-03-10", "2002-10-09", 1.5, -0.18),
    ("2007-10-09", "2008-08-31", 1.4, -0.15),
    ("2008-09-01", "2009-03-09", 3.0, -0.70),
    ("2009-03-10", "2009-12-31", 1.6, 0.45),
    ("2020-02-20", "2020-03-23", 4.5, -3.50),
    ("2020-03-24", "2020-08-31", 2.0, 0.85),
    ("2022-01-03", "2022-10-12", 1.6, -0.30),
]

CAP_POWER = 0.6

# Approximate S&P 500 calendar-year total returns, used only to anchor the *shape* of the
# synthetic market path (crashes and bull runs in the right years). Intra-year paths, all
# stock-level returns and every planted anomaly remain random. 2025 is a placeholder.
SP500_TR = {1998: .286, 1999: .210, 2000: -.091, 2001: -.119, 2002: -.221, 2003: .287, 2004: .109,
            2005: .049, 2006: .158, 2007: .055, 2008: -.370, 2009: .265, 2010: .151, 2011: .021,
            2012: .160, 2013: .324, 2014: .137, 2015: .014, 2016: .120, 2017: .218, 2018: -.044,
            2019: .315, 2020: .184, 2021: .287, 2022: -.181, 2023: .263, 2024: .250, 2025: .10}

RF_PATH = {1998: .05, 1999: .05, 2000: .06, 2001: .035, 2002: .017, 2003: .01, 2004: .014,
           2005: .032, 2006: .049, 2007: .045, 2008: .015, 2009: .002, 2010: .001, 2011: .001,
           2012: .001, 2013: .001, 2014: .001, 2015: .002, 2016: .005, 2017: .01, 2018: .02,
           2019: .021, 2020: .004, 2021: .001, 2022: .02, 2023: .05, 2024: .051, 2025: .043}


def _ar1(rng, n_t, n_s, rho):
    out = np.empty((n_t, n_s))
    out[0] = rng.standard_normal(n_s)
    eps = rng.standard_normal((n_t, n_s)) * np.sqrt(1 - rho**2)
    for t in range(1, n_t):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def _spread(diff: np.ndarray, rows: np.ndarray, cols: np.ndarray, amount: np.ndarray, length: int):
    """Spread ``amount`` evenly over ``length`` rows starting at ``rows`` (difference-array trick)."""
    T = diff.shape[0] - 1
    ok = rows < T
    rows, cols, amount = rows[ok], cols[ok], amount[ok]
    np.add.at(diff, (rows, cols), amount / length)
    np.add.at(diff, (np.minimum(rows + length, T), cols), -amount / length)


def generate(start="1998-01-01", end="2025-12-31", seed=7, n_initial=420, ipo_per_year=22,
             annual_delist_prob=0.035, alpha_scale=1.0, post_2005_decay=0.6, news_per_month=1.2,
             timezone="America/New_York", with_news=True, anchor_market=True,
             interaction: float = 0.0) -> DataBundle:
    """``interaction`` > 0 plants one nonlinear effect for diagnostics only: the momentum premium
    is scaled by (1 + interaction * tanh(-z_idio_vol)) — strong in low-volatility names, weak in
    volatile ones (same average). 0 (default) reproduces the standard world exactly."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    years = dates.year.to_numpy()
    n_years = (dates[-1] - dates[0]).days / 365.25

    # ------------------------------------------------------------ listings
    n_ipo = int(rng.poisson(ipo_per_year * n_years))
    M = n_initial + n_ipo
    tickers = np.array([f"S{i:04d}" for i in range(M)])
    list_day = np.zeros(M, dtype=int)
    list_day[n_initial:] = np.sort(rng.integers(1, T - 30, n_ipo))
    life = rng.exponential(1.0 / annual_delist_prob, M) * 252
    delist_day = (list_day + life).astype(int)
    delist_day[delist_day >= T - 1] = -1
    failure = rng.random(M) < 0.4
    alive = np.zeros((T, M), dtype=bool)
    for i in range(M):
        end_i = delist_day[i] if delist_day[i] >= 0 else T - 1
        alive[list_day[i]:end_i + 1, i] = True

    # ------------------------------------------------------------ market
    vol_mult = np.exp(_ar1(rng, T, 1, 0.985)[:, 0] * 0.18)
    mkt_drift = np.full(T, 0.065)
    for s, e, vm, dr in STRESS_WINDOWS:
        m = (dates >= s) & (dates <= e)
        vol_mult[m] *= vm
        mkt_drift[m] += dr
    rf_ann = np.array([RF_PATH.get(int(y), 0.02) for y in years])
    rf_d = rf_ann / 252
    mkt_ex = mkt_drift / 252 + 0.13 / np.sqrt(252) * vol_mult * rng.standard_normal(T)
    if anchor_market:
        lr = np.log1p(np.clip(mkt_ex + rf_d, -0.5, None))
        for y in np.unique(years):
            m = years == y
            if int(y) in SP500_TR:
                frac = m.sum() / 252.0
                target = np.log1p(SP500_TR[int(y)]) * min(1.0, frac)
                lr[m] += (target - lr[m].sum()) / m.sum()
        mkt_ex = np.expm1(lr) - rf_d

    n_sec = len(SECTORS)
    sector_id = rng.integers(0, n_sec, M)
    sec_ret = 0.10 / np.sqrt(252) * np.sqrt(vol_mult)[:, None] * rng.standard_normal((T, n_sec))

    beta = rng.uniform(0.6, 1.4, M)
    ivol = np.clip(np.exp(rng.normal(np.log(0.21), 0.35, M)), 0.10, 0.75)
    z_ivol = (ivol - ivol.mean()) / ivol.std()

    # ------------------------------------------------------------ planted alpha (annualized)
    month_key = (dates.year - dates.year[0]) * 12 + dates.month - dates.month[0]
    month_id = np.asarray(month_key)
    n_m = month_id.max() + 1
    decay_d = np.where(dates >= pd.Timestamp("2005-01-01"), post_2005_decay, 1.0) * alpha_scale
    q = _ar1(rng, n_m, M, 0.97)            # quality (observable through fundamentals)
    v = _ar1(rng, n_m, M, 0.98)            # value   (observable through B/M, E/P)
    md = _ar1(rng, n_m, M, 0.92)           # momentum drift (observable only through past returns)
    mom_mult = 1.0 + float(interaction) * np.tanh(-z_ivol)[None, :] if interaction else 1.0
    a_month = 0.025 * q + 0.015 * v + 0.18 * md * mom_mult - 0.02 * z_ivol[None, :]
    # (demeaned with last month's cap weights in the price loop below, so the benchmark
    #  index does not mechanically inherit the anomalies)

    # 1-month reversal: a monthly liquidity shock that partially reverts next month
    shock = rng.normal(0, 0.035, (n_m, M))
    days_in_m = np.bincount(month_id)
    rev = np.zeros((n_m, M))
    rev[1:] = -0.35 * shock[:-1]
    alpha_d = rev[month_id] / days_in_m[month_id][:, None] * decay_d[:, None]
    noise_shock = shock[month_id] / days_in_m[month_id][:, None]

    idio = (ivol / np.sqrt(252))[None, :] * (1 + 0.5 * (vol_mult[:, None] - 1)) * rng.standard_normal((T, M))
    core = rf_d[:, None] + beta[None, :] * mkt_ex[:, None] + sec_ret[:, sector_id] + alpha_d + idio + noise_shock

    # failures drift down in their last 6 months (unobservable distress)
    drift_diff = np.zeros((T + 1, M))
    fails = np.where((delist_day >= 0) & failure)[0]
    _spread(drift_diff, np.maximum(delist_day[fails] - 126, list_day[fails]), fails,
            np.full(len(fails), -0.30), 126)

    overnight_jump = np.zeros((T, M))
    intraday_jump = np.zeros((T, M))

    # mergers: +25% premium jump 30 days before delisting
    mergers = np.where((delist_day >= 0) & ~failure)[0]
    mj = np.maximum(delist_day[mergers] - 30, list_day[mergers] + 1)
    overnight_jump[mj, mergers] += 0.25

    # ------------------------------------------------------------ earnings / fundamentals
    q_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    ev_day, ev_stock, ev_q = [], [], []
    for qn, qe in enumerate(q_ends):
        lag = rng.integers(25, 71, M)
        pub_idx = dates.searchsorted(qe + pd.to_timedelta(lag, unit="D"))
        qe_idx = min(dates.searchsorted(qe), T - 1)
        ok = (pub_idx < T - 1) & alive[qe_idx] & alive[np.minimum(pub_idx, T - 1), np.arange(M)]
        idx = np.where(ok)[0]
        ev_day.append(pub_idx[idx]); ev_stock.append(idx); ev_q.append(np.full(len(idx), qn))
    ev_day = np.concatenate(ev_day); ev_stock = np.concatenate(ev_stock); ev_q = np.concatenate(ev_q)
    n_ev = len(ev_day)
    ev_e = rng.standard_normal(n_ev)
    ev_after = rng.random(n_ev) < 0.6
    jump_day = np.minimum(np.where(ev_after, ev_day + 1, ev_day), T - 1)
    np.add.at(overnight_jump, (jump_day, ev_stock), 0.04 * ev_e)
    pead_amt = 0.015 * ev_e * decay_d[jump_day]
    _spread(drift_diff, jump_day, ev_stock, pead_amt, 60)

    # news drift will be added below; assemble returns after news
    news_df = None
    if with_news:
        news_decay = np.clip(1 - np.maximum(0, years - 2012) / 12.0, 0.2, 1.0)
        tz_dates = dates.tz_localize(timezone)
        alive_count = alive.sum(axis=0)
        # regular articles
        art_stock, art_day, art_tone, art_hour = [], [], [], []
        for i in range(M):
            days_alive = np.where(alive[:, i])[0]
            if len(days_alive) == 0:
                continue
            n = rng.poisson(news_per_month * alive_count[i] / 21)
            if n == 0:
                continue
            art_day.append(rng.choice(days_alive, n))
            art_stock.append(np.full(n, i))
            art_tone.append(rng.standard_normal(n))
            art_hour.append(rng.uniform(0, 24, n))
        # earnings articles (tone ~ surprise)
        art_day.append(ev_day); art_stock.append(ev_stock)
        art_tone.append(ev_e + 0.5 * rng.standard_normal(len(ev_e)))
        art_hour.append(np.where(ev_after, 16.6, 7.5))
        art_day = np.concatenate(art_day); art_stock = np.concatenate(art_stock)
        art_tone = np.concatenate(art_tone); art_hour = np.concatenate(art_hour)
        # first trading day whose close (16:00) is after the article
        eff_day = np.where(art_hour < 16.0, art_day, art_day + 1)
        imm = 0.004 * art_tone
        in_session = eff_day == art_day
        valid = eff_day < T
        intraday_jump_idx = valid & in_session
        np.add.at(intraday_jump, (eff_day[intraday_jump_idx], art_stock[intraday_jump_idx]), imm[intraday_jump_idx])
        over_idx = valid & ~in_session
        np.add.at(overnight_jump, (eff_day[over_idx], art_stock[over_idx]), imm[over_idx])
        tone_eff = np.where(art_tone < 0, 1.6 * art_tone, art_tone)
        drift_amt = 0.005 * tone_eff * news_decay[np.minimum(eff_day, T - 1)] * decay_d[np.minimum(eff_day, T - 1)]
        _spread(drift_diff, eff_day + 1, art_stock, drift_amt, 3)

        # texts (vectorised word draws; full body carries more signal than the headline)
        n_art = len(art_day)
        ts = (tz_dates[np.minimum(art_day, T - 1)] + pd.to_timedelta(art_hour, unit="h")).round("s")
        n_words = rng.poisson(1.0 + 2.0 * np.abs(art_tone)) + 1
        p_pos = np.repeat(1 / (1 + np.exp(-2.0 * art_tone)), n_words)
        is_pos = rng.random(len(p_pos)) < p_pos
        widx = rng.integers(0, len(POS_WORDS), len(p_pos))
        words = np.where(is_pos, np.array(POS_WORDS)[widx], np.array(NEG_WORDS)[widx])
        fills = np.array(NEUTRAL)[rng.integers(0, len(NEUTRAL), (n_art, 3))]
        bounds = np.r_[0, np.cumsum(n_words)]
        head_has = rng.random(n_art) < 0.4
        tick = tickers[art_stock]
        bodies, heads = [], []
        for k in range(n_art):
            w = words[bounds[k]:bounds[k + 1]]
            bodies.append(f"{tick[k]}: {fills[k, 0]} commented on {fills[k, 1]}. "
                          + " ".join(f"Reports cite {x} results." for x in w)
                          + f" {fills[k, 2].capitalize()} will be watched.")
            heads.append(f"{tick[k]} {w[0]} update" if head_has[k] else f"{tick[k]} company update")
        news_df = pd.DataFrame({"ticker": tick, "timestamp": ts.tz_convert("UTC"),
                                "headline": heads, "body": bodies})
        news_df = news_df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    drift = np.cumsum(drift_diff[:-1], axis=0)
    ret = core + drift + overnight_jump + intraday_jump

    # ------------------------------------------------------------ prices (month by month)
    p0 = rng.uniform(15, 120, M)
    mcap0 = np.exp(rng.normal(np.log(8e9), 1.0, M))
    mcap0[n_initial:] = np.exp(rng.normal(np.log(3.5e9), 0.8, n_ipo))
    shares = mcap0 / p0
    logp = np.zeros((T, M))
    persist = np.zeros((T, M))
    base = np.full(M, np.nan)
    run = np.zeros(M)
    starts = np.r_[0, np.where(np.diff(month_id))[0] + 1]
    ends = np.r_[starts[1:], T]
    for m, (s0, e0) in enumerate(zip(starts, ends)):
        if s0 == 0:
            w = mcap0 ** CAP_POWER * alive[0]
        else:
            px = p0 * np.exp(logp[s0 - 1] - base)
            w = np.where(alive[s0 - 1] & np.isfinite(px), (px * shares) ** CAP_POWER, 0.0)
        al = alive[s0:e0].any(axis=0)
        a = a_month[m]
        wa = w * al
        a_dm = a - (wa @ a) / wa.sum() if wa.sum() > 0 else a - a[al].mean()
        pa = (a_dm / 252)[None, :] * decay_d[s0:e0][:, None]
        persist[s0:e0] = pa
        r = np.clip(ret[s0:e0] + pa, -0.9, 3.0)
        r[~alive[s0:e0]] = np.nan
        blk = run + np.cumsum(np.where(np.isnan(r), 0.0, np.log1p(r)), axis=0)
        logp[s0:e0] = blk
        run = blk[-1]
        new = (list_day >= s0) & (list_day < e0)
        base[new] = logp[list_day[new], np.where(new)[0]]
        ret[s0:e0] = r
    close = p0[None, :] * np.exp(logp - base[None, :])
    close[~alive] = np.nan
    core = core + persist
    prev_close = np.vstack([np.full(M, np.nan), close[:-1]])
    on_ret = 0.35 * (core - rf_d[:, None]) + overnight_jump + rng.normal(0, 0.003, (T, M))
    open_ = prev_close * (1 + on_ret)
    first = list_day
    open_[first, np.arange(M)] = close[first, np.arange(M)] / (1 + ret[first, np.arange(M)] * 0.5)
    open_[~alive] = np.nan
    dvol = (ivol / np.sqrt(252))[None, :] * (1 + 0.5 * (vol_mult[:, None] - 1))
    hi = np.fmax(open_, close) * (1 + np.abs(rng.normal(0, 0.5, (T, M))) * dvol)
    lo = np.fmin(open_, close) * (1 - np.abs(rng.normal(0, 0.5, (T, M))) * dvol)

    raw_cap = close * shares[None, :]
    # float/free-cap compression: keeps index concentration realistic over 28 years of
    # random-walk prices (log-cap dispersion of a lognormal walk grows without bound)
    med = np.nanmedian(raw_cap, axis=1, keepdims=True)
    mcap = raw_cap ** CAP_POWER * med ** (1 - CAP_POWER)
    turnover = np.exp(rng.normal(np.log(0.006), 0.4, M))
    event_boost = 1 + 4 * (np.abs(overnight_jump) > 0.01)
    dollar_vol = mcap * turnover[None, :] * np.exp(rng.normal(0, 0.3, (T, M))) * event_boost * np.sqrt(vol_mult)[:, None]
    volume = dollar_vol / close

    # delisting returns: failures lose 30-90% on the delisting, mergers 0
    dr = pd.Series(dtype=float)
    dl = np.where(delist_day >= 0)[0]
    if len(dl):
        vals = np.where(failure[dl], rng.uniform(-0.9, -0.3, len(dl)), 0.0)
        dr = pd.Series(vals, index=tickers[dl])

    # ------------------------------------------------------------ benchmark (cap-weighted, incl. delisting)
    r_full = np.where(np.isnan(ret), 0.0, ret)
    for i in dl:
        d = delist_day[i] + 1
        if d < T:
            r_full[d, i] = dr[tickers[i]]
    w_prev = np.vstack([np.zeros(M), np.nan_to_num(mcap[:-1])])
    w_sum = w_prev.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1
    bench_ret = (w_prev / w_sum * r_full).sum(axis=1)
    bench_ret[0] = 0
    bench_close = 100 * np.cumprod(1 + bench_ret)
    on_full = np.where(np.isnan(open_ / prev_close), 0.0, open_ / prev_close - 1)
    bench_on = (w_prev / w_sum * on_full).sum(axis=1)
    bench_open = np.r_[100.0, bench_close[:-1] * (1 + bench_on[1:])]

    # ------------------------------------------------------------ VIX proxy
    rv = pd.Series(bench_ret, index=dates).rolling(21, min_periods=5).std() * np.sqrt(252) * 100
    vix = (rv.bfill() * 1.1 + 4 + pd.Series(_ar1(rng, T, 1, 0.95)[:, 0] * 1.5, index=dates)).clip(lower=9)

    # ------------------------------------------------------------ membership (top 500 by cap, monthly)
    mcap_df = pd.DataFrame(mcap, index=dates, columns=tickers)
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max().values
    rank_me = mcap_df.loc[me].rank(axis=1, ascending=False)
    memb_me = (rank_me <= 500)
    membership = memb_me.reindex(dates).shift(1).ffill().fillna(False).astype(bool) & mcap_df.notna()

    # ------------------------------------------------------------ fundamentals / earnings / estimates
    pe_dates = q_ends[ev_q]
    pe_idx = np.minimum(dates.searchsorted(pe_dates), T - 1)
    m_ev = month_id[pe_idx]
    mc = mcap[pe_idx, ev_stock]
    okf = np.isfinite(mc)
    z = rng.standard_normal((n_ev, 5))
    ta = mc * 0.8 * np.exp(0.3 * z[:, 0])
    bm = np.exp(-0.7 + 0.5 * v[m_ev, ev_stock] + 0.25 * z[:, 1])
    be = mc * bm
    gp = ta * (0.07 + 0.025 * q[m_ev, ev_stock] + 0.01 * z[:, 2])
    ni = be * (0.025 + 0.01 * q[m_ev, ev_stock] + 0.006 * v[m_ev, ev_stock] + 0.01 * z[:, 3])
    rev_ = ta * np.maximum(0.05, 0.25 + 0.05 * z[:, 4])
    eps = ni / shares[ev_stock]
    pub = dates[ev_day]
    fundamentals = pd.DataFrame({"ticker": tickers[ev_stock], "period_end": pe_dates, "publish_date": pub,
                                 "total_assets": ta, "book_equity": be, "gross_profit": gp,
                                 "net_income": ni, "revenue": rev_, "eps": eps})[okf].reset_index(drop=True)
    hours = np.where(ev_after, 16.5, 7.5)
    ann_ts = (pub + pd.to_timedelta(hours, unit="h")).tz_localize(timezone).tz_convert("UTC")
    px = close[ev_day, ev_stock]
    cons = eps - 0.15 * np.abs(eps) * ev_e - 0.0002 * px * ev_e
    earnings = pd.DataFrame({"ticker": tickers[ev_stock], "announce_ts": ann_ts,
                             "eps_actual": eps, "eps_consensus": cons})[okf].reset_index(drop=True)

    # analyst consensus: the 3 month-ends before an announcement revise toward the coming surprise
    me_idx = dates.searchsorted(me)
    rev_inc = np.zeros((len(me), M))
    c = np.minimum(0.9, 0.25 * decay_d[ev_day])
    pos = np.searchsorted(me_idx, ev_day) - 1
    for lag_m in range(3):
        j = pos - lag_m
        okj = j >= 0
        np.add.at(rev_inc, (j[okj], ev_stock[okj]),
                  0.01 * (c[okj] * ev_e[okj] + np.sqrt(1 - c[okj] ** 2) * rng.standard_normal(okj.sum())))
    fy1 = 2.0 * np.exp(np.cumsum(rev_inc, axis=0))
    est = pd.DataFrame(fy1, index=pd.DatetimeIndex(me), columns=tickers)
    alive_me = pd.DataFrame(alive[me_idx], index=pd.DatetimeIndex(me), columns=tickers)
    est = est.where(alive_me).stack().rename("fy1_consensus").reset_index()
    est.columns = ["date", "ticker", "fy1_consensus"]

    news = news_df

    wide = lambda a: pd.DataFrame(a, index=dates, columns=tickers)
    bundle = DataBundle(
        open=wide(open_), high=wide(hi), low=wide(lo), close=wide(close), volume=wide(volume),
        market_cap=mcap_df, sectors=pd.Series(np.array(SECTORS)[sector_id], index=tickers),
        benchmark=pd.DataFrame({"open": bench_open, "close": bench_close}, index=dates),
        membership=membership, delist_returns=dr, fundamentals=fundamentals, earnings=earnings,
        estimates=est, news=news, vix=vix.rename("vix"),
        risk_free=pd.Series(rf_ann, index=dates, name="rf"),
        meta={"source": "synthetic", "survivorship_free": True, "pit_fundamentals": True,
              "sector_pit": True, "timezone": timezone, "alpha_scale": alpha_scale, "seed": seed,
              "warning": "SYNTHETIC DATA — results say nothing about real markets."},
    )
    bundle.validate()
    return bundle


def from_config(cfg) -> DataBundle:
    s = cfg.data.synthetic
    return generate(start=cfg.data.start, end=cfg.data.end, seed=s.seed, n_initial=s.n_initial,
                    ipo_per_year=s.ipo_per_year, annual_delist_prob=s.annual_delist_prob,
                    alpha_scale=s.alpha_scale, post_2005_decay=s.post_2005_decay,
                    news_per_month=s.news_per_month, timezone=cfg.market.timezone)
