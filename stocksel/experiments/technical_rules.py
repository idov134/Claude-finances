"""Simple technical rules by market and segment. Pre-registered in docs/technical_rules/PROTOCOL.md.

    python -m stocksel.data.build_etf_csv --data data_csv     # ETF prices + Yahoo sectors (local only)
    python -m stocksel.experiments.technical_rules           # run once

Two arms, each rule compared with buy-and-hold of the same market:
* ETF arm (primary, no survivorship bias): SPY, QQQ and the nine sector SPDRs;
* stock arm (secondary, survivor-only data): equal-weight baskets of S&P 500 members (all, the 100
  largest, technology, energy), each stock traded by the rule.
Discovery 2000-04 to 2012-12, confirmation 2013-01 to the end of the data.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.rules import monthly, open_to_open, shift_stats, simulate
from ..features.technical import (FAMILY, RULES, Indicators, Market, cross_sectional_features, month_end_mask,
                                  rule_positions, sector_breadth)
from ..validation.metrics import (alpha_returns, alpha_tstat, cagr, deflated_sharpe, max_drawdown,
                                  performance_table)
from ..validation.trials import TrialRegistry

PROTOCOL = "technical_rules_v1"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "technical_rules"
START, SPLIT_END, CONF_START = "2000-04-03", "2012-12-31", "2013-01-01"
SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
ETFS = ["SPY", "QQQ"] + SECTOR_ETFS
SEGMENTS = {"ALL": None, "TOP100": None, "TECH": "Technology", "ENERGY": "Energy"}
FOCUS = {"etf": ["SPY", "QQQ", "XLK", "XLE"], "stock": ["ALL", "TOP100", "TECH", "ENERGY"]}
COST_BPS = {"etf": 3.0, "stock": 10.0}
MIN_HISTORY = 300
TOP_N = 100
RULE = {"t": 2.0, "dsr": 0.95, "shift_p": 0.05, "dd_shift_p": 0.01, "dd_ratio": 2.0 / 3.0}
SAME_AS_BH = 1e-9          # monthly tracking error below this: the rule did nothing different from buy-and-hold
BOOT = {"n": 1000, "block": 6, "seed": 3}
N_SHIFTS = {"etf": 500, "stock": 200}
FEATURE_SIGN = {"rsi14": -1, "bb_pctb": -1, "dist_sma150": 1, "dist_sma200": 1, "macd_hist": 1, "prox_52w_high": 1}


def log(msg: str) -> None:
    print(f"[technical {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Inputs:
    etf_open: pd.DataFrame
    etf_close: pd.DataFrame
    stock_open: pd.DataFrame
    stock_close: pd.DataFrame
    stock_mcap: pd.DataFrame
    member: pd.DataFrame                 # bool, point-in-time index membership
    sectors: pd.Series                   # ticker -> Yahoo sector (current classification)
    rf: pd.Series                        # annualized T-bill rate
    vix: pd.Series
    sector_etfs: list[str] = field(default_factory=lambda: list(SECTOR_ETFS))
    prev_alpha_sharpes: np.ndarray = field(default_factory=lambda: np.array([]))
    n_previous: int = 0
    coverage: pd.DataFrame | None = None


# --------------------------------------------------------------------------- universes
def stock_universes(inp: Inputs) -> dict[str, pd.DataFrame]:
    c = inp.stock_close
    hist = c.notna().cumsum()
    base = inp.member.reindex_like(c).fillna(False).astype(bool) & c.notna() & (hist >= MIN_HISTORY)
    sec = inp.sectors.reindex(c.columns).fillna("Unknown")
    out = {"ALL": base}
    me = month_end_mask(c.index)
    caps = inp.stock_mcap.reindex_like(c).where(base)[me.to_numpy()]
    top = caps.rank(axis=1, ascending=False, method="first") <= TOP_N
    top = top.astype(float).reindex(c.index).ffill().fillna(0.0).astype(bool)
    out["TOP100"] = base & top
    for name, label in SEGMENTS.items():
        if label:
            out[name] = base & pd.DataFrame(np.repeat((sec == label).to_numpy()[None, :], len(c), axis=0),
                                            index=c.index, columns=c.columns)
    return out


# --------------------------------------------------------------------------- evaluation
def window_metrics(s: pd.DataFrame, b: pd.DataFrame, rf_d: pd.Series, spy: pd.Series, lo: str, hi: str,
                   n_assets: pd.Series | None = None) -> dict:
    s, b = s.loc[lo:hi], b.loc[lo:hi]
    rfw = rf_d.reindex(s.index).ffill().fillna(0.0)
    sm, bm, rfm = monthly(s["ret"]), monthly(b["ret"]), monthly(rfw)
    p = performance_table(sm, bm, rfm, boot=BOOT)
    a = alpha_tstat(sm - rfm, bm - rfm)
    ag = alpha_tstat(monthly(s["ret"] + s["cost"]) - rfm, bm - rfm)      # before costs
    if (sm - bm).std() < SAME_AS_BH:      # identical to buy-and-hold: alpha is rounding noise, not a result
        a = {**a, "alpha": 0.0, "t_alpha": 0.0}
        ag = {**ag, "alpha": 0.0, "t_alpha": 0.0}
    yrs = len(s) / 252.0
    per_asset = float(n_assets.reindex(s.index).mean()) if n_assets is not None else 1.0
    return {"months": p["n_periods"], "cagr": p["cagr"], "bh_cagr": p["bench_cagr"],
            "spy_cagr": cagr(monthly(spy.reindex(s.index).fillna(0.0))), "sharpe": p["sharpe"],
            "bh_sharpe": p["bench_sharpe"], "vol": p["ann_vol"], "bh_vol": p["bench_vol"],
            "max_dd": max_drawdown(s["ret"]), "bh_max_dd": max_drawdown(b["ret"]), "alpha": a["alpha"],
            "beta": a["beta"], "t_alpha": a["t_alpha"], "alpha_gross": ag["alpha"], "t_alpha_gross": ag["t_alpha"],
            "excess_vs_bh": p["excess_ann"],
            "exposure": float(s["exposure"].mean()), "turnover_yr": float(s["turnover"].sum() / yrs),
            "cost_drag": float(s["cost"].sum() / yrs),
            "entries_per_asset_yr": float(s["entries"].sum() / yrs / max(per_asset, 1.0))}


def monthly_alpha(s: pd.DataFrame, b: pd.DataFrame, rf_d: pd.Series) -> pd.Series:
    rfm = monthly(rf_d.reindex(s.index).ffill().fillna(0.0))
    return alpha_returns(monthly(s["ret"]) - rfm, monthly(b["ret"]) - rfm)


def drop_top_days_ok(s: pd.DataFrame, b: pd.DataFrame, rf_d: pd.Series, beta: float, frac: float = 0.001) -> bool:
    """Brief sanity check: removing the best 0.1% of days must not erase the edge."""
    rf = rf_d.reindex(s.index).ffill().fillna(0.0)
    act = (s["ret"] - rf) - beta * (b["ret"] - rf)
    k = max(1, math.ceil(frac * len(act)))
    return bool(act.drop(act.nlargest(k).index).mean() > 0)


def _month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    m = index.to_period("M")
    return np.flatnonzero(np.r_[True, m[1:] != m[:-1]])


def etf_shift_p(pos: pd.Series, r: pd.Series, rf_d: pd.Series, bh: pd.DataFrame, lo: str, hi: str, cost_bps: float,
                n: int, seed: int) -> tuple[float, float]:
    """(p-value of alpha, p-value of max drawdown) against n random circular shifts of the rule's positions."""
    idx = bh.loc[lo:hi].index
    pe = pos.shift(1).fillna(0.0).reindex(idx).fillna(0.0).to_numpy()
    rr = r.reindex(idx).fillna(0.0).to_numpy()
    rf = rf_d.reindex(idx).ffill().fillna(0.0).to_numpy()
    rfm = monthly(pd.Series(rf, index=idx))
    bm_ex = (monthly(bh.loc[lo:hi, "ret"]) - rfm).to_numpy()
    rng = np.random.default_rng(seed)
    ks = np.r_[0, rng.integers(21, len(idx) - 21, size=n)]
    al, dd = shift_stats(pe, rr, rf, bm_ex, _month_starts(idx), cost_bps / 1e4, ks)
    return float((1 + np.sum(al[1:] >= al[0])) / (n + 1)), float((1 + np.sum(dd[1:] >= dd[0])) / (n + 1))


def stock_shift_p(pos: pd.DataFrame, w: pd.DataFrame, r: pd.DataFrame, rf_d: pd.Series, bh: pd.DataFrame, lo: str,
                  hi: str, cost_bps: float, n: int, seed: int) -> tuple[float, float]:
    """Basket version of ``etf_shift_p``: every stock's positions are shifted by the same k."""
    idx = pos.loc[lo:hi].index
    P = pos.loc[lo:hi].fillna(0.0).to_numpy()
    ww, rr = w.loc[lo:hi], r.loc[lo:hi]
    b = bh.loc[lo:hi]
    bm_ex = monthly(b["ret"]) - monthly(rf_d.reindex(b.index).ffill().fillna(0.0))
    rng = np.random.default_rng(seed)
    al, dd = [], []
    for k in np.r_[0, rng.integers(21, len(idx) - 21, size=n)]:
        sig = pd.DataFrame(np.roll(P, int(k), axis=0), index=idx, columns=pos.columns)
        s = simulate(sig, ww, rr, rf_d, cost_bps)
        sm = monthly(s["ret"]) - monthly(rf_d.reindex(s.index).ffill().fillna(0.0))
        al.append(alpha_tstat(sm, bm_ex.reindex(sm.index))["alpha"])
        dd.append(max_drawdown(s["ret"]))
    al, dd = np.array(al), np.array(dd)
    return float((1 + np.sum(al[1:] >= al[0])) / (n + 1)), float((1 + np.sum(dd[1:] >= dd[0])) / (n + 1))


def rowwise_spearman(f: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, min_n: int = 10) -> pd.Series:
    ok = mask & f.notna() & y.notna()
    a, b = f.where(ok).rank(axis=1), y.where(ok).rank(axis=1)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    r = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return r.where(ok.sum(axis=1) >= min_n)


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


# --------------------------------------------------------------------------- study
def run(inp: Inputs, out: Path, n_shifts: dict | None = None, seed: int = 11) -> dict:
    t0 = time.time()
    n_shifts = n_shifts or N_SHIFTS
    out.mkdir(parents=True, exist_ok=True)
    reg = TrialRegistry(out, stage=PROTOCOL)
    rf_all = inp.rf.sort_index()
    breadth = sector_breadth(inp.etf_close, inp.sector_etfs, 150)

    # ---------------- ETF arm
    ec, eo = inp.etf_close, inp.etf_open
    rf_e = (rf_all.reindex(ec.index).ffill() / 252.0).fillna(0.0)
    mkt_e = Market(vix=inp.vix.reindex(ec.index).ffill(), breadth=breadth.reindex(ec.index))
    r_e = open_to_open(eo)
    one = ec.notna().astype(float).where(ec.notna())
    bh_e = {m: simulate(one[[m]], one[[m]].fillna(0), r_e[[m]], rf_e, COST_BPS["etf"]) for m in ec.columns}
    spy = bh_e["SPY"]["ret"] if "SPY" in bh_e else next(iter(bh_e.values()))["ret"]
    end_e = str(bh_e[ec.columns[0]].index[-1].date())
    windows = {"etf": {"disc": (START, SPLIT_END), "conf": (CONF_START, end_e), "full": (START, end_e)}}
    rows, alpha_cols, extra = [], {}, {}
    ind_e = Indicators(ec)
    for rule in RULES:
        pos = rule_positions(rule, ind_e, mkt_e)
        for m in ec.columns:
            s = simulate(pos[[m]], one[[m]].fillna(0), r_e[[m]], rf_e, COST_BPS["etf"])
            row = {"arm": "etf", "market": m, "rule": rule, "family": FAMILY[rule]}
            for w, (lo, hi) in windows["etf"].items():
                row.update({f"{w}_{k}": v for k, v in window_metrics(s, bh_e[m], rf_e, spy, lo, hi).items()})
            lo, hi = windows["etf"]["full"]
            sf, bf = s.loc[lo:hi], bh_e[m].loc[lo:hi]
            alpha_cols[f"etf|{m}|{rule}"] = monthly_alpha(sf, bf, rf_e)
            row["shift_p"], row["dd_shift_p"] = etf_shift_p(pos[m], r_e[m], rf_e, bh_e[m], lo, hi, COST_BPS["etf"],
                                                            n_shifts["etf"], seed)
            extra[(m, rule)] = (s, pos)
            conf = {"protocol": PROTOCOL, "arm": "etf", "market": m, "rule": rule, "cost_bps": COST_BPS["etf"]}
            rfm = monthly(rf_e.reindex(sf.index).ffill().fillna(0.0))
            reg.log(conf, (monthly(sf["ret"]) - rfm).rename(None),
                    {k: v for k, v in row.items() if k.startswith("full_")})
            rows.append(row)
        log(f"etf arm: {rule}")

    # ---------------- stock arm
    sc, so = inp.stock_close, inp.stock_open
    rf_s = (rf_all.reindex(sc.index).ffill() / 252.0).fillna(0.0)
    mkt_s = Market(vix=inp.vix.reindex(sc.index).ffill(), breadth=breadth.reindex(sc.index))
    r_s = open_to_open(so)
    unis = stock_universes(inp)
    weights = {g: u.astype(float).div(u.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) for g, u in unis.items()}
    sizes = {g: u.sum(axis=1).astype(float) for g, u in unis.items()}
    bh_s = {g: simulate(unis[g].astype(float), weights[g], r_s, rf_s, COST_BPS["stock"]) for g in unis}
    end_s = str(bh_s["ALL"].index[-1].date())
    windows["stock"] = {"disc": (START, SPLIT_END), "conf": (CONF_START, end_s), "full": (START, end_s)}
    ind_s = Indicators(sc)
    stock_pos = {}
    for rule in RULES:
        pos = rule_positions(rule, ind_s, mkt_s)
        stock_pos[rule] = pos
        for g in unis:
            s = simulate(pos, weights[g], r_s, rf_s, COST_BPS["stock"])
            row = {"arm": "stock", "market": g, "rule": rule, "family": FAMILY[rule]}
            for w, (lo, hi) in windows["stock"].items():
                row.update({f"{w}_{k}": v for k, v in
                            window_metrics(s, bh_s[g], rf_s, spy, lo, hi, sizes[g]).items()})
            lo, hi = windows["stock"]["full"]
            sf, bf = s.loc[lo:hi], bh_s[g].loc[lo:hi]
            alpha_cols[f"stock|{g}|{rule}"] = monthly_alpha(sf, bf, rf_s)
            row["shift_p"], row["dd_shift_p"] = np.nan, np.nan
            extra[(g, rule)] = (s, pos)
            conf = {"protocol": PROTOCOL, "arm": "stock", "market": g, "rule": rule, "cost_bps": COST_BPS["stock"]}
            rfm = monthly(rf_s.reindex(sf.index).ffill().fillna(0.0))
            reg.log(conf, (monthly(sf["ret"]) - rfm).rename(None),
                    {k: v for k, v in row.items() if k.startswith("full_")})
            rows.append(row)
        log(f"stock arm: {rule}")
    T = pd.DataFrame(rows)

    # ---------------- multiple testing: alpha DSR over every real-data trial
    AR = pd.DataFrame({k: v.set_axis(v.index.to_period("M")) for k, v in alpha_cols.items()})
    sd = AR.std(ddof=1)
    sr_this = (AR.mean() / sd).where(sd > SAME_AS_BH).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sr_all = np.r_[sr_this, np.asarray(inp.prev_alpha_sharpes, float)]
    n_total = int(inp.n_previous + len(T))
    def _dsr(x: pd.Series) -> float:
        x = x.dropna()
        return deflated_sharpe(x.to_numpy(), n_total, sr_all)["dsr"] if x.std() > SAME_AS_BH else np.nan
    T["full_dsr_alpha"] = [_dsr(AR[f"{a}|{m}|{r}"]) for a, m, r in zip(T["arm"], T["market"], T["rule"])]

    # ---------------- decisions
    T["candidate"] = (T["disc_t_alpha"] >= RULE["t"]) & (T["conf_t_alpha"] >= RULE["t"])
    T["risk_pre"] = ((T["disc_max_dd"] >= RULE["dd_ratio"] * T["disc_bh_max_dd"])
                     & (T["conf_max_dd"] >= RULE["dd_ratio"] * T["conf_bh_max_dd"])
                     & (T["disc_sharpe"] >= T["disc_bh_sharpe"]) & (T["conf_sharpe"] >= T["conf_bh_sharpe"]))
    T["full_alpha_2x_cost"] = np.nan
    T["drop_top_ok"] = pd.Series([None] * len(T), index=T.index, dtype=object)
    checks = {}
    for i in T.index[T["candidate"] | T["risk_pre"]]:
        a, m, rule = T.at[i, "arm"], T.at[i, "market"], T.at[i, "rule"]
        lo, hi = windows[a]["full"]
        s, pos = extra[(m, rule)]
        if a == "stock":
            T.at[i, "shift_p"], T.at[i, "dd_shift_p"] = stock_shift_p(pos, weights[m], r_s, rf_s, bh_s[m], lo, hi,
                                                                      COST_BPS["stock"], n_shifts["stock"], seed)
        if not T.at[i, "candidate"]:
            continue
        if a == "etf":
            s2 = simulate(pos[[m]], one[[m]].fillna(0), r_e[[m]], rf_e, 2 * COST_BPS["etf"])
            b, rfd = bh_e[m], rf_e
        else:
            s2 = simulate(pos, weights[m], r_s, rf_s, 2 * COST_BPS["stock"])
            b, rfd = bh_s[m], rf_s
        a2 = window_metrics(s2, b, rfd, spy, lo, hi)["alpha"]
        top_ok = drop_top_days_ok(s.loc[lo:hi], b.loc[lo:hi], rfd, T.at[i, "full_beta"])
        checks[f"{a}|{m}|{rule}"] = {"alpha_2x_cost": a2, "shift_p": T.at[i, "shift_p"], "drop_top_0.1pct_ok": top_ok}
        T.at[i, "full_alpha_2x_cost"] = a2
        T.at[i, "drop_top_ok"] = top_ok
    T["edge_confirmed"] = (T["candidate"] & (T["full_dsr_alpha"] >= RULE["dsr"]) & (T["full_alpha_2x_cost"] > 0)
                           & (T["shift_p"] <= RULE["shift_p"]) & (T["drop_top_ok"] == True))  # noqa: E712
    T["risk_control"] = T["risk_pre"] & (T["dd_shift_p"] <= RULE["dd_shift_p"])

    # ---------------- global persistence and best-in-discovery
    from scipy.stats import spearmanr
    glob = {}
    for a in ("etf", "stock"):
        t = T[T["arm"] == a]
        rho = spearmanr(t["disc_alpha_gross"], t["conf_alpha_gross"], nan_policy="omit")
        rho_net = spearmanr(t["disc_alpha"], t["conf_alpha"], nan_policy="omit")
        glob[a] = {"trials": int(len(t)), "spearman_disc_conf_alpha_gross": float(rho.statistic),
                   "spearman_p": float(rho.pvalue), "spearman_net_for_reference": float(rho_net.statistic),
                   "t_ge_2_disc": int((t["disc_t_alpha"] >= 2).sum()),
                   "t_ge_2_conf": int((t["conf_t_alpha"] >= 2).sum()), "t_ge_2_both": int(t["candidate"].sum()),
                   "expected_by_chance_per_half": round(0.0228 * len(t), 1),
                   "alpha_positive_disc": int((t["disc_alpha"] > 0).sum()),
                   "alpha_positive_conf": int((t["conf_alpha"] > 0).sum()),
                   "edge_confirmed": int(t["edge_confirmed"].sum()), "risk_pre": int(t["risk_pre"].sum()),
                   "risk_control": int(t["risk_control"].sum())}
    best = {}
    for a, markets in FOCUS.items():
        for m in markets:
            t = T[(T["arm"] == a) & (T["market"] == m)].copy()
            if t.empty:
                continue
            pick = t.loc[t["disc_alpha"].idxmax()]
            t["conf_rank"] = t["conf_alpha"].rank(ascending=False)
            best[f"{a}|{m}"] = {"rule": pick["rule"], "disc_alpha": pick["disc_alpha"],
                                "disc_t": pick["disc_t_alpha"], "conf_alpha": pick["conf_alpha"],
                                "conf_t": pick["conf_t_alpha"],
                                "conf_rank_of_16": float(t.loc[pick.name, "conf_rank"])}

    # ---------------- cross-sectional technical IC by segment (monthly, next-open to next-open)
    me = month_end_mask(sc.index)
    on = so.shift(-1)[me.to_numpy()]                       # open of the day after each month end
    fwd = on.shift(-1) / on - 1
    feats = cross_sectional_features(ind_s)
    ic = {}
    for g, u in unis.items():
        mask = u[me.to_numpy()]
        for fname, f in feats.items():
            s_ic = rowwise_spearman(f[me.to_numpy()], fwd, mask)
            d, c = s_ic.loc[START:SPLIT_END], s_ic.loc[CONF_START:]
            ic[f"{g}|{fname}"] = {"expected_sign": FEATURE_SIGN[fname], "disc_ic": float(d.mean()),
                                  "disc_t": _t(d), "conf_ic": float(c.mean()), "conf_t": _t(c),
                                  "promising": bool(abs(_t(d)) >= 2 and abs(_t(c)) >= 2
                                                    and np.sign(d.mean()) == np.sign(c.mean()))}

    # ---------------- segment buy-and-hold vs SPY (descriptive)
    seg = {}
    for a, src in (("etf", bh_e), ("stock", bh_s)):
        for m, b in src.items():
            for w, (lo, hi) in windows[a].items():
                bb = b.loc[lo:hi, "ret"]
                sp = spy.reindex(bb.index).fillna(0.0)
                seg[f"{a}|{m}|{w}"] = {"cagr": cagr(monthly(bb)), "spy_cagr": cagr(monthly(sp)),
                                       "max_dd": max_drawdown(bb), "spy_max_dd": max_drawdown(sp)}
    sizes_summary = {g: {w: float(sizes[g].loc[lo:hi].mean()) for w, (lo, hi) in windows["stock"].items()}
                     for g in sizes}
    cov = {}
    if inp.coverage is not None and len(inp.coverage):
        cv = inp.coverage.set_index(pd.to_datetime(inp.coverage["date"]))
        ratio = cv["with_prices"] / cv["members"]
        cov = {"disc": float(ratio.loc[START:SPLIT_END].mean()), "conf": float(ratio.loc[CONF_START:].mean())}

    T.to_csv(out / "trials.csv", index=False, float_format="%.6g")
    res = {"protocol": PROTOCOL, "windows": windows, "cost_bps": COST_BPS, "n_trials_this_study": int(len(T)),
           "n_trials_charged": n_total, "global": glob, "candidates": checks,
           "edge_confirmed": T.loc[T["edge_confirmed"], ["arm", "market", "rule"]].to_dict("records"),
           "risk_control": T.loc[T["risk_control"], ["arm", "market", "rule"]].to_dict("records"),
           "best_in_discovery": best, "cross_sectional_ic": ic, "segment_buy_and_hold": seg,
           "stock_segment_sizes": sizes_summary, "stock_data_coverage": cov, "runtime_sec": time.time() - t0}
    from .reentry import _clean_json
    (out / "results.json").write_text(json.dumps(_clean_json(res), indent=2, default=str))
    log(f"done in {res['runtime_sec']:.0f}s: {json.dumps(_clean_json(glob))}")
    return res


# --------------------------------------------------------------------------- real data
def load_inputs(config: str = "config.yaml") -> Inputs:
    from ..config import load_config
    from ..data import load_bundle
    from .real_firstlook import OVERRIDES as FIRSTLOOK
    cfg = load_config(config, FIRSTLOOK)
    d = REPO / cfg.data.csv_dir
    b = load_bundle(cfg)
    etf = pd.read_csv(d / "etf_prices.csv", parse_dates=["date"])
    eo = etf.pivot(index="date", columns="ticker", values="open")[ETFS]
    ec = etf.pivot(index="date", columns="ticker", values="close")[ETFS]
    sec = pd.read_csv(d / "sectors_yahoo.csv").set_index("ticker")["sector"]
    # every earlier real-data trial (first look + fundamentals), as alpha Sharpe vs SPY, for the DSR
    spy_m = ec["SPY"].resample("ME").last().pct_change()
    rf_m = (1 + b.risk_free.reindex(ec.index).ffill() / 252.0).resample("ME").prod() - 1
    prev, n_prev = [], 0
    for reg_dir in (REPO / "artifacts" / "real_firstlook" / "trials", REPO / "docs" / "real_fundamentals"):
        rp = reg_dir / "returns.csv"
        if rp.exists():
            R = pd.read_csv(rp, index_col=0, parse_dates=True)
            R.index = R.index.to_period("M")
            bx = (spy_m - rf_m).set_axis(spy_m.index.to_period("M"))
            for c in R.columns:
                a = alpha_returns(R[c], bx.reindex(R.index)).dropna()
                if len(a) > 12 and a.std() > 0:
                    prev.append(a.mean() / a.std())
        rj = reg_dir / "registry.jsonl"
        if rj.exists():
            n_prev += len({json.loads(l)["trial_id"] for l in rj.read_text().splitlines() if l.strip()})
    cov = pd.read_csv(d / "coverage.csv") if (d / "coverage.csv").exists() else None
    return Inputs(etf_open=eo, etf_close=ec, stock_open=b.open, stock_close=b.close, stock_mcap=b.market_cap,
                  member=b.membership, sectors=sec, rf=b.risk_free, vix=b.vix, prev_alpha_sharpes=np.array(prev),
                  n_previous=n_prev, coverage=cov)


def main() -> dict:
    if (OUT / "results.json").exists():
        raise SystemExit("this study already ran once")
    if not (OUT / "PROTOCOL.md").exists():
        raise SystemExit("commit the pre-registration first")
    inp = load_inputs()
    log(f"inputs: {inp.etf_close.shape[1]} ETFs, {inp.stock_close.shape[1]} stocks, "
        f"{inp.n_previous} earlier real-data trials")
    return run(inp, OUT)


if __name__ == "__main__":
    main()
