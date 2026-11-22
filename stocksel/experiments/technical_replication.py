"""Replication of the technical-rule leads on 16 country ETFs never used before.

Pre-registered in docs/technical_replication/PROTOCOL.md. Run once:

    python -m stocksel.data.build_etf_csv --data data_csv --intl
    python -m stocksel.experiments.technical_replication
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.rules import monthly, open_to_open, simulate
from ..features.technical import Indicators, Market, rule_positions
from ..validation.metrics import max_drawdown
from ..validation.trials import TrialRegistry
from .technical_rules import (CONF_START, REPO, SECTOR_ETFS, SPLIT_END, START, etf_shift_p, monthly_alpha,
                              window_metrics)

PROTOCOL = "technical_replication_v1"
OUT = REPO / "docs" / "technical_replication"
INTL = ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWN", "EWO", "EWP", "EWQ", "EWS", "EWU", "EWW"]
DIP = ["BB_MR_TREND", "RSI2"]
TREND = ["SMA200", "GOLDEN", "FABER10"]
COST_BPS, US_COST_BPS = 10.0, 3.0
CAP = 0.25
N_SHIFTS = 500
H1 = {"t": 2.0, "min_positive": 11}
H2 = {"dd_ratio": 2.0 / 3.0, "p": 0.05, "replicated": 9, "partly": 5}
N_PREVIOUS = 297


def log(msg: str) -> None:
    print(f"[replication {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_t(x: pd.Series, lags: int = 3) -> dict:
    """Annualized mean of a monthly series with a Newey-West t-statistic."""
    v = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(v)
    if n < 12:
        return {"mean_ann": float("nan"), "t": float("nan"), "months": n}
    u = v - v.mean()
    s = (u @ u) / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (u[lag:] @ u[:-lag]) / n
    return {"mean_ann": float(v.mean() * 12), "t": float(v.mean() / math.sqrt(s / n)) if s > 0 else float("nan"),
            "months": n}


def rotation(pos: pd.DataFrame, valid: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spread capital over the markets with an active signal: min(1/#active, cap) each, the rest in cash."""
    active = (pos.fillna(0.0) > 0) & valid
    k = active.sum(axis=1)
    w = (1.0 / k.replace(0, np.nan)).clip(upper=cap).fillna(0.0)
    return active.astype(float), active.astype(float).mul(w, axis=0)


def _windows(index: pd.DatetimeIndex) -> dict:
    end = str(index[-1].date())
    return {"disc": (START, SPLIT_END), "conf": (CONF_START, end), "full": (START, end)}


def portfolio_eval(eo: pd.DataFrame, ec: pd.DataFrame, rf: pd.Series, spy: pd.Series, rule: str, cost: float) -> dict:
    r = open_to_open(eo)
    rf_d = (rf.reindex(ec.index).ffill() / 252.0).fillna(0.0)
    valid = ec.notna()
    ew = simulate(valid.astype(float), valid.astype(float).div(valid.sum(axis=1), axis=0), r, rf_d, cost)
    pos = rule_positions(rule, Indicators(ec), Market(pd.Series(dtype=float), pd.Series(dtype=float)))
    sig, base = rotation(pos, valid, CAP)
    out = {}
    for c in (cost, 2 * cost):
        s = simulate(sig, base, r, rf_d, c)
        out[f"{c:g}bps"] = {w: window_metrics(s, ew, rf_d, spy, lo, hi) for w, (lo, hi) in _windows(s.index).items()}
    m = out[f"{cost:g}bps"]
    out["usable"] = bool(m["full"]["sharpe"] >= m["full"]["bh_sharpe"] and m["full"]["t_alpha"] >= 2.0
                         and m["disc"]["alpha"] > 0 and m["conf"]["alpha"] > 0
                         and m["full"]["max_dd"] >= H2["dd_ratio"] * m["full"]["bh_max_dd"])
    return out


def run(eo: pd.DataFrame, ec: pd.DataFrame, rf: pd.Series, spy: pd.Series, out: Path,
        us_eo: pd.DataFrame | None = None, us_ec: pd.DataFrame | None = None, n_shifts: int = N_SHIFTS,
        seed: int = 11) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    reg = TrialRegistry(out, stage=PROTOCOL)
    r = open_to_open(eo)
    rf_d = (rf.reindex(ec.index).ffill() / 252.0).fillna(0.0)
    one = ec.notna().astype(float)
    bh = {m: simulate(one[[m]], one[[m]], r[[m]], rf_d, COST_BPS) for m in ec.columns}
    win = _windows(bh[ec.columns[0]].index)
    ind = Indicators(ec)
    mkt = Market(pd.Series(dtype=float), pd.Series(dtype=float))
    rows, alpha_m, alpha_2x = [], {k: {} for k in DIP + TREND}, {k: {} for k in DIP}
    for rule in DIP + TREND:
        pos = rule_positions(rule, ind, mkt)
        for m in ec.columns:
            s = simulate(pos[[m]], one[[m]], r[[m]], rf_d, COST_BPS)
            row = {"market": m, "rule": rule}
            for w, (lo, hi) in win.items():
                row.update({f"{w}_{k}": v for k, v in window_metrics(s, bh[m], rf_d, spy, lo, hi).items()})
            lo, hi = win["full"]
            row["shift_p"], row["dd_shift_p"] = etf_shift_p(pos[m], r[m], rf_d, bh[m], lo, hi, COST_BPS, n_shifts,
                                                            seed)
            alpha_m[rule][m] = monthly_alpha(s.loc[lo:hi], bh[m].loc[lo:hi], rf_d)
            if rule in DIP:
                s2 = simulate(pos[[m]], one[[m]], r[[m]], rf_d, 2 * COST_BPS)
                alpha_2x[rule][m] = monthly_alpha(s2.loc[lo:hi], bh[m].loc[lo:hi], rf_d)
            rfm = monthly(rf_d.reindex(s.loc[lo:hi].index))
            reg.log({"protocol": PROTOCOL, "market": m, "rule": rule, "cost_bps": COST_BPS},
                    (monthly(s.loc[lo:hi, "ret"]) - rfm).rename(None),
                    {k: v for k, v in row.items() if k.startswith("full_")})
            rows.append(row)
        log(f"per-market: {rule}")
    T = pd.DataFrame(rows)

    # H1: pooled dip-buying alpha
    h1 = {}
    for rule in DIP:
        A = pd.DataFrame(alpha_m[rule])
        pooled, pooled2 = A.mean(axis=1), pd.DataFrame(alpha_2x[rule]).mean(axis=1)
        t = T[T["rule"] == rule].set_index("market")
        n_pos = int((t["full_alpha"] > 0).sum())
        res = {"full": mean_t(pooled), "disc": mean_t(pooled.loc[START:SPLIT_END]),
               "conf": mean_t(pooled.loc[CONF_START:]), "full_20bps": mean_t(pooled2),
               "markets_positive": n_pos, "per_market_alpha": t["full_alpha"].round(4).to_dict()}
        res["replicated"] = bool(res["full"]["t"] >= H1["t"] and res["disc"]["mean_ann"] > 0
                                 and res["conf"]["mean_ann"] > 0 and n_pos >= H1["min_positive"]
                                 and res["full_20bps"]["mean_ann"] > 0)
        h1[rule] = res
    h1_verdict = ("replicates" if all(v["replicated"] for v in h1.values()) else
                  "partly replicates" if any(v["replicated"] for v in h1.values()) else "does not replicate")

    # H2: trend rules as drawdown control
    T["risk_pre"] = ((T["disc_max_dd"] >= H2["dd_ratio"] * T["disc_bh_max_dd"])
                     & (T["conf_max_dd"] >= H2["dd_ratio"] * T["conf_bh_max_dd"])
                     & (T["disc_sharpe"] >= T["disc_bh_sharpe"]) & (T["conf_sharpe"] >= T["conf_bh_sharpe"]))
    T["risk_flag"] = T["risk_pre"] & (T["dd_shift_p"] <= H2["p"])
    h2 = {}
    for rule in TREND:
        t = T[T["rule"] == rule]
        n = int(t["risk_flag"].sum())
        h2[rule] = {"markets_flagged": n, "markets_dd_and_sharpe": int(t["risk_pre"].sum()),
                    "median_dd_shift_p": float(t["dd_shift_p"].median()),
                    "median_dd_ratio_full": float((t["full_max_dd"] / t["full_bh_max_dd"]).median()),
                    "median_cagr_minus_bh_full": float((t["full_cagr"] - t["full_bh_cagr"]).median()),
                    "verdict": ("replicates" if n >= H2["replicated"] else
                                "partly replicates" if n >= H2["partly"] else "does not replicate")}

    # H3: rotation portfolios (+ in-sample reference on US sectors)
    h3 = {rule: portfolio_eval(eo, ec, rf, spy, rule, COST_BPS) for rule in DIP}
    ref = {}
    if us_eo is not None and us_ec is not None:
        ref = {rule: portfolio_eval(us_eo, us_ec, rf, spy, rule, US_COST_BPS) for rule in DIP}
    for rule, res in h3.items():
        m = res[f"{COST_BPS:g}bps"]["full"]
        reg.log({"protocol": PROTOCOL, "portfolio": "rotation", "rule": rule, "cost_bps": COST_BPS}, None, m)
    log("portfolios done")

    T.to_csv(out / "trials.csv", index=False, float_format="%.6g")
    result = {"protocol": PROTOCOL, "windows": win, "markets": list(ec.columns), "cost_bps": COST_BPS,
              "n_trials_registry_total": N_PREVIOUS + len(T) + len(h3),
              "H1_dip_buying": {"verdict": h1_verdict, **h1}, "H2_trend_drawdown": h2,
              "H3_rotation": h3, "in_sample_reference_us_sectors": ref, "runtime_sec": time.time() - t0}
    from .reentry import _clean_json
    (out / "results.json").write_text(json.dumps(_clean_json(result), indent=2, default=str))
    log(f"done in {result['runtime_sec']:.0f}s: H1 {h1_verdict}; H2 "
        f"{ {k: v['verdict'] for k, v in h2.items()} }; H3 usable { {k: v['usable'] for k, v in h3.items()} }")
    return result


def liquid_subset(prices: pd.DataFrame, k: int = 8) -> list[str]:
    """The k markets with the highest average dollar volume since 2013 (descriptive robustness)."""
    p = prices[prices["date"] >= CONF_START]
    dv = (p["close"] * p["volume"]).groupby(p["ticker"]).mean()
    return list(dv.sort_values(ascending=False).index[:k])


def main() -> dict:
    if (OUT / "results.json").exists():
        raise SystemExit("this study already ran once")
    if not (OUT / "PROTOCOL.md").exists():
        raise SystemExit("commit the pre-registration first")
    d = REPO / "data_csv"
    intl = pd.read_csv(d / "etf_intl_prices.csv", parse_dates=["date"])
    us = pd.read_csv(d / "etf_prices.csv", parse_dates=["date"])
    eo = intl.pivot(index="date", columns="ticker", values="open")[INTL]
    ec = intl.pivot(index="date", columns="ticker", values="close")[INTL]
    us_eo = us.pivot(index="date", columns="ticker", values="open")
    us_ec = us.pivot(index="date", columns="ticker", values="close")
    rf = pd.read_csv(d / "rf.csv", index_col=0, parse_dates=True)["rf"]
    spy = simulate(us_ec[["SPY"]].notna().astype(float), us_ec[["SPY"]].notna().astype(float),
                   open_to_open(us_eo[["SPY"]]), (rf.reindex(us_ec.index).ffill() / 252).fillna(0.0), 3.0)["ret"]
    res = run(eo, ec, rf, spy, OUT, us_eo[SECTOR_ETFS], us_ec[SECTOR_ETFS])
    # descriptive: H1 on the 8 most liquid markets only
    top = liquid_subset(intl)
    sub = {}
    for rule in DIP:
        al = []
        for m in top:
            one = ec[[m]].notna().astype(float)
            rf_d = (rf.reindex(ec.index).ffill() / 252.0).fillna(0.0)
            r = open_to_open(eo[[m]])
            pos = rule_positions(rule, Indicators(ec[[m]]), Market(pd.Series(dtype=float), pd.Series(dtype=float)))
            lo, hi = res["windows"]["full"]
            s = simulate(pos, one, r, rf_d, COST_BPS).loc[lo:hi]
            b = simulate(one, one, r, rf_d, COST_BPS).loc[lo:hi]
            al.append(monthly_alpha(s, b, rf_d))
        sub[rule] = mean_t(pd.concat(al, axis=1).mean(axis=1))
    res["H1_liquid_8"] = {"markets": top, **sub}
    from .reentry import _clean_json
    (OUT / "results.json").write_text(json.dumps(_clean_json(res), indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
