"""Real-data first look (pre-registered in docs/real_firstlook/PROTOCOL.md). Run once:

    python -m stocksel.data.build_yahoo_csv --out data_csv      # local data, never committed
    python -m stocksel.experiments.real_firstlook

Writes aggregate results (no raw prices) to docs/real_firstlook/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import numpy as np

from ..config import load_config
from ..data import load_bundle
from ..data.universe import eligible
from ..models.labels import forward_returns
from ..pipeline import run_holdout, run_research
from ..validation.metrics import performance_table
from ..reports.html import write_report
from ..validation.readiness import holdout_passed
from .reentry import _clean_json

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "real_firstlook"
PRICE_ONLY = ["mom_12_1", "mom_6_1", "rev_1m", "illiq", "vol_252", "idio_vol", "size"]
OVERRIDES = {
    "data.provider": "csv", "data.csv_dir": "data_csv", "data.start": "1999-01-01",
    "data.cache_dir": "artifacts/real_firstlook/cache", "paths.artifacts": "artifacts/real_firstlook",
    "live.ledger_dir": "artifacts/real_firstlook/paper", "report.output_dir": "reports_out/real_firstlook",
    "report.title": "stocksel — REAL DATA first look (Yahoo, survivorship-biased)",
    "features.sets.price_only": PRICE_ONLY, "models.feature_set": "price_only",
    "trials.feature_sets": ["price_only"],
}
EXTRA = {"relative": {"portfolio.mandate": "relative"},
         "relative_regime": {"portfolio.mandate": "relative", "portfolio.regime_filter.enabled": True}}
KEYS = ("cagr", "bench_cagr", "excess_ann", "excess_ci_low", "excess_ci_high", "p_excess_positive", "sharpe",
        "sharpe_ci_low", "sharpe_ci_high", "bench_sharpe", "sortino", "max_drawdown", "bench_max_drawdown", "calmar",
        "information_ratio", "capm_alpha", "capm_beta", "hit_rate_periods", "turnover_ann", "cost_drag_ann",
        "n_periods")


def pick(m: dict) -> dict:
    return {k: m.get(k) for k in KEYS}


def universe_reference(cfg) -> pd.DataFrame:
    """Cap- and equal-weighted return of every eligible stock (same survivor-only universe), frictionless."""
    b = load_bundle(cfg)
    dates = b.month_end_dates()
    elig = eligible(b, cfg, dates)
    lab = forward_returns(b, dates, 1, cfg.labels.benchmark, cfg.data.default_delisting_return, elig)
    fwd = lab.fwd.where(elig)
    w = b.market_cap.loc[dates].where(elig & fwd.notna())
    out = pd.DataFrame({"universe_cw": (fwd * w).sum(axis=1) / w.sum(axis=1), "universe_ew": fwd.mean(axis=1),
                        "spy": lab.bench, "n_eligible": elig.sum(axis=1)})
    return out.replace([np.inf, -np.inf], np.nan)


def vs_reference(monthly: pd.DataFrame, ref: pd.DataFrame) -> dict:
    out = {}
    for col in ("universe_cw", "universe_ew"):
        j = pd.concat({"net": monthly["net"], "ref": ref[col], "rf": monthly["rf"]}, axis=1).dropna()
        m = performance_table(j["net"], j["ref"], j["rf"], boot={"n": 2000, "block": 6, "seed": 3})
        out[col] = {"ref_cagr": m["bench_cagr"], "excess_ann": m["excess_ann"], "excess_ci_low": m["excess_ci_low"],
                    "excess_ci_high": m["excess_ci_high"], "p_excess_positive": m["p_excess_positive"],
                    "information_ratio": m["information_ratio"], "ref_sharpe": m["bench_sharpe"],
                    "ref_max_drawdown": m["bench_max_drawdown"], "months": m["n_periods"]}
    return out


def main(config: str = "config.yaml") -> dict:
    t0 = time.time()
    cfg = load_config(config, OVERRIDES)
    data_dir = REPO / cfg.data.csv_dir
    if not (data_dir / "prices.csv").exists():
        raise SystemExit("build the dataset first: python -m stocksel.data.build_yahoo_csv --out data_csv")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("the real-data first look already ran; its holdout is used")
    res = run_research(cfg, extra_portfolios=EXTRA)
    write_report(res, OUT / "research_report.html")
    hres = run_holdout(cfg, extra_portfolios=EXTRA)
    write_report(hres, OUT / "holdout_report.html")
    ref = universe_reference(cfg)
    ref.to_csv(OUT / "universe_reference_monthly.csv")
    bias_matched = {"research": {"default": vs_reference(res["monthly"], ref),
                                 **{k: vs_reference(v["monthly"], ref) for k, v in res["extra_portfolios"].items()}},
                    "holdout": {"default": vs_reference(hres["monthly"], ref),
                                **{k: vs_reference(v["monthly"], ref) for k, v in hres["extra_portfolios"].items()}}}
    info = json.loads((data_dir / "dataset_info.json").read_text())
    cov = pd.read_csv(data_dir / "coverage.csv")
    cov["share"] = cov["with_prices"] / cov["members"]
    cov["year"] = cov["date"].str[:4]
    out = {
        "protocol": "real_firstlook_v1", "dataset": info,
        "coverage_by_year": cov.groupby("year")["share"].mean().round(3).to_dict(),
        "research": {"window": [str(res["monthly"].index.min().date()), str(res["monthly"].index.max().date())],
                     "default": pick(res["final_metrics"]), "cls": res["cls"],
                     "dsr": res["dsr"], "pbo": res["pbo"].get("pbo"), "n_trials": res["n_trials"],
                     "gate_last": res["gate"][max(res["gate"])],
                     "extra": {k: pick(v["metrics"]) for k, v in res["extra_portfolios"].items()},
                     "sanity": {k: v.get("passed") for k, v in res["sanity"].items()},
                     "engine_with_kill": pick(res["engine_metrics"]), "kill_events": res["kill_events"],
                     "engine_no_kill": pick(res["engine_no_kill_metrics"]),
                     "decay": res["decay_table"].reset_index().to_dict(orient="records"),
                     "regimes": res["sensitivity"]["regimes"].reset_index().to_dict(orient="records"),
                     "red_flags": res["red_flags"]},
        "holdout": {"window": [str(hres["monthly"].index.min().date()), str(hres["monthly"].index.max().date())],
                    "default": pick(hres["final_metrics"]), "cls": hres["cls"],
                    "extra": {k: pick(v["metrics"]) for k, v in hres["extra_portfolios"].items()},
                    "passed_readiness_gate": holdout_passed(hres, cfg), "lock": hres["lock"],
                    "red_flags": hres["red_flags"]},
        "bias_matched_reference": bias_matched,
        "runtime_sec": time.time() - t0,
    }
    (OUT / "results.json").write_text(json.dumps(_clean_json(out), indent=2, default=str))
    print(json.dumps({"research": out["research"]["default"], "holdout": out["holdout"]["default"]}, default=str))
    return out


if __name__ == "__main__":
    main()
