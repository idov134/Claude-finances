"""One pre-registered synthetic re-entry comparison, with no holdout access.

Run ``python -m stocksel.experiments.reentry`` from the repository root. This is
an engineering experiment, not evidence of expected profits in real markets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_daily
from ..backtest.monthly import monthly_backtest
from ..config import load_config, vol_target
from ..data import load_bundle
from ..features.pipeline import build_features
from ..models.labels import forward_returns, label_panel
from ..models.train import model_view, walk_forward
from ..portfolio.sizing import SizingParams
from ..validation.holdout import assert_research_data, holdout_start
from ..validation.metrics import (bootstrap_ci, cagr, classification_metrics, deflated_sharpe,
                                  performance_table)
from ..validation.pbo import pbo
from ..validation.trials import TrialRegistry


CANDIDATE = {"enabled": True, "cooldown_days": 21, "trend_days": 126,
             "recovery_fraction": 0.05}
KNOWN_PREVIOUS_TRIALS = 48


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(cfg, output: str | Path = "artifacts/gpt_review", bundle=None) -> dict:
    if cfg.data.provider != "synthetic":
        raise ValueError("This pre-registration is synthetic only; real data requires a new protocol")
    root = Path(output)
    reg = TrialRegistry(root / "trials", stage="reentry_review")
    configurations = {
        "permanent_halt": {"enabled": False, **{k: v for k, v in CANDIDATE.items() if k != "enabled"}},
        "guarded_reentry": CANDIDATE,
    }
    prereg = {
        name: {"experiment": "gpt_reentry_v1", "config_hash": cfg.hash(),
               "model": "logit", "reentry": rule,
               "known_previous_trials": KNOWN_PREVIOUS_TRIALS,
               "data": "synthetic_research_only", "engine": "lagged_open_costs_initial_equity_v1"}
        for name, rule in configurations.items()
    }
    for config in prereg.values():
        reg.log(config, status="planned", tags={"protocol": "docs/ALGO_REVIEW.md"})
    try:
        full = bundle if bundle is not None else load_bundle(cfg)
        hs = holdout_start(full.dates, int(cfg.validation.holdout_months))
        data = full.truncate(hs - pd.Timedelta(days=1))
        assert_research_data(data, hs)
        del full
        panel = build_features(data, cfg)
        feats = list(cfg.features.sets[cfg.models.feature_set])
        X = panel.features(feats)
        labels = forward_returns(data, panel.dates, int(cfg.labels.horizon_months),
                                 cfg.labels.benchmark, cfg.data.default_delisting_return, panel.universe)
        L = label_panel(labels, X.index)
        wf = walk_forward(X, L, ["logit"], cfg)
        if wf.preds.empty:
            raise ValueError("No walk-forward predictions; comparison is unrun")
        preds = model_view(wf.preds, "logit")
        targets = L.reindex(preds.index)
        classification = classification_metrics(preds.prob, targets.y, targets.excess,
                                                 preds.index.get_level_values(0))
        if classification["accuracy"] > float(cfg.validation.red_flags.max_accuracy):
            raise RuntimeError("Directional accuracy exceeds leakage red flag; stop and investigate")
        sizing = SizingParams.from_cfg(cfg, 1 - float(cfg.signals.haircut), vol_target(cfg))
        monthly = monthly_backtest(preds, panel, labels, data, cfg, sizing)
    except Exception as exc:
        for config in prereg.values():
            reg.log(config, status="failed", error=repr(exc))
        raise

    results, returns = {}, {}
    for name, rule in configurations.items():
        config = prereg[name]
        try:
            engine = run_daily(data, monthly.weights,
                               cfg.copy_with({"risk.kill_switch.reentry": rule}), kill_switch=True)
            m = engine.monthly()
            if len(m) < 24:
                raise ValueError("Too few OOS months for the registered inference")
            rf = (data.risk_free.reindex(m.index, method="ffill") / 12
                  if data.risk_free is not None else pd.Series(0., index=m.index))
            metrics = performance_table(m.net, m.bench, rf, boot=dict(cfg.validation.bootstrap))
            if metrics["sharpe"] > float(cfg.validation.red_flags.max_sharpe):
                raise RuntimeError("Sharpe exceeds leakage red flag; stop and investigate")
            daily_peak = engine.daily.equity.cummax().clip(lower=float(cfg.capital.initial))
            daily_dd = float((engine.daily.equity / daily_peak - 1).min())
            metrics["monthly_max_drawdown"] = metrics["max_drawdown"]
            metrics["max_drawdown"] = daily_dd
            metrics["calmar"] = metrics["cagr"] / abs(daily_dd) if daily_dd < 0 else np.nan
            metrics["kill_events"] = len(engine.kill_events)
            metrics["reentry_events"] = len(engine.reentry_events)
            metrics["total_cost_usd"] = float(engine.daily.cost.sum())
            lo, hi, _ = bootstrap_ci(m.net, stat=cagr, **{
                k: v for k, v in dict(cfg.validation.bootstrap).items() if k in {"n", "block", "seed"}})
            metrics["cagr_ci_low"], metrics["cagr_ci_high"] = lo, hi
            returns[name] = m.net - rf
            results[name] = metrics
            reg.log(config, returns[name], metrics)
        except Exception as exc:
            reg.log(config, status="failed", error=repr(exc))
            raise
    matrix = reg.returns()
    trial_count = reg.n_trials("reentry_review")
    charged_count = KNOWN_PREVIOUS_TRIALS + trial_count
    current_srs = reg.trial_sharpes()
    # Prior returns are absent from the repository. Use a nonzero sampling-
    # variance floor, label the result diagnostic, and prohibit promotion.
    variance = max(float(np.var(current_srs, ddof=1)), 1 / len(matrix))
    for name in results:
        results[name]["dsr_diagnostic"] = deflated_sharpe(
            returns[name], charged_count, var_sr=variance)["dsr"]
    pair = pbo(pd.DataFrame(returns), **dict(cfg.validation.pbo))
    diff = returns["guarded_reentry"] - returns["permanent_halt"]
    lo, hi, prob = bootstrap_ci(diff, **{
        k: v for k, v in dict(cfg.validation.bootstrap).items() if k in {"n", "block", "seed"}})
    report = {
        "protocol": "gpt_reentry_v1", "synthetic_only": True,
        "config_snapshot": cfg.to_dict(), "config_hash": cfg.hash(),
        "research_start": str(data.dates[0].date()), "research_end": str(data.dates[-1].date()),
        "excluded_holdout_start": str(hs.date()), "holdout_opened": False,
        "n_tickers": len(data.tickers), "n_predictions": len(wf.preds),
        "new_distinct_trials": trial_count, "dsr_charged_trials_at_least": charged_count,
        "dsr_limitation": "Prior trial-return variance unavailable; diagnostic only, not certification",
        "metrics": results,
        "classification": classification,
        "pair_pbo": {k: v for k, v in pair.items() if k != "logits"},
        "combined_pbo": None,
        "annualized_net_improvement_ci": [lo * 12, hi * 12],
        "bootstrap_probability_net_improvement": prob,
        "decision": "Disabled by default: synthetic only and prior full trial matrix unavailable",
    }
    report = _clean_json(report)
    root.mkdir(parents=True, exist_ok=True)
    (root / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output", default="artifacts/gpt_review")
    args = ap.parse_args()
    print(json.dumps(run(load_config(args.config), args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
