"""Bounded, pre-registered comparison diagnostics; never promotes a strategy.

Run from the checkout with ``python -m stocksel.experiments.comparison``.
See docs/research_comparison/PROTOCOL.md for the registered eleven configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.monthly import candidates_at, market_vol
from ..config import load_config, vol_target
from ..data import load_bundle
from ..features.pipeline import build_features
from ..models.labels import forward_returns, label_panel
from ..models.train import model_view, walk_forward
from ..portfolio.sizing import SizingParams, portfolio_vol, target_weights, waterfill
from ..validation.holdout import assert_research_data, holdout_start
from ..validation.metrics import classification_metrics, drawdown_series, performance_table
from ..validation.pbo import pbo
from ..validation.trials import TrialRegistry
from .reentry import _clean_json

PROTOCOL = "research_comparison_v1"
RULES = ("current_logit", "transparent_factor", "equal_weight")
MULTIPLIERS = (1, 2, 3)
BOOT = {"n": 500, "block": 6, "seed": 3}


def research_code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), root / "config.py"]
    for package in ("data", "features", "models", "portfolio", "backtest", "validation"):
        paths.extend(sorted((root / package).rglob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def research_only(full, cfg):
    boundary = min(holdout_start(full.dates, int(cfg.validation.holdout_months)), pd.Timestamp("2024-01-01"))
    data = full.truncate(boundary - pd.Timedelta(days=1))
    assert_research_data(data, boundary)
    return data, boundary


def dataset_fingerprint(bundle):
    """Hash the actual permitted inputs, including event availability records."""
    digest = hashlib.sha256()
    for name in ("open", "high", "low", "close", "volume", "market_cap", "sectors", "benchmark",
                 "membership", "delist_returns", "fundamentals", "earnings", "estimates", "news",
                 "vix", "risk_free"):
        value = getattr(bundle, name)
        digest.update(name.encode())
        if value is not None:
            digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
            digest.update(str(getattr(value, "columns", "series")).encode())
    return digest.hexdigest()


def rank_diagnostics(scores, labels):
    """Cross-sectional rank IC and forward-return quintiles, never probabilities."""
    frame = pd.DataFrame({"score": scores, "excess": labels.excess, "forward": labels.fwd}).dropna()
    frame["decision_date"] = frame.index.get_level_values(0)
    ic = frame.groupby("decision_date").apply(lambda g: g.score.corr(g.excess, method="spearman"), include_groups=False)
    rank = frame.groupby("decision_date").score.rank(method="first", pct=True)
    frame["bucket"] = np.minimum(5, np.ceil(rank * 5)).astype(int)
    buckets = frame.groupby(["decision_date", "bucket"])[["forward", "excess"]].mean()
    return {"ic_mean": float(ic.mean()), "ic_months": int(ic.notna().sum()),
            "quintiles_low_to_high": [{"bucket": int(bucket), "mean_monthly_return": float(group.forward.mean()),
                "mean_monthly_excess": float(group.excess.mean()), "months": len(group)}
                for bucket, group in buckets.groupby(level="bucket")],
            "definition": "Equal-weight forward returns within monthly score quintiles; gross of implementation costs."}


def forecast_diagnostics(predictions, targets):
    valid = predictions.prob.notna() & targets.y.notna()
    pr, target = predictions.loc[valid], targets.loc[valid]
    result = classification_metrics(pr.prob, target.y, target.excess, pr.index.get_level_values(0))
    reference = classification_metrics(np.full(len(pr), .5), target.y)
    frame = pd.DataFrame({"probability": pr.prob, "outcome": target.y})
    # Fixed bins are declared beforehand; empty bins remain empty, not zero-rate.
    frame["bin"] = pd.cut(frame.probability, np.linspace(0, 1, 11), include_lowest=True)
    result["calibration"] = [{"bin": str(label), "n": len(group),
        "mean_probability": float(group.probability.mean()) if len(group) else None,
        "observed_fraction": float(group.outcome.mean()) if len(group) else None}
        for label, group in frame.groupby("bin", observed=False)]
    result["constant_half_probability"] = reference
    result["event"] = "Next-open to next-month-rebalance-open stock return exceeds the same-window benchmark return"
    result["horizon_months"] = 1
    return result


def allocated_weights(rule, cands, scores, previous, cfg, market_sigma):
    prm = SizingParams.from_cfg(cfg, 1 - float(cfg.signals.haircut), vol_target(cfg))
    if rule == "current_logit":
        return target_weights(cands, previous, prm, market_sigma)[0]
    selected = cands
    if rule == "transparent_factor":
        selected = cands.loc[scores.reindex(cands.index).dropna().sort_values(ascending=False).head(prm.top_n).index]
    elif rule != "equal_weight":
        raise ValueError("Unknown pre-registered allocation rule")
    if selected.empty:
        return pd.Series(dtype=float)
    weights = waterfill(pd.Series(1., index=selected.index), pd.Series(prm.max_position, index=selected.index),
                        selected.sector.fillna("Unknown"), prm.max_sector, prm.max_gross)
    sigma = portfolio_vol(weights, selected.beta, selected.idio_vol, market_sigma)
    weights *= min(1., prm.vol_target / sigma) if sigma > 0 else 1.
    return weights[weights > 1e-8]


def fee_reserved_weights(target, previous, adv, sigma, equity, cost_model):
    """Fund costs from cash rather than assume free fractional leverage."""
    index = target.index.union(previous.index)
    old = previous.reindex(index).fillna(0.)
    wanted = target.reindex(index).fillna(0.)

    def cost(w):
        trade = abs(w - old)
        usd = cost_model.cost_usd(trade.to_numpy() * equity,
                                 adv.reindex(index).to_numpy(), sigma.reindex(index).to_numpy())
        return sum(float(x.sum()) for x in usd.values()) / equity

    # Contraction from full investment keeps cash nonnegative; no return data used.
    weights = wanted.copy()
    for _ in range(30):
        fees = cost(weights)
        if weights.sum() + fees <= 1 + 1e-12:
            return weights[weights > 1e-10], fees, float(abs(weights - old).sum())
        weights = wanted * max(0., 1 - fees) / max(1., float(wanted.sum()))
    raise ValueError("Could not reserve cash for modeled transaction costs")


def executable_targets(wanted, previous, entry_prices):
    """Unavailable entry opens are unfilled orders, not zero-return investments.

    This uses availability at execution, never future exit returns. No alternative
    name is selected after seeing the opening prices. Existing positions without
    an executable price are retained for the strict return-availability check.
    """
    index = wanted.index.union(previous.index)
    result = wanted.reindex(index).fillna(0.)
    prices = entry_prices.reindex(index)
    unavailable = ~np.isfinite(prices) | (prices <= 0)
    result.loc[unavailable] = previous.reindex(index).fillna(0.).loc[unavailable]
    return result[result > 1e-10], int((unavailable & (wanted.reindex(index).fillna(0.) > 0)).sum())


def backtest(rule, multiplier, predictions, factors, panel, labels, bundle, cfg):
    by_date = {date: group.droplevel(0) for date, group in predictions.groupby(level=0)}
    dates = pd.DatetimeIndex([date for date in by_date if np.isfinite(labels.bench.get(date, np.nan))])
    sigmas = market_vol(bundle, dates)
    cost_model = CostModel(cfg.costs, multiplier)
    previous, equity, rows = pd.Series(dtype=float), float(cfg.capital.initial), []
    for date in dates:
        cands = candidates_at(date, by_date, panel, bundle.sectors)
        wanted = allocated_weights(rule, cands, factors.xs(date), previous, cfg, float(sigmas.loc[date]))
        wanted, unfilled = executable_targets(wanted, previous, bundle.open.loc[labels.entry.loc[date]])
        weights, fees, turnover = fee_reserved_weights(wanted, previous, panel.aux["adv"].loc[date],
                                                       panel.aux["vol_d"].loc[date], equity, cost_model)
        forward = labels.fwd.loc[date].reindex(weights.index)
        if forward.isna().any():
            raise ValueError("Unavailable held forward return; refusing zero substitution")
        rf = float(bundle.risk_free.loc[:date].iloc[-1]) / 12 if bundle.risk_free is not None else 0.
        exposure = float(weights.sum())
        gross = float((weights * forward).sum()) + (1 - exposure) * rf
        # Fee reserve is spent at entry, so it does not also accrue cash interest.
        net = gross - fees * (1 + rf)
        equity *= 1 + net
        if equity <= 0:
            raise ValueError("Nonpositive portfolio equity")
        previous = weights * (1 + forward) / (1 + net)
        # forward_returns already credits terminal delisting proceeds in the
        # ending equity. Securities with no exit open are now cash, not phantom
        # holdings requiring an extra sale at the next rebalance.
        exit_prices = bundle.open.loc[labels.exit.loc[date]].reindex(previous.index)
        previous = previous[np.isfinite(exit_prices) & (exit_prices > 0)]
        sec = weights.groupby(bundle.sectors.reindex(weights.index)).sum()
        rows.append({"date": date, "net": net, "gross": gross, "bench": labels.bench.loc[date],
                     "rf": rf, "cost": fees, "turnover": turnover, "exposure": exposure,
                     "max_weight": float(weights.max()) if len(weights) else 0.,
                     "max_sector": float(sec.max()) if len(sec) else 0., "n_positions": len(weights),
                     "herfindahl": float((weights ** 2).sum()), "equity": equity, "unfilled_entry_targets": unfilled})
    return pd.DataFrame(rows).set_index("date")


def longest_drawdown_months(returns):
    dd = drawdown_series(returns)
    longest = current = 0
    for value in dd:
        current = current + 1 if value < -1e-12 else 0
        longest = max(longest, current)
    return longest


def economics(gross_excess_ann, net_excess_ann, turnover_ann, annual_fixed_cost):
    return {"annual_fixed_cost_usd": annual_fixed_cost,
            "fixed_cost_is_illustrative_user_input": True,
            "total_break_even_cost_bps_per_traded_dollar": gross_excess_ann / turnover_ann * 10000
                if gross_excess_ann > 0 and turnover_ann > 0 else None,
            "additional_break_even_cost_bps_per_traded_dollar": net_excess_ann / turnover_ann * 10000
                if net_excess_ann > 0 and turnover_ann > 0 else None,
            "capital_usd_to_cover_fixed_cost_if_point_estimate_persisted": annual_fixed_cost / net_excess_ann
                if net_excess_ann > 0 else None,
            "limitations": "Linear point-estimate scenario, not a viable-edge claim; taxes excluded and impact changes with capital."}


def summarize(monthly, annual_fixed_cost=600.):
    metrics = performance_table(monthly.net, monthly.bench, monthly.rf, boot=BOOT)
    metrics.update(cumulative_return=float((1 + monthly.net).prod() - 1),
                   drawdown_duration_months=longest_drawdown_months(monthly.net),
                   turnover_ann=float(monthly.turnover.mean() * 12),
                   cost_drag_ann=float(monthly.cost.mean() * 12), exposure_mean=float(monthly.exposure.mean()),
                   maximum_position_weight=float(monthly.max_weight.max()), maximum_sector_weight=float(monthly.max_sector.max()),
                   herfindahl_mean=float(monthly.herfindahl.mean()), tail_fifth_percentile_month=float(monthly.net.quantile(.05)),
                   worst_month=float(monthly.net.min()), terminal_equity=float(monthly.equity.iloc[-1]))
    metrics["unfilled_entry_targets"] = int(monthly.unfilled_entry_targets.sum()) if "unfilled_entry_targets" in monthly else 0
    ex_gross = float((monthly.gross - monthly.bench).mean() * 12)
    metrics["economics"] = economics(ex_gross, metrics["excess_ann"], metrics["turnover_ann"], annual_fixed_cost)
    metrics["economics"]["positive_excess_ci"] = metrics["excess_ci_low"] > 0
    metrics["economics"]["annual_fixed_cost_fraction_at_initial_capital"] = annual_fixed_cost / (monthly.equity.iloc[0] / (1 + monthly.net.iloc[0]))
    bsd = monthly.bench.std()
    scale = min(1., monthly.net.std() / bsd) if bsd > 0 else 0.
    risk_matched = monthly.rf + scale * (monthly.bench - monthly.rf)
    matched = performance_table(monthly.net, risk_matched, monthly.rf, boot=BOOT)
    metrics["risk_matched_reference"] = {"scale": scale, "excess_ann": matched["excess_ann"],
        "excess_ci_low": matched["excess_ci_low"], "excess_ci_high": matched["excess_ci_high"],
        "definition": "Ex-post full-window volatility scaling, cash remainder; descriptive, not tradable."}
    metrics["subperiods"] = {}
    for name, start, end in (("pre_2008", "1900", "2007-12-31"), ("gfc", "2008", "2009-12-31"),
                              ("2010s", "2010", "2019-12-31"), ("2020s", "2020", "2023-12-31")):
        part = monthly.loc[start:end]
        if len(part):
            m = performance_table(part.net, part.bench, part.rf, boot=BOOT)
            metrics["subperiods"][name] = {k: m[k] for k in ("n_periods", "cagr", "excess_ann", "sharpe", "max_drawdown")}
    return metrics


def run(cfg, output="artifacts/research_comparison", bundle=None, annual_fixed_cost=600.):
    if cfg.data.provider != "synthetic" or int(cfg.labels.horizon_months) != 1:
        raise ValueError("This registered comparison requires synthetic data and a one-month horizon")
    if not np.isfinite(annual_fixed_cost) or annual_fixed_cost < 0:
        raise ValueError("Annual fixed cost must be finite and nonnegative")
    started = time.monotonic()
    code_fingerprint = research_code_fingerprint()
    root = Path(output)
    registry = TrialRegistry(root / "trials", stage=PROTOCOL)
    configs = {f"{rule}_cost{mult}": {"protocol": PROTOCOL, "rule": rule, "cost_multiplier": mult,
                "config_hash": cfg.hash(), "research_code_sha256": code_fingerprint} for rule in RULES for mult in MULTIPLIERS}
    configs.update({rule: {"protocol": PROTOCOL, "rule": rule, "config_hash": cfg.hash(), "research_code_sha256": code_fingerprint}
                    for rule in ("cash", "synthetic_benchmark")})
    for config in configs.values():
        registry.log(config, status="planned")
    try:
        full = bundle if bundle is not None else load_bundle(cfg)
        data, boundary = research_only(full, cfg)
        if data.meta.get("source") != "synthetic":
            raise ValueError("Injected data contradicts the synthetic-only protocol")
        fingerprint = dataset_fingerprint(data)
        panel = build_features(data, cfg)
        X = panel.features(list(cfg.features.sets[cfg.models.feature_set]))
        labels = forward_returns(data, panel.dates, 1, cfg.labels.benchmark,
                                 cfg.data.default_delisting_return, panel.universe)
        targets = label_panel(labels, X.index)
        wf = walk_forward(X, targets, ["logit"], cfg, keep_models=True)
        if wf.preds.empty:
            raise ValueError("No out-of-sample predictions")
        predictions = model_view(wf.preds, "logit")
        aligned = targets.reindex(predictions.index)
        forecast = forecast_diagnostics(predictions, aligned)
        if forecast["accuracy"] > float(cfg.validation.red_flags.max_accuracy):
            raise ValueError("Accuracy leakage red flag; comparison halted")
        factors = (panel.X.mom_12_1 - panel.X.vol_252) / 2
        scores = {"current_logit": rank_diagnostics(predictions.score, aligned),
                  "transparent_factor": rank_diagnostics(factors.reindex(predictions.index), aligned)}
    except Exception as exc:
        for config in configs.values():
            registry.log(config, status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    results, returns, primary = {}, {}, None
    for name, config in configs.items():
        try:
            if config["rule"] in RULES:
                monthly = backtest(config["rule"], config["cost_multiplier"], predictions, factors, panel, labels, data, cfg)
                if primary is None:
                    primary = monthly
            else:
                monthly = primary.copy()
                monthly["net"] = monthly.rf if name == "cash" else monthly.bench
                monthly["gross"] = monthly.net
                monthly[["cost", "turnover", "max_weight", "max_sector", "herfindahl"]] = 0.
                monthly["unfilled_entry_targets"] = 0
                monthly["exposure"] = 0. if name == "cash" else 1.
                monthly["equity"] = float(cfg.capital.initial) * (1 + monthly.net).cumprod()
            metrics = summarize(monthly, annual_fixed_cost)
            if config["rule"] in RULES and metrics["sharpe"] > float(cfg.validation.red_flags.max_sharpe):
                raise ValueError("Sharpe leakage red flag; comparison halted")
            results[name] = metrics
            returns[name] = monthly.net
            registry.log(config, monthly.net - monthly.rf, metrics)
        except Exception as exc:
            registry.log(config, status="failed", error=type(exc).__name__ + ": " + str(exc))
            raise
    diagnostic = pbo(registry.returns(), **dict(cfg.validation.pbo))
    folds = [{"predict_from": str(date.date()), "train_first_decision": str(models["logit"].train_start.date()),
              "train_last_decision": str(models["logit"].train_end.date()), "fit_rows": models["logit"].n_train,
              "purging": "Training label end <= predict_from; fit labels strictly precede calibration start"}
              for date, models in wf.models.items()]
    report = _clean_json({"protocol": PROTOCOL, "synthetic_only": True, "config_hash": cfg.hash(),
        "research_code_sha256": code_fingerprint,
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "config_snapshot": cfg.to_dict(), "research_input_sha256": fingerprint,
        "research_start": str(data.dates[0].date()), "research_end": str(data.dates[-1].date()),
        "excluded_holdout_start": str(boundary.date()), "holdout_opened": False, "folds": folds,
        "n_tickers": len(data.tickers), "n_oos_predictions": len(predictions),
        "new_configurations": registry.n_trials(PROTOCOL), "completed_configurations_this_run": len(results),
        "failed_attempt_records": sum(r["status"] == "failed" for r in registry.records(PROTOCOL)),
        "known_previous_attempts_at_least": 50,
        "forecast": forecast, "ranking": scores, "investment": results,
        "current_matrix_pbo_diagnostic": {k: v for k, v in diagnostic.items() if k != "logits"},
        "combined_dsr": None, "combined_pbo": None,
        "multiple_testing_limitation": "Original complete trial matrix absent; diagnostic subset PBO cannot certify improvement or promotion.",
        "conventions": {"currency": "USD", "return_type": "synthetic adjusted total-return approximation",
            "annualization": "12 equally weighted monthly open-to-open periods", "risk_free": "synthetic annual rate at signal date / 12",
            "drawdown": "month-end only; may understate intramonth loss", "costs": "commission + spread + slippage + square-root impact on gross traded notional",
            "benchmark": "Same-window synthetic cap-weighted index; frictionless, not a verified investable ETF",
            "taxes": "not modeled", "fixed_cost": "illustrative input separate from trading-cost returns",
            "model": "current configured logit monthly selection; no daily stops/kill switch in this diagnostic"},
        "runtime_seconds": time.monotonic() - started, "decision": "No promotion: synthetic-only tooling comparison; retain current defaults."})
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(returns).to_csv(root / "named_returns.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/research_comparison")
    parser.add_argument("--annual-fixed-cost", type=float, default=600.)
    args = parser.parse_args()
    report = run(load_config(args.config), args.output, annual_fixed_cost=args.annual_fixed_cost)
    print(json.dumps({"output": args.output, "runtime_seconds": report["runtime_seconds"],
                      "configurations": report["new_configurations"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
