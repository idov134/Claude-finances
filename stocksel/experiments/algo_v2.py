"""Algorithm research v2: pre-registered single-change experiments + independent-world confirmation.

Protocol: docs/algo_research_v2/PROTOCOL.md (committed before any run).

    python -m stocksel.experiments.algo_v2 dev        # development world (seed 7, research window only)
    python -m stocksel.experiments.algo_v2 confirm    # once, after the development decision is committed

Never opens a holdout, never trades. Every planned, completed and failed configuration is written
to docs/algo_research_v2/registry.jsonl. Results are synthetic engineering evidence only.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.monthly import monthly_backtest
from ..config import load_config, vol_target
from ..data import synthetic
from ..features.news import LexiconScorer
from ..features.pipeline import build_features
from ..features.registry import haircut_factor
from ..models.ensemble import ensemble_predictions, gate_decisions
from ..models.labels import forward_returns, label_panel
from ..models.train import model_view, smooth_predictions, walk_forward
from ..portfolio.sizing import SizingParams
from ..validation.history import load_history, trial_sharpes
from ..validation.holdout import assert_research_data
from ..validation.metrics import (classification_metrics, deflated_sharpe, max_drawdown, performance_table,
                                  sharpe, stationary_bootstrap_idx)
from ..validation.pbo import pbo
from ..validation.trials import TrialRegistry
from .reentry import _clean_json

PROTOCOL = "algo_research_v2"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "algo_research_v2"
RESEARCH_END = pd.Timestamp("2023-12-29")      # the 2024-2025 synthetic holdout is used; never touched
DEV_SEED = 7
CONFIRM_SEEDS = (101, 202, 303)
BOOT = {"n": 2000, "block": 6, "seed": 3}

# The effective default: the online gate admitted only these three in all 21 original windows.
BASE = {"models.candidates": ["logit", "ridge", "enet"]}
SINGLE_CHANGES = {
    "E1_sector_neutral": {"features.neutralize": "sector"},
    "E2_residual_momentum": {"models.feature_set": "core_comp_resid"},
    "E3_tails_target": {"labels.target": "tails", "labels.tail_frac": 0.3},
    "E4_score_smoothing": {"portfolio.score_smoothing": 0.5},
    "E6_tuned_lgbm": {"models.candidates": ["logit", "ridge", "enet", "lgbm_tuned"]},
}
MANDATE_ALTERNATIVE = {"E5_relative_mandate": {"portfolio.mandate": "relative"}}
ADOPTABLE = ("E1_sector_neutral", "E2_residual_momentum", "E3_tails_target", "E4_score_smoothing", "E6_tuned_lgbm")
RULE = {"p_outperform": 0.90, "max_dd_worse_pp": 2.0, "max_accuracy": 0.65, "max_sharpe": 2.0}


def log(msg: str) -> None:
    print(f"[algo_v2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- worlds
def world(cfg, seed: int, interaction: float = 0.0, breadth: float = 0.0):
    s = cfg.data.synthetic
    full = synthetic.generate(start=cfg.data.start, end=cfg.data.end, seed=seed, n_initial=s.n_initial,
                              ipo_per_year=s.ipo_per_year, annual_delist_prob=s.annual_delist_prob,
                              alpha_scale=s.alpha_scale, post_2005_decay=s.post_2005_decay,
                              news_per_month=s.news_per_month, timezone=cfg.market.timezone,
                              interaction=interaction, breadth=breadth)
    data = full.truncate(RESEARCH_END)
    assert_research_data(data, RESEARCH_END + pd.Timedelta(days=1))
    return data


class Evaluator:
    """Caches features / labels / walk-forward predictions so each change re-fits only what it changes."""

    def __init__(self, cfg, data, world_tag: str):
        self.cfg, self.data, self.tag = cfg, data, world_tag
        self.panels, self.labels, self.preds = {}, {}, {}

    def _panel(self, c):
        key = c.features.get("neutralize", "none")
        if key not in self.panels:
            log(f"[{self.tag}] features (neutralize={key})")
            self.panels[key] = build_features(self.data, c, LexiconScorer())
        return self.panels[key]

    def _labels(self, c, panel):
        key = c.features.get("neutralize", "none")
        if key not in self.labels:
            lab = forward_returns(self.data, panel.dates, 1, c.labels.benchmark, c.data.default_delisting_return,
                                  panel.universe)
            self.labels[key] = (lab, label_panel(lab, panel.X.index))
        return self.labels[key]

    def _prm(self, c, feats):
        return SizingParams.from_cfg(c, haircut_factor(feats, c), vol_target(c))

    def _predictions(self, c, panel, lab, L):
        feats = [f for f in c.features.sets[c.models.feature_set] if f in panel.X.columns]
        cands = list(c.models.candidates)
        key = (c.features.get("neutralize", "none"), c.models.feature_set, c.labels.get("target", "binary"),
               float(c.labels.get("tail_frac", 0.3)), tuple(cands))
        if key not in self.preds:
            log(f"[{self.tag}] walk-forward {key}")
            wf = walk_forward(panel.features(feats), L, cands, c, keep_models=False)
            prm = self._prm(c, feats)
            nets = {m: monthly_backtest(model_view(wf.preds, m), panel, lab, self.data, c, prm).net for m in cands}
            dec = gate_decisions(nets, lab.exit, wf.schedule, c)
            self.preds[key] = (ensemble_predictions(wf.preds, dec, cands), dec, feats)
        return self.preds[key]

    def run(self, overrides: dict) -> dict:
        c = self.cfg.copy_with({**BASE, **overrides})
        panel = self._panel(c)
        lab, L = self._labels(c, panel)
        preds, dec, feats = self._predictions(c, panel, lab, L)
        preds = smooth_predictions(preds, float(c.portfolio.get("score_smoothing", 0.0) or 0.0))
        bt = monthly_backtest(preds, panel, lab, self.data, c, self._prm(c, feats),
                              regime_filter=bool(c.portfolio.regime_filter.enabled))
        m = bt.monthly
        perf = performance_table(m["net"], m["bench"], m["rf"], boot=BOOT)
        perf.update(turnover_ann=float(m["turnover"].mean() * 12), cost_drag_ann=float(m["cost"].mean() * 12),
                    exposure_mean=float(m["exposure"].mean()), unfilled_orders=int(m["unfilled"].sum()))
        Lp = L.reindex(preds.index)
        cls = classification_metrics(preds["prob"], Lp["y"], Lp["excess"], preds.index.get_level_values(0))
        admitted = pd.DataFrame(dec).T
        return {"monthly": m, "perf": perf, "cls": cls, "features": feats,
                "gate_admission_share": admitted.mean().to_dict() if len(admitted) else {}}


# --------------------------------------------------------------------------- statistics
def paired(m_new: pd.DataFrame, m_base: pd.DataFrame) -> dict:
    j = pd.concat({"n": m_new["net"], "b": m_base["net"]}, axis=1).dropna()
    diff = (j["n"] - j["b"]).to_numpy()
    rng = np.random.default_rng(BOOT["seed"])
    draws = np.array([diff[stationary_bootstrap_idx(len(diff), BOOT["block"], rng)].mean() for _ in range(BOOT["n"])])
    return {"delta_ann": float(diff.mean() * 12), "ci_low": float(np.quantile(draws, 0.025) * 12),
            "ci_high": float(np.quantile(draws, 0.975) * 12), "p_outperform": float((draws > 0).mean()),
            "months": int(len(diff))}


def pooled(diffs: list[np.ndarray]) -> dict:
    rng = np.random.default_rng(BOOT["seed"])
    stats = []
    for _ in range(BOOT["n"]):
        stats.append(np.mean([d[stationary_bootstrap_idx(len(d), BOOT["block"], rng)].mean() for d in diffs]))
    stats = np.array(stats)
    point = float(np.mean([d.mean() for d in diffs]))
    return {"delta_ann": point * 12, "ci_low": float(np.quantile(stats, 0.025) * 12),
            "ci_high": float(np.quantile(stats, 0.975) * 12), "p_outperform": float((stats > 0).mean())}


def passes(res: dict, base: dict, cmp: dict) -> tuple[bool, list[str]]:
    why = []
    if not (cmp["delta_ann"] > 0 and cmp["p_outperform"] >= RULE["p_outperform"]):
        why.append(f"net improvement not established (P={cmp['p_outperform']:.2f} < {RULE['p_outperform']})")
    if not (res["perf"]["sharpe"] >= base["perf"]["sharpe"]):
        why.append("Sharpe below baseline")
    if res["perf"]["max_drawdown"] < base["perf"]["max_drawdown"] - RULE["max_dd_worse_pp"] / 100:
        why.append("max drawdown worse by more than 2 pp")
    if (res["cls"].get("accuracy") or 0) > RULE["max_accuracy"] or (res["perf"]["sharpe"] or 0) > RULE["max_sharpe"]:
        why.append("red flag: accuracy > 65% or Sharpe > 2")
    return (not why), why


def summary_row(name, res, cmp=None, dsr=None):
    p, c = res["perf"], res["cls"]
    row = {"config": name, "cagr": p["cagr"], "bench_cagr": p["bench_cagr"], "excess_ann": p["excess_ann"],
           "excess_ci_low": p["excess_ci_low"], "excess_ci_high": p["excess_ci_high"], "sharpe": p["sharpe"],
           "sortino": p["sortino"], "max_drawdown": p["max_drawdown"], "calmar": p["calmar"],
           "information_ratio": p["information_ratio"], "capm_beta": p["capm_beta"],
           "turnover_ann": p["turnover_ann"], "cost_drag_ann": p["cost_drag_ann"], "exposure": p["exposure_mean"],
           "hit_rate_months": p["hit_rate_periods"], "auc": c.get("auc"), "accuracy": c.get("accuracy"),
           "ic_mean": c.get("ic_mean"), "log_loss": c.get("log_loss")}
    if cmp:
        row.update({f"vs_B0_{k}": v for k, v in cmp.items() if k != "months"})
    if dsr is not None:
        row["dsr_full_history"] = dsr
    return row


# --------------------------------------------------------------------------- stages
def _registry():
    OUT.mkdir(parents=True, exist_ok=True)
    return TrialRegistry(OUT, stage=PROTOCOL)


def _conf(name, overrides, world_tag, cfg):
    return {"protocol": PROTOCOL, "name": name, "world": world_tag, "overrides": overrides, "base": BASE,
            "config_hash": cfg.hash()}


def _attempt(reg, conf, fn, returns_to_registry: bool):
    reg.log(conf, status="planned")
    try:
        res = fn()
        m = res["monthly"]
        reg.log(conf, (m["net"] - m["rf"]).rename(None) if returns_to_registry else None,
                {**res["perf"], **{f"cls_{k}": v for k, v in res["cls"].items()}})
        return res
    except Exception as exc:
        reg.log(conf, status="failed", error=f"{exc!r}\n{traceback.format_exc()[-1500:]}")
        raise


def run_dev(cfg) -> dict:
    t0 = time.time()
    reg = _registry()
    data = world(cfg, DEV_SEED)
    ev = Evaluator(cfg, data, f"seed{DEV_SEED}")
    results = {}
    plan = {"B0": {}, **SINGLE_CHANGES, **MANDATE_ALTERNATIVE}
    for name, ov in plan.items():
        log(f"dev: {name}")
        results[name] = _attempt(reg, _conf(name, ov, f"seed{DEV_SEED}", cfg), lambda ov=ov: ev.run(ov), True)
    base = results["B0"]
    cmp = {n: paired(r["monthly"], base["monthly"]) for n, r in results.items() if n != "B0"}
    verdict = {n: passes(results[n], base, cmp[n]) for n in ADOPTABLE}
    passing = [n for n in ADOPTABLE if verdict[n][0]]
    combo = {}
    for n in passing:
        combo.update(SINGLE_CHANGES[n])
    if passing:
        log(f"dev: C1 = {passing}")
        results["C1"] = _attempt(reg, _conf("C1", combo, f"seed{DEV_SEED}", cfg), lambda: ev.run(combo), True)
        cmp["C1"] = paired(results["C1"]["monthly"], base["monthly"])

    # diagnostic D1: can the gate admit a nonlinear model when the market really is nonlinear?
    log("diagnostic D1: interaction world")
    dw = world(cfg, DEV_SEED, interaction=1.0)
    dev_d1 = Evaluator(cfg, dw, f"seed{DEV_SEED}_interaction1")
    d1 = {name: _attempt(reg, _conf(f"D1_{name}", ov, f"seed{DEV_SEED}_interaction1", cfg),
                         lambda ov=ov: dev_d1.run(ov), False)
          for name, ov in {"B0": {}, "E6_tuned_lgbm": SINGLE_CHANGES["E6_tuned_lgbm"]}.items()}
    d1_cmp = paired(d1["E6_tuned_lgbm"]["monthly"], d1["B0"]["monthly"])

    # multiple testing over the whole history (original 48 + Codex + this protocol)
    hist = load_history()
    srs = trial_sharpes(hist["returns"])
    dsr = {n: deflated_sharpe((r["monthly"]["net"] - r["monthly"]["rf"]).to_numpy(), hist["n_trials"], srs)["dsr"]
           for n, r in results.items()}
    common = hist["returns"].dropna(axis=0, how="any") if not hist["returns"].empty else hist["returns"]
    pbo_all = pbo(hist["returns"].dropna(axis=1, how="all").dropna(axis=0, how="any"), 16, 5000)
    rows = [summary_row(n, r, cmp.get(n), dsr.get(n)) for n, r in results.items()]
    out = {"protocol": PROTOCOL, "stage": "dev", "world": f"seed{DEV_SEED}", "research_end": str(RESEARCH_END.date()),
           "rule": RULE, "results": rows, "verdict": {n: {"pass": v[0], "reasons": v[1]} for n, v in verdict.items()},
           "passing": passing, "combination": combo,
           "gate_admission_share": {n: r["gate_admission_share"] for n, r in results.items()},
           "diagnostic_D1": {"rows": [summary_row(f"D1_{n}", r) for n, r in d1.items()], "E6_vs_B0": d1_cmp,
                             "gate_admission_share": d1["E6_tuned_lgbm"]["gate_admission_share"]},
           "history": {"n_trials": hist["n_trials"], "n_completed": hist["n_completed"],
                       "by_source": hist["by_source"], "pbo_full_history": pbo_all.get("pbo"),
                       "pbo_matrix_shape": list(common.shape)},
           "runtime_sec": time.time() - t0}
    (OUT / "dev_results.json").write_text(json.dumps(_clean_json(out), indent=2))
    pd.DataFrame({n: r["monthly"]["net"] for n, r in results.items()}).to_csv(OUT / "dev_net_returns.csv")
    log(f"dev done in {out['runtime_sec']:.0f}s; passing={passing}")
    return out


def run_confirm(cfg, force: bool = False) -> dict:
    t0 = time.time()
    dev_path, out_path = OUT / "dev_results.json", OUT / "confirm_results.json"
    if not dev_path.exists():
        raise RuntimeError("run the development stage first")
    if out_path.exists() and not force:
        raise RuntimeError("confirmation already run once; re-running would turn it into tuning data")
    dev = json.loads(dev_path.read_text())
    combo = dev["combination"]
    reg = _registry()
    per_world, diffs, diffs_rel, flags = {}, [], [], []
    rets = {}
    for seed in CONFIRM_SEEDS:
        tag = f"seed{seed}"
        log(f"confirm: world {tag}")
        ev = Evaluator(cfg, world(cfg, seed), tag)
        runs = {"B0": {}, **({"C1": combo} if combo else {}), **MANDATE_ALTERNATIVE}
        res = {n: _attempt(reg, _conf(n, ov, tag, cfg), lambda ov=ov: ev.run(ov), False) for n, ov in runs.items()}
        rows = [summary_row(n, r, paired(r["monthly"], res["B0"]["monthly"]) if n != "B0" else None)
                for n, r in res.items()]
        per_world[tag] = rows
        for n, r in res.items():
            rets[f"{tag}:{n}"] = r["monthly"]["net"]
        if "C1" in res:
            j = pd.concat({"n": res["C1"]["monthly"]["net"], "b": res["B0"]["monthly"]["net"]}, axis=1).dropna()
            diffs.append((j["n"] - j["b"]).to_numpy())
            p, c = res["C1"]["perf"], res["C1"]["cls"]
            if (c.get("accuracy") or 0) > RULE["max_accuracy"] or (p["sharpe"] or 0) > RULE["max_sharpe"]:
                flags.append(tag)
        jr = pd.concat({"n": res["E5_relative_mandate"]["monthly"]["net"], "b": res["B0"]["monthly"]["net"]},
                       axis=1).dropna()
        diffs_rel.append((jr["n"] - jr["b"]).to_numpy())
    decision = {"adopt_C1": False, "reason": "no candidate passed the development stage"}
    if diffs:
        pc = pooled(diffs)
        wins = int(sum(d.mean() > 0 for d in diffs))
        ok = pc["p_outperform"] >= RULE["p_outperform"] and wins >= 2 and not flags
        decision = {"adopt_C1": bool(ok), "pooled_C1_vs_B0": pc, "worlds_with_positive_delta": wins,
                    "red_flag_worlds": flags,
                    "reason": "passed" if ok else "pooled evidence or world count below the pre-registered rule"}
    out = {"protocol": PROTOCOL, "stage": "confirm", "worlds": list(map(int, CONFIRM_SEEDS)), "combination": combo,
           "per_world": per_world, "decision": decision, "relative_mandate_pooled_vs_B0": pooled(diffs_rel),
           "runtime_sec": time.time() - t0}
    out_path.write_text(json.dumps(_clean_json(out), indent=2))
    pd.DataFrame(rets).to_csv(OUT / "confirm_net_returns.csv")
    log(f"confirm done in {out['runtime_sec']:.0f}s; decision={decision}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stocksel.experiments.algo_v2")
    ap.add_argument("stage", choices=["dev", "confirm"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="re-run confirmation (recorded; contaminates it)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.data.provider != "synthetic":
        raise SystemExit("this protocol is synthetic-only; real data needs its own pre-registration")
    (run_dev if args.stage == "dev" else lambda c: run_confirm(c, args.force))(cfg)


if __name__ == "__main__":
    main()
