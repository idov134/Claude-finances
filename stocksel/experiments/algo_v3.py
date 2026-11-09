"""Algorithm research v3: mandate frontier (relative + regime filter) and the gate-power diagnostic.

Protocol: docs/algo_research_v3/PROTOCOL.md (committed before the run). Run once:

    python -m stocksel.experiments.algo_v3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.monthly import monthly_backtest
from ..config import load_config
from ..models.ensemble import prob_outperform
from ..models.train import model_view, walk_forward
from ..validation.history import load_history, trial_sharpes
from ..validation.metrics import classification_metrics, deflated_sharpe, sharpe, stationary_bootstrap_idx
from ..validation.trials import TrialRegistry
from . import algo_v2 as v2
from .reentry import _clean_json

PROTOCOL = "algo_research_v3"
OUT = v2.REPO / "docs" / "algo_research_v3"
DEV_SEED = 7
CONFIRM_SEEDS = (404, 505, 606)
RUNS = {"B0": {}, "E5_relative": {"portfolio.mandate": "relative"},
        "M1_relative_regime": {"portfolio.mandate": "relative", "portfolio.regime_filter.enabled": True}}


def excess_pooled(series: list[np.ndarray]) -> dict:
    rng = np.random.default_rng(v2.BOOT["seed"])
    stats = np.array([np.mean([s[stationary_bootstrap_idx(len(s), v2.BOOT["block"], rng)].mean() for s in series])
                      for _ in range(v2.BOOT["n"])])
    return {"excess_ann": float(np.mean([s.mean() for s in series]) * 12),
            "ci_low": float(np.quantile(stats, 0.025) * 12), "ci_high": float(np.quantile(stats, 0.975) * 12),
            "p_positive": float((stats > 0).mean())}


def run(cfg) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "results.json"
    if out_path.exists():
        raise RuntimeError("v3 already ran once; re-running would turn its fresh worlds into tuning data")
    reg = TrialRegistry(OUT, stage=PROTOCOL)
    tol = float(cfg.capital.max_drawdown)
    worlds, rows, rets, ex_series, dd_ok, flags = {}, {}, {}, [], [], []
    for seed in (DEV_SEED, *CONFIRM_SEEDS):
        tag = f"seed{seed}"
        v2.log(f"v3 world {tag}")
        ev = v2.Evaluator(cfg, v2.world(cfg, seed), tag)
        res = {}
        for name, ov in RUNS.items():
            conf = {"protocol": PROTOCOL, "name": name, "world": tag, "overrides": ov, "base": v2.BASE,
                    "config_hash": cfg.hash()}
            res[name] = v2._attempt(reg, conf, lambda ov=ov: ev.run(ov), returns_to_registry=(seed == DEV_SEED))
        rows[tag] = [v2.summary_row(n, r, v2.paired(r["monthly"], res["B0"]["monthly"]) if n != "B0" else None)
                     for n, r in res.items()]
        m1 = res["M1_relative_regime"]
        dd_ok.append(m1["perf"]["max_drawdown"] >= -tol)
        if (m1["cls"].get("accuracy") or 0) > v2.RULE["max_accuracy"] or (m1["perf"]["sharpe"] or 0) > v2.RULE["max_sharpe"]:
            flags.append(tag)
        if seed != DEV_SEED:
            mm = m1["monthly"]
            ex_series.append((mm["net"] - mm["bench"]).to_numpy())
        for n, r in res.items():
            rets[f"{tag}:{n}"] = r["monthly"]["net"]

    pooled = excess_pooled(ex_series)
    adopt = pooled["excess_ann"] > 0 and pooled["p_positive"] >= 0.90 and all(dd_ok) and not flags
    decision = {"adopt_M1": bool(adopt), "pooled_M1_excess_vs_benchmark": pooled,
                "max_drawdown_within_tolerance_all_worlds": bool(all(dd_ok)), "tolerance": tol,
                "red_flag_worlds": flags,
                "reason": "passed" if adopt else ("drawdown tolerance exceeded" if not all(dd_ok)
                                                  else "excess not established")}

    # ---------------------------------------------------------------- D1b gate-power diagnostic
    v2.log("v3 diagnostic D1b (interaction world)")
    dw = v2.world(cfg, DEV_SEED, interaction=1.0)
    ev = v2.Evaluator(cfg, dw, "seed7_interaction1")
    c = cfg.copy_with({**v2.BASE, "models.candidates": ["logit", "lgbm_tuned"]})
    panel = ev._panel(c)
    lab, L = ev._labels(c, panel)
    feats = [f for f in c.features.sets[c.models.feature_set] if f in panel.X.columns]
    conf = {"protocol": PROTOCOL, "name": "D1b_standalone_logit_vs_lgbm_tuned", "world": "seed7_interaction1",
            "config_hash": cfg.hash()}
    reg.log(conf, status="planned")
    wf = walk_forward(panel.features(feats), L, ["logit", "lgbm_tuned"], c, keep_models=False)
    prm = ev._prm(c, feats)
    d1b = {}
    nets = {}
    for mname in ("logit", "lgbm_tuned"):
        v = model_view(wf.preds, mname)
        Lv = L.reindex(v.index)
        bt = monthly_backtest(v, panel, lab, dw, c, prm)
        nets[mname] = bt.net
        cls = classification_metrics(v["prob"], Lv["y"], Lv["excess"], v.index.get_level_values(0))
        d1b[mname] = {"auc": cls.get("auc"), "ic_mean": cls.get("ic_mean"), "ic_tstat": cls.get("ic_tstat"),
                      "log_loss": cls.get("log_loss"), "net_sharpe": sharpe(bt.net - bt.monthly["rf"]),
                      "net_ann": float(bt.net.mean() * 12)}
    gate_path = []
    for R in wf.schedule:
        realized = lab.exit.reindex(nets["logit"].index) <= R
        diff = (nets["lgbm_tuned"] - nets["logit"]).loc[realized[realized].index].dropna()
        gate_path.append({"retrain": str(pd.Timestamp(R).date()), "months": int(len(diff)),
                          "mean_diff_ann": float(diff.mean() * 12) if len(diff) else None,
                          "p_outperform": prob_outperform(diff.to_numpy(), int(cfg.models.gate.block_size))
                          if len(diff) >= 6 else None})
    reg.log(conf, None, {"logit_auc": d1b["logit"]["auc"], "lgbm_auc": d1b["lgbm_tuned"]["auc"]})

    hist = load_history()
    dev_m1 = [r for r in rows[f"seed{DEV_SEED}"] if r["config"] == "M1_relative_regime"]
    out = {"protocol": PROTOCOL, "worlds": {"dev": DEV_SEED, "confirm": list(CONFIRM_SEEDS)}, "rows": rows,
           "decision": decision, "diagnostic_D1b": {"standalone": d1b, "gate_path": gate_path,
                                                     "gate_threshold": float(cfg.models.gate.min_prob_outperform)},
           "history": {"n_trials": hist["n_trials"], "by_source": hist["by_source"]},
           "runtime_sec": time.time() - t0}
    out_path.write_text(json.dumps(_clean_json(out), indent=2))
    pd.DataFrame(rets).to_csv(OUT / "net_returns_all_worlds.csv")
    v2.log(f"v3 done in {out['runtime_sec']:.0f}s; decision={decision}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stocksel.experiments.algo_v3")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.data.provider != "synthetic":
        raise SystemExit("this protocol is synthetic-only")
    run(cfg)


if __name__ == "__main__":
    main()
