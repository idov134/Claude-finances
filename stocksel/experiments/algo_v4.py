"""Algorithm research v4: probability centering (the cash-collapse fix). Pre-registered in
docs/algo_research_v4/PROTOCOL.md. Run once:

    python -m stocksel.experiments.algo_v4            # synthetic worlds (decisive)
    python -m stocksel.experiments.algo_v4 --real-posthoc   # after the real v1.1 run (labelled, non-decisive)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from ..backtest.monthly import monthly_backtest
from ..config import load_config, vol_target
from ..data import load_bundle
from ..features.news import make_scorer
from ..features.pipeline import build_features
from ..models.labels import forward_returns
from ..portfolio.sizing import SizingParams
from ..validation.metrics import performance_table
from ..validation.trials import TrialRegistry
from . import algo_v2 as v2
from .reentry import _clean_json

PROTOCOL = "algo_research_v4"
OUT = v2.REPO / "docs" / "algo_research_v4"
STANDARD = (1101, 1202, 1303)
BREADTH = (2404, 2505, 2606)
BREADTH_LEVEL = 2.0
RUNS = {"B0": {}, "E7_prob_centering": {"portfolio.prob_centering": "cross_section"}}


def cash_months(m: pd.DataFrame) -> int:
    return int((m["exposure"] < 0.10).sum())


def run(cfg) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "results.json"
    if out_path.exists():
        raise RuntimeError("v4 already ran once")
    reg = TrialRegistry(OUT, stage=PROTOCOL)
    rows, diffs, rets, dd_ok, flags = {}, {"standard": [], "breadth": []}, {}, [], []
    plan = [(v2.DEV_SEED, 0.0, "dev"), *[(s, 0.0, "standard") for s in STANDARD],
            *[(s, BREADTH_LEVEL, "breadth") for s in BREADTH]]
    for seed, br, group in plan:
        tag = f"seed{seed}" + (f"_breadth{br:g}" if br else "")
        v2.log(f"v4 world {tag} ({group})")
        ev = v2.Evaluator(cfg, v2.world(cfg, seed, breadth=br), tag)
        res = {}
        for name, ov in RUNS.items():
            conf = {"protocol": PROTOCOL, "name": name, "world": tag, "overrides": ov, "base": v2.BASE,
                    "config_hash": cfg.hash()}
            res[name] = v2._attempt(reg, conf, lambda ov=ov: ev.run(ov), returns_to_registry=(seed == v2.DEV_SEED))
        b, e = res["B0"], res["E7_prob_centering"]
        rows[tag] = []
        for n, r in res.items():
            row = v2.summary_row(n, r, v2.paired(r["monthly"], b["monthly"]) if n != "B0" else None)
            row.update(cash_months=cash_months(r["monthly"]), group=group)
            rows[tag].append(row)
            rets[f"{tag}:{n}"] = r["monthly"]["net"]
        if group in diffs:
            j = pd.concat({"e": e["monthly"]["net"], "b": b["monthly"]["net"]}, axis=1).dropna()
            diffs[group].append((j["e"] - j["b"]).to_numpy())
        dd_ok.append(e["perf"]["max_drawdown"] >= b["perf"]["max_drawdown"] - 0.03)
        if (e["cls"].get("accuracy") or 0) > v2.RULE["max_accuracy"] or (e["perf"]["sharpe"] or 0) > v2.RULE["max_sharpe"]:
            flags.append(tag)
    std, brd = v2.pooled(diffs["standard"]), v2.pooled(diffs["breadth"])
    wins = int(sum(d.mean() > 0 for d in diffs["breadth"]))
    checks = {"no_harm_standard": std["ci_low"] > -0.01, "mechanism_breadth": brd["p_outperform"] >= 0.90 and wins >= 2,
              "drawdown_margin": bool(all(dd_ok)), "no_red_flags": not flags}
    decision = {"adopt_E7": bool(all(checks.values())), "checks": checks, "pooled_standard": std,
                "pooled_breadth": brd, "breadth_worlds_positive": wins, "red_flag_worlds": flags}
    out = {"protocol": PROTOCOL, "rows": rows, "decision": decision, "breadth_level": BREADTH_LEVEL,
           "runtime_sec": time.time() - t0}
    out_path.write_text(json.dumps(_clean_json(out), indent=2))
    pd.DataFrame(rets).to_csv(OUT / "net_returns_all_worlds.csv")
    v2.log(f"v4 done in {out['runtime_sec']:.0f}s; decision={decision}")
    return out


def real_posthoc() -> dict:
    """Labelled post hoc: E7 vs default on the real research window. Never the holdout."""
    from .real_firstlook import OVERRIDES
    cfg = load_config("config.yaml", OVERRIDES)
    res = pickle.loads((v2.REPO / cfg.paths.artifacts / "research_result.pkl").read_bytes())
    full = load_bundle(cfg)
    hstart = pd.Timestamp(res["hstart"])
    data = full.truncate(hstart - pd.Timedelta(days=1))
    panel = build_features(data, cfg, make_scorer(cfg))
    lab = forward_returns(data, panel.dates, 1, cfg.labels.benchmark, cfg.data.default_delisting_return, panel.universe)
    preds = res["final_preds"]
    keep, vt = res["spec"]["keep"], res["spec"]["vol_target"]
    reg = TrialRegistry(v2.REPO / cfg.paths.artifacts / "trials", stage="research")
    out = {}
    for name, ov in {"default": {}, "E7_prob_centering": {"portfolio.prob_centering": "cross_section"}}.items():
        c2 = cfg.copy_with(ov)
        bt = monthly_backtest(preds, panel, lab, data, c2, SizingParams.from_cfg(c2, keep, vt))
        m = performance_table(bt.net, bt.bench, bt.monthly["rf"], boot={"n": 2000, "block": 6, "seed": 3})
        m.update(cash_months=cash_months(bt.monthly), exposure_mean=float(bt.monthly["exposure"].mean()))
        if name != "default":
            reg.log({"posthoc": PROTOCOL, "name": name, "overrides": ov}, (bt.net - bt.monthly["rf"]).rename(None), m,
                    tags={"posthoc": True, "decisive": False})
        out[name] = m
    path = OUT / "real_posthoc.json"
    path.write_text(json.dumps(_clean_json({"label": "POST HOC on data that revealed the flaw; not evidence",
                                            "window": [str(lab.entry.index.min().date()), str(hstart.date())],
                                            "results": out}), indent=2))
    print(json.dumps(_clean_json(out), indent=1)[:3000])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stocksel.experiments.algo_v4")
    ap.add_argument("--real-posthoc", action="store_true")
    args = ap.parse_args(argv)
    if args.real_posthoc:
        real_posthoc()
    else:
        run(load_config("config.yaml"))


if __name__ == "__main__":
    main()
