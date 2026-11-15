"""Real data + free point-in-time fundamentals (SEC XBRL). Pre-registered in docs/real_fundamentals/PROTOCOL.md.

    python -m stocksel.data.build_sec_fundamentals --data data_csv
    python -m stocksel.experiments.real_fundamentals      # run once
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..data import load_bundle
from ..validation.decay import signal_ic
from ..validation.history import trial_sharpes
from ..validation.holdout import holdout_start
from ..validation.metrics import alpha_returns, classification_metrics, deflated_sharpe, performance_table
from ..validation.trials import TrialRegistry
from . import algo_v2 as v2
from .real_firstlook import OVERRIDES as FIRSTLOOK, PRICE_ONLY
from .reentry import _clean_json

PROTOCOL = "real_fundamentals_v1"
OUT = v2.REPO / "docs" / "real_fundamentals"
FUND = ["value_bm", "value_ep", "quality_gpa", "quality_roe", "sue", "pead"]
OVERRIDES = {**FIRSTLOOK, "paths.artifacts": "artifacts/real_fundamentals",
             "data.cache_dir": "artifacts/real_fundamentals/cache",
             "features.sets.price_fund": PRICE_ONLY + FUND}
PRIMARY = ("2014-01-01", "2024-09-30")
RUNS = {"F0_abs": {"models.feature_set": "price_only"},
        "F1_abs": {"models.feature_set": "price_fund"},
        "F0_rel": {"models.feature_set": "price_only", "portfolio.mandate": "relative"},
        "F1_rel": {"models.feature_set": "price_fund", "portfolio.mandate": "relative"}}
BOOT = {"n": 2000, "block": 6, "seed": 3}


def window_stats(res: dict, lo: str, hi: str) -> dict:
    m = res["monthly"].loc[lo:hi]
    p = res["preds"]
    d = p.index.get_level_values(0)
    sel = (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    L = res["labels"].reindex(p.index[sel])
    cls = classification_metrics(p["prob"][sel], L["y"], L["excess"], d[sel])
    perf = performance_table(m["net"], m["bench"], m["rf"], boot=BOOT)
    return {"perf": perf, "cls": cls, "monthly": m}


def main(config: str = "config.yaml") -> dict:
    t0 = time.time()
    cfg = load_config(config, OVERRIDES)
    data_dir = v2.REPO / cfg.data.csv_dir
    if not (data_dir / "pit_fundamentals.flag").exists():
        raise SystemExit("build fundamentals first: python -m stocksel.data.build_sec_fundamentals --data data_csv")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("this study already ran once")
    full = load_bundle(cfg)
    if not full.meta.get("pit_fundamentals"):
        raise SystemExit("fundamentals were not loaded as point-in-time")
    hstart = holdout_start(full.dates, int(cfg.validation.holdout_months))
    research = full.truncate(hstart - pd.Timedelta(days=1))
    reg = TrialRegistry(OUT, stage=PROTOCOL)
    ev = v2.Evaluator(cfg, research, "real_research")
    res = {}
    for name, ov in RUNS.items():
        v2.log(f"real fundamentals: {name}")
        conf = {"protocol": PROTOCOL, "name": name, "overrides": ov, "base": v2.BASE, "config_hash": cfg.hash()}
        res[name] = v2._attempt(reg, conf, lambda ov=ov: ev.run(ov), returns_to_registry=True)

    prim = {n: window_stats(r, *PRIMARY) for n, r in res.items()}
    full_w = {n: {"perf": r["perf"], "cls": r["cls"]} for n, r in res.items()}
    paired = {"abs": v2.paired(prim["F1_abs"]["monthly"], prim["F0_abs"]["monthly"]),
              "rel": v2.paired(prim["F1_rel"]["monthly"], prim["F0_rel"]["monthly"])}

    # alpha DSR charged with every real-data trial (first look v1 + v1.1) plus this study
    fl = v2.REPO / "artifacts" / "real_firstlook" / "trials"
    fl_rets = pd.read_csv(fl / "returns.csv", index_col=0, parse_dates=True) if (fl / "returns.csv").exists() \
        else pd.DataFrame()
    fl_n = len({json.loads(l)["trial_id"] for l in (fl / "registry.jsonl").read_text().splitlines() if l.strip()}) \
        if (fl / "registry.jsonl").exists() else 52
    n_total = max(52, fl_n) + reg.n_trials(PROTOCOL)
    allR = pd.concat([fl_rets, reg.returns()], axis=1)
    m1 = prim["F1_abs"]["monthly"]
    b_ex = (m1["bench"] - m1["rf"])
    RA = allR.reindex(m1.index).apply(lambda c: alpha_returns(c, b_ex)).dropna(axis=1, how="all")

    def _alpha(n):
        m = prim[n]["monthly"]
        return alpha_returns(m["net"] - m["rf"], m["bench"] - m["rf"]).dropna().to_numpy()
    dsr_alpha = {n: deflated_sharpe(_alpha(n), n_total, trial_sharpes(RA))["dsr"] for n in res}
    c1 = prim["F1_abs"]["cls"]
    crit = {"ic_and_auc": bool((c1.get("ic_tstat") or 0) >= 2.0 and (c1.get("auc") or 0) > 0.505),
            "beats_price_only": bool(paired["abs"]["delta_ann"] > 0 and paired["abs"]["p_outperform"] >= 0.90),
            "alpha_dsr": bool((dsr_alpha["F1_abs"] or 0) >= 0.95)}
    outcome = ("skill established" if all(crit.values()) else
               "signal present, not significant after testing" if crit["ic_and_auc"] else "no skill")

    # single-signal IC on real data (research window, 2010+ where fundamentals exist)
    panel = ev.panels["none"]
    L = res["F1_abs"]["labels"]
    ic = signal_ic(panel.X, L["excess"])
    ic10 = ic.loc["2010-01-01":]
    single = {c: {"ic_mean": float(ic10[c].mean()), "ic_t": float(ic10[c].mean() / ic10[c].std() * np.sqrt(ic10[c].count()))}
              for c in ic10.columns if c in PRICE_ONLY + FUND}

    # post-2024 period: viewed in the first look, NOT a clean holdout
    v2.log("post-2024 period (previously viewed)")
    evf = v2.Evaluator(cfg, full, "real_full")
    post = {}
    for name in ("F0_abs", "F1_abs", "F0_rel", "F1_rel"):
        r = evf.run(RUNS[name])
        post[name] = window_stats(r, str(hstart.date()), "2026-12-31")
    coverage = {c: float(panel.raw[c].where(panel.universe).loc["2014":].notna().sum().sum() /
                         panel.universe.loc["2014":].sum().sum()) for c in FUND if c in panel.raw}
    strip = lambda d: {k: v for k, v in d.items() if k != "monthly"}
    out = {"protocol": PROTOCOL, "primary_window": PRIMARY, "decision": {"criteria": crit, "outcome": outcome},
           "primary": {n: strip(v) for n, v in prim.items()}, "full_oos": full_w, "paired_primary": paired,
           "alpha_dsr_primary": dsr_alpha, "n_trials_charged": n_total, "single_signal_ic_2010_2024": single,
           "fundamental_coverage_2014_on": coverage,
           "post_2024_previously_viewed": {n: strip(v) for n, v in post.items()},
           "sec_info": json.loads((data_dir / "sec_info.json").read_text()), "runtime_sec": time.time() - t0}
    (OUT / "results.json").write_text(json.dumps(_clean_json(out), indent=2, default=str))
    v2.log(f"done in {out['runtime_sec']:.0f}s; outcome={outcome} criteria={crit}")
    return out


if __name__ == "__main__":
    main()
