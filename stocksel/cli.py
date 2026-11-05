"""Command line: python -m stocksel {research,holdout,recommend,paper,monitor}."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

from .config import load_config


def _out(cfg) -> Path:
    p = Path(cfg.report.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_research(cfg, args):
    from .pipeline import run_research
    from .reports.html import write_report
    res = run_research(cfg, quick=args.quick)
    path = write_report(res, _out(cfg) / "research_report.html")
    print(f"report: {path}")
    for f in res["red_flags"]:
        print(f"  [{f['level']}] {f['message']}")


def cmd_holdout(cfg, args):
    from .pipeline import run_holdout
    from .reports.html import write_report
    res = run_holdout(cfg, force=args.force_reopen)
    m = res["final_metrics"]
    from .validation.readiness import holdout_passed
    passed = holdout_passed(res, cfg)
    spec_path = Path(cfg.paths.artifacts) / "final_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["holdout_passed"] = bool(passed and not res["lock"]["contaminated"])
    spec["validation_protocol"] = "real_pit_holdout_v1"
    spec["holdout_metrics"] = {k: m.get(k) for k in ("excess_ann", "sharpe", "bench_sharpe", "max_drawdown")}
    spec_path.write_text(json.dumps(spec, indent=2, default=str))
    path = write_report(res, _out(cfg) / "holdout_report.html")
    print(f"holdout report: {path} | passed={spec['holdout_passed']}")


def cmd_recommend(cfg, args):
    from .live.paper import PaperLedger
    from .live.recommend import recommend
    prev = None
    if args.use_paper:
        led = PaperLedger(cfg)
        if led.state["positions"]:
            prev = pd.Series({t: 0.0 for t in led.state["positions"]})
    recs, meta = recommend(cfg, asof=args.asof, prev_weights=prev)
    out = _out(cfg)
    csv = out / f"recommendations_{meta['asof']}.csv"
    recs.to_csv(csv, index=False)
    (out / f"recommendations_{meta['asof']}.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        print(recs.to_string(index=False))
    print(f"\nexposure {meta['gross_exposure']:.1%} | cash {meta['cash_pct']:.1%} | validated={meta['system_validated']}")
    print(f"saved: {csv}")


def cmd_paper(cfg, args):
    from .data import load_bundle
    from .live.paper import PaperLedger
    led = PaperLedger(cfg)
    if args.action == "submit":
        recs = pd.read_csv(args.file)
        led.submit(recs, args.asof)
    elif args.action == "update":
        led.update(load_bundle(cfg, use_cache=False))
    print(json.dumps(led.status(), indent=2))


def cmd_review(cfg, args):
    """Quarterly signal-decay review on the latest data; prints signals that look dead."""
    from .data import load_bundle
    from .features.news import make_scorer
    from .features.pipeline import build_features
    from .models.labels import forward_returns, label_panel
    from .validation.decay import decay_table, signal_ic
    b = load_bundle(cfg, use_cache=False)
    p = build_features(b, cfg, make_scorer(cfg))
    L = label_panel(forward_returns(b, p.dates, 1), p.X.index)
    tab = decay_table(signal_ic(p.X, L["excess"]), cfg, int(cfg.live.decay_review_months), float(cfg.live.dead_signal_tstat))
    print(tab.to_string())
    dead = list(tab.index[tab["status"] != "alive"])
    print(f"\ncandidates for removal (add to features.disabled after review): {dead}")


def cmd_monitor(cfg, args):
    from .live.monitor import feature_drift, kill_switch_check, weekly_comparison
    from .live.paper import PaperLedger
    res = pickle.loads((Path(cfg.paths.artifacts) / "research_result.pkl").read_bytes())
    led = PaperLedger(cfg)
    eq = led.equity()
    if eq.empty:
        print("no paper equity yet")
        return
    wk = weekly_comparison(eq, res["engine_daily"])
    print(wk.tail(12).to_string())
    dec = kill_switch_check(eq, res["monthly"]["net"], cfg)
    print(f"kill switch: {'TRIGGERED' if dec.kill else 'ok'} — {dec.reason}")
    from .data import load_bundle
    from .features.news import make_scorer
    from .features.pipeline import build_features
    b = load_bundle(cfg, use_cache=False)
    p = build_features(b, cfg, make_scorer(cfg))
    recent = p.dates[-3:]
    train = p.dates[(p.dates >= p.dates[-1] - pd.DateOffset(months=123)) & (p.dates < recent[0])]
    pick = lambda ds: {k: v.loc[ds].where(p.universe.loc[ds]) for k, v in p.raw.items()}
    drift = feature_drift(pick(train), pick(recent), float(cfg.live.psi_alert))
    print("\nfeature drift (training window vs last 3 months):")
    print(drift.to_string())
    print(json.dumps(led.status(), indent=2))


def main(argv=None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser(prog="stocksel")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("research"); r.add_argument("--quick", action="store_true")
    h = sub.add_parser("holdout"); h.add_argument("--force-reopen", action="store_true",
                                                  help="re-open a burned holdout (report will be marked CONTAMINATED)")
    rc = sub.add_parser("recommend"); rc.add_argument("--asof", default=None); rc.add_argument("--use-paper", action="store_true")
    p = sub.add_parser("paper"); p.add_argument("action", choices=["submit", "update", "status"])
    p.add_argument("--file"); p.add_argument("--asof")
    sub.add_parser("monitor")
    sub.add_parser("review")
    app = sub.add_parser("app", help="start the local research dashboard (no brokerage execution)")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", default=8000, type=int)
    app.add_argument("--feed", choices=["synthetic", "alpaca", "yfinance"], default=None)
    app.add_argument("--demo", action="store_true", help="deterministic manual synthetic replay in a separate persistent demo account")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "app":
        if args.demo:
            if args.feed not in (None, "synthetic"):
                ap.error("--demo requires the synthetic feed")
            cfg = cfg.copy_with({"data.provider": "synthetic", "data.start": "1998-01-01",
                "data.end": "2025-12-31", "data.synthetic.seed": 7, "live.feed": "synthetic",
                "paths.artifacts": "artifacts/demo_release", "live.ledger_dir": "artifacts/demo_release/paper"})
        import uvicorn
        from .web.app import create_app
        uvicorn.run(create_app(cfg, feed_name=args.feed), host=args.host, port=args.port)
        return
    {"research": cmd_research, "holdout": cmd_holdout, "recommend": cmd_recommend, "paper": cmd_paper,
     "monitor": cmd_monitor, "review": cmd_review}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
