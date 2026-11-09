"""Orchestration: research (walk-forward + full validation), one-shot holdout, final spec."""
from __future__ import annotations

import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.engine import run_daily
from .backtest.monthly import holding_spells, monthly_backtest, trade_stats
from .config import Config, vol_target
from .data import load_bundle
from .features.news import make_scorer
from .features.pipeline import build_features
from .features.registry import EXPECTED_SIGN, haircut_factor
from .models.ensemble import ensemble_predictions, gate_decisions
from .models.explain import ensemble_contributions, mean_abs_importance
from .models.labels import forward_returns, label_panel
from .models.train import model_view, smooth_predictions, walk_forward
from .portfolio.sizing import SizingParams
from .validation import decay, sanity, sensitivity
from .validation.cpcv import cpcv_path_predictions
from .validation.history import load_history, trial_sharpes
from .validation.kfold import purged_kfold_scores
from .validation.holdout import HoldoutLock, assert_research_data, holdout_start
from .validation.metrics import (classification_metrics, deflated_sharpe, effective_trials, performance_table,
                                 sharpe)
from .validation.pbo import pbo
from .validation.redflags import red_flags
from .validation.trials import TrialRegistry


def log(msg: str) -> None:
    print(f"[stocksel {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cfg_h(cfg: Config, h: int) -> Config:
    return cfg.copy_with({"labels.horizon_months": int(h)})


def _sections_hash(cfg: Config) -> str:
    d = cfg.to_dict()
    sub = {k: d[k] for k in ("data", "universe", "features", "signals", "costs", "models", "validation", "news",
                             "execution", "portfolio", "risk", "labels")}
    return Config(sub).hash()


def _trial_metrics(bt, rf_boot) -> dict:
    m = performance_table(bt.net, bt.bench, bt.monthly["rf"], boot=rf_boot)
    m["turnover_ann"] = float(bt.monthly["turnover"].mean() * 12)
    m["cost_drag_ann"] = float(bt.monthly["cost"].mean() * 12)
    return m


def run_research(cfg: Config, bundle=None, quick: bool = False, extra_portfolios: dict | None = None) -> dict:
    """``extra_portfolios``: {name: config overrides} — pre-registered alternative portfolio rules (e.g. a
    relative mandate) evaluated on the same final predictions; each is logged as a trial."""
    t_start = time.time()
    art = Path(cfg.paths.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    full = bundle if bundle is not None else load_bundle(cfg)
    hstart = holdout_start(full.dates, int(cfg.validation.holdout_months))
    data = full.truncate(hstart - pd.Timedelta(days=1))
    assert_research_data(data, hstart)
    log(f"research data {data.dates[0].date()} .. {data.dates[-1].date()} | holdout from {hstart.date()} (locked)")

    scorer = make_scorer(cfg)
    panel = build_features(data, cfg, scorer)
    log(f"features: {panel.X.shape[0]} stock-months, {panel.X.shape[1]} signals, "
        f"median universe {int(panel.universe.sum(axis=1).median())}")
    vt = vol_target(cfg)
    boot = dict(cfg.validation.bootstrap)
    reg = TrialRegistry(art / "trials", stage="research")
    shash = _sections_hash(cfg)
    cands = list(cfg.models.candidates)
    tr = cfg.trials
    variants = list(itertools.product(tr.top_n, tr.buffer, tr.weighting, tr.regime_filter))
    pre = (cfg.portfolio.top_n, cfg.portfolio.buffer, cfg.portfolio.weighting, cfg.portfolio.regime_filter.enabled)
    if pre not in variants:
        variants.append(pre)
    store = {}
    horizons = sorted(set(tr.horizons) | {cfg.labels.horizon_months})
    fsets = list(dict.fromkeys(list(tr.feature_sets) + [cfg.models.feature_set]))
    for h in horizons:
        cfg_h = _cfg_h(cfg, h)
        labels = forward_returns(data, panel.dates, h, cfg.labels.benchmark, cfg.data.default_delisting_return,
                                 panel.universe)
        L = label_panel(labels, panel.X.index)
        for fs in fsets:
            feats = [f for f in cfg.features.sets[fs] if f in panel.X.columns]
            keep = haircut_factor(feats, cfg)
            X = panel.features(feats)
            log(f"walk-forward h={h} features={fs} ({len(feats)}) models={cands}")
            wf = walk_forward(X, L, cands, cfg_h)
            smooth = float(cfg.portfolio.get("score_smoothing", 0.0) or 0.0)
            views = {m: smooth_predictions(model_view(wf.preds, m), smooth) for m in cands}
            nets = {}
            cls = {}
            for m in cands + ["ensemble"]:
                if m == "ensemble":
                    dec = gate_decisions(nets, labels.exit, wf.schedule, cfg)
                    views["ensemble"] = smooth_predictions(ensemble_predictions(wf.preds, dec, cands), smooth)
                    store[(h, fs)] = {"wf": wf, "labels": labels, "L": L, "X": X, "views": views,
                                      "decisions": dec, "keep": keep}
                v = views[m]
                Lv = L.reindex(v.index)
                cls[m] = classification_metrics(v["prob"], Lv["y"], Lv["excess"], v.index.get_level_values(0))
                for (tn, buf, wt, rgm) in variants:
                    prm = SizingParams.from_cfg(cfg, keep, vt, top_n=tn, buffer=buf, weighting=wt)
                    conf = {"model": m, "features": fs, "horizon": int(h), "top_n": int(tn), "buffer": bool(buf),
                            "weighting": wt, "regime_filter": bool(rgm), "sections": shash}

                    def run(v=v, prm=prm, rgm=rgm):
                        bt = monthly_backtest(v, panel, labels, data, cfg, prm, regime_filter=rgm)
                        return bt, _trial_metrics(bt, boot)
                    try:
                        bt, met = run()
                        met.update({f"cls_{k}": val for k, val in cls[m].items()})
                        reg.log(conf, (bt.net - bt.monthly["rf"]).rename(None), met)
                        if (tn, buf, wt, rgm) == pre:
                            nets[m] = bt.net
                    except Exception as exc:  # failures are recorded, never hidden
                        reg.log(conf, None, None, status="failed", error=repr(exc))
            store[(h, fs)]["cls"] = cls
    log(f"trial registry: {reg.n_trials('research')} distinct configurations (incl. earlier runs and failures)")

    # ------------------------------------------------------------ final (pre-registered) strategy
    key = (int(cfg.labels.horizon_months), cfg.models.feature_set)
    final_model = "ensemble"
    if cfg.validation.selection == "best_sharpe":
        recs = [r for r in reg.records("research") if r["status"] == "ok" and r["config"].get("sections") == shash]
        best = max(recs, key=lambda r: r["metrics"].get("sharpe") or -9)
        key = (best["config"]["horizon"], best["config"]["features"])
        final_model = best["config"]["model"]
    S = store[key]
    labels, L, X, keep = S["labels"], S["L"], S["X"], S["keep"]
    final_preds = S["views"][final_model]
    prm = SizingParams.from_cfg(cfg, keep, vt)
    bt = monthly_backtest(final_preds, panel, labels, data, cfg, prm, regime_filter=cfg.portfolio.regime_filter.enabled)
    bt_ew = monthly_backtest(final_preds, panel, labels, data, cfg, SizingParams.from_cfg(cfg, keep, vt, weighting="equal"))
    bt_base = monthly_backtest(S["views"][cfg.models.baseline], panel, labels, data, cfg, prm)
    final_metrics = _trial_metrics(bt, boot)
    extra_results = {}
    for name, ov in (extra_portfolios or {}).items():
        c2 = cfg.copy_with(ov)
        b2 = monthly_backtest(final_preds, panel, labels, data, c2, SizingParams.from_cfg(c2, keep, vol_target(c2)),
                              regime_filter=bool(c2.portfolio.regime_filter.enabled))
        m2 = _trial_metrics(b2, boot)
        reg.log({"model": final_model, "features": key[1], "horizon": int(key[0]), "extra_portfolio": name,
                 "overrides": ov, "sections": shash}, (b2.net - b2.monthly["rf"]).rename(None), m2)
        extra_results[name] = {"overrides": ov, "metrics": m2, "monthly": b2.monthly}
    rf_m = bt.monthly["rf"]
    sig_s, sig_b = bt.net.std(), bt.bench.std()
    bench_vm = rf_m + (bt.bench - rf_m) * (sig_s / sig_b if sig_b > 0 else 1)
    vm_metrics = performance_table(bt.net, bench_vm, rf_m, boot=boot)
    cls_final = classification_metrics(final_preds["prob"], L.reindex(final_preds.index)["y"],
                                       L.reindex(final_preds.index)["excess"], final_preds.index.get_level_values(0))
    log("daily engine (stops, delisting, kill switch)")
    eng = run_daily(data, bt.weights, cfg)
    eng_nk = run_daily(data, bt.weights, cfg, kill_switch=False)
    em = eng.monthly()
    rf_e = data.risk_free.resample("ME").mean().reindex(em.index).fillna(0) / 12 if data.risk_free is not None else None
    engine_metrics = performance_table(em["net"], em["bench"], rf_e, boot=boot)
    engine_nk_metrics = performance_table(eng_nk.monthly()["net"], eng_nk.monthly()["bench"], rf_e, boot=boot)
    tstats = {"engine": trade_stats(eng.trades.rename(columns={"excess": "excess"}).assign(
        months=(pd.to_datetime(eng.trades["exit_date"]) - pd.to_datetime(eng.trades["entry_date"])).dt.days / 30.4)
        if len(eng.trades) else None), "spells": trade_stats(holding_spells(bt, labels))}

    # ------------------------------------------------------------ DSR / PBO over the full registry
    n_trials = reg.n_trials("research")
    R = reg.returns()
    ids = {r["trial_id"] for r in reg.records("research") if r["status"] == "ok"}
    R = R[[c for c in R.columns if c in ids]]
    # On the repository's default synthetic world, every earlier search (original 48 trials, Codex's
    # experiments, algorithm research v2) used the same data: charge DSR/PBO for all of it.
    n_trials_run, history_used = n_trials, False
    srs = reg.trial_sharpes()
    if data.meta.get("source") == "synthetic" and data.meta.get("seed") == 7:
        hist = load_history(extra={"this_run": (reg.records("research"), R)})
        n_trials, history_used = max(n_trials, hist["n_trials"]), True
        HR = hist["returns"].reindex(R.index) if len(R) else hist["returns"]
        R = HR.dropna(axis=1, how="all")
        srs = trial_sharpes(R.dropna(axis=0, how="any")) if len(R) else srs
    dsr = deflated_sharpe((bt.net - rf_m).to_numpy(), n_trials, srs)
    pbo_res = pbo(R.dropna(axis=0, how="any") if history_used else R,
                  int(cfg.validation.pbo.n_blocks), int(cfg.validation.pbo.max_combinations))
    n_eff = effective_trials(R) if R.shape[1] > 1 else 1.0
    dsr_eff = deflated_sharpe((bt.net - rf_m).to_numpy(), max(2, int(round(n_eff))), srs)
    log(f"DSR={dsr['dsr']:.3f} (N={n_trials}{' incl. full history' if history_used else ''}, "
        f"this run {n_trials_run}, N_eff={n_eff:.1f}) PBO={pbo_res['pbo']:.2f}")

    # ------------------------------------------------------------ sanity
    bt_fn = lambda v: monthly_backtest(v, panel, labels, data, cfg, prm)
    san = {}
    if not quick:
        log("sanity: shuffled labels / influence / noise canary")
        base_cls = S["cls"].get(cfg.models.baseline, {})
        san["shuffled_labels"] = sanity.shuffled_label_test(X, L, _cfg_h(cfg, key[0]), bt_fn,
                                                            true_ic=base_cls.get("ic_mean"))
    san["influence"] = sanity.influence_test(bt, float(cfg.validation.influence_drop_frac))
    san["noise_canary"] = sanity.noise_canary(X, L, cfg)

    # ------------------------------------------------------------ sensitivity
    log("sensitivity: regimes, sectors, size, parameters, costs")

    def run_bt(universe_mask=None, prm_over=None, cost_multiplier=1.0):
        p = SizingParams.from_cfg(cfg, keep, vt, **(prm_over or {}))
        return monthly_backtest(final_preds, panel, labels, data, cfg, p, universe_mask=universe_mask,
                                cost_multiplier=cost_multiplier)
    sens = {
        "regimes": sensitivity.period_table(bt.monthly, dict(cfg.validation.regimes)),
        "structural_break": sensitivity.split_table(bt.monthly, cfg.validation.structural_break),
        "market_state": sensitivity.market_state_table(bt.monthly, panel.regime, cfg.portfolio.regime_filter.vix_threshold),
        "yearly": sensitivity.yearly_table(bt.monthly),
        "size": sensitivity.size_robustness(run_bt, panel),
    }
    if not quick:
        sens["sectors"] = sensitivity.sector_exclusion(run_bt, panel, data.sectors)
        sens["parameters"] = sensitivity.parameter_grid(run_bt, {"top_n": [30, 50, 80], "exit_rank_mult": [1.0, 1.6, 2.2],
                                                                  "kelly_fraction": [0.1, 0.25]})
    cost_rows = {}
    for mult in (0.5, 1.0, 2.0, 3.0):
        b2 = run_bt(cost_multiplier=mult)
        cost_rows[f"x{mult:g}"] = {"excess_ann": float((b2.net - b2.bench).mean() * 12),
                                   "sharpe": sharpe(b2.net - b2.monthly["rf"])}
    sens["costs"] = pd.DataFrame(cost_rows).T

    # ------------------------------------------------------------ purged k-fold diagnostics
    kfold = None
    if not quick:
        log("purged + embargoed k-fold diagnostics")
        kfold = purged_kfold_scores(X, L, cands, _cfg_h(cfg, key[0]))

    # ------------------------------------------------------------ CPCV
    cpcv = {}
    last_R = max(S["decisions"])
    members = [m for m, ok in S["decisions"][last_R].items() if ok]
    if not quick:
        log(f"CPCV paths for members {members}")
        paths = cpcv_path_predictions(X, L, members, cfg)
        srs = []
        for pth in paths:
            b3 = monthly_backtest(pth, panel, labels, data, cfg, prm)
            srs.append({"sharpe": sharpe(b3.net - b3.monthly["rf"]), "excess_ann": float((b3.net - b3.bench).mean() * 12)})
        cpcv = {"paths": pd.DataFrame(srs), "members": members}

    # ------------------------------------------------------------ signal decay + haircut single-signal tests
    ic = decay.signal_ic(panel.X, L["excess"])
    dtab = decay.decay_table(ic, cfg, int(cfg.live.decay_review_months), float(cfg.live.dead_signal_tstat))
    single = {}
    n_univ = int(panel.universe.sum(axis=1).median())
    for c in cfg.features.sets[cfg.models.feature_set]:
        if c not in panel.X.columns:
            continue
        sv = pd.DataFrame({"prob": 0.6, "score": EXPECTED_SIGN.get(c, 1) * panel.X[c], "er": 0.0})
        p1 = SizingParams.from_cfg(cfg, 1.0, 10.0, top_n=max(10, n_univ // 5), buffer=False, min_prob=0.0)
        b4 = monthly_backtest(sv[sv.index.get_level_values(0) >= final_preds.index.get_level_values(0).min()],
                              panel, labels, data, cfg, p1)
        ex = float((b4.net - b4.bench).mean() * 12)
        h_ = float(cfg.signals.haircut) if cfg.signals.registry.get(c, {}).get("published", True) else 0.0
        single[c] = {"net_excess_ann": ex, "after_haircut": ex * (1 - h_), "sharpe": sharpe(b4.net - b4.monthly["rf"]),
                     "turnover_ann": float(b4.monthly["turnover"].mean() * 12)}
    dtab = dtab.join(pd.DataFrame(single).T, how="left")

    # ------------------------------------------------------------ importance (SHAP) of the latest members
    last_models = S["wf"].models.get(last_R, {})
    recent = X[X.index.get_level_values(0) >= X.index.get_level_values(0).max() - pd.DateOffset(months=24)]
    importance = {}
    if last_models and members:
        contrib = ensemble_contributions([last_models[m] for m in members if m in last_models], recent)
        importance = mean_abs_importance(contrib)
    keep_imp = haircut_factor(list(X.columns), cfg, importance) if importance else keep

    # ------------------------------------------------------------ news (separate daily strategy)
    news = {"info": panel.news_info}
    if panel.news_daily is not None:
        nd = panel.news_daily
        news.update({"gross_sharpe": sharpe(nd["gross"], ppy=252), "net_sharpe": sharpe(nd["net"], ppy=252),
                     "turnover_daily": float(nd["turnover"].mean()),
                     "by_year": nd.groupby(nd.index.year).apply(
                         lambda g: pd.Series({"gross_sr": sharpe(g["gross"], ppy=252), "net_sr": sharpe(g["net"], ppy=252)}),
                         include_groups=False),
                     "weight": panel.news_weight})

    flags = red_flags(final_metrics, cls_final, dsr, pbo_res, cfg, data.meta, san)
    spec = {"config_hash": cfg.hash(), "sections_hash": shash, "feature_set": key[1], "horizon": key[0],
            "model": final_model, "members": members, "features": list(X.columns), "keep": keep,
            "keep_importance": keep_imp, "vol_target": vt, "research_end": str(data.dates[-1].date()),
            "holdout_start": str(hstart.date()), "dsr": dsr["dsr"], "pbo": pbo_res["pbo"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    spec["research_gate"] = {
        "full_validation": not quick,
        "sanity_passed": all(san.get(name, {}).get("passed") is True
                             for name in ("shuffled_labels", "influence", "noise_canary")),
        "blocking_flags": [flag for flag in flags if flag["level"] in ("critical", "serious")],
    }
    (art / "final_spec.json").write_text(json.dumps(spec, indent=2, default=str))
    result = {
        "kind": "research", "config": cfg.to_dict(), "meta": data.meta, "hstart": hstart,
        "survivorship_warning": data.survivorship_warning(), "spec": spec,
        "final_metrics": final_metrics, "vol_matched_metrics": vm_metrics, "engine_metrics": engine_metrics,
        "engine_no_kill_metrics": engine_nk_metrics, "cls": cls_final, "trade_stats": tstats,
        "monthly": bt.monthly, "monthly_ew": bt_ew.monthly, "extra_portfolios": extra_results,
        "final_preds": final_preds, "monthly_baseline": bt_base.monthly,
        "engine_daily": eng.daily, "engine_trades": eng.trades, "kill_events": eng.kill_events,
        "engine_nk_daily": eng_nk.daily, "dsr": dsr, "dsr_eff": dsr_eff, "pbo": {k: v for k, v in pbo_res.items()},
        "n_trials": n_trials, "n_trials_this_run": n_trials_run, "history_charged": history_used, "n_eff": n_eff, "trial_returns": R, "trial_records": reg.records("research"),
        "model_cls": {f"{k[1]}|h{k[0]}": v["cls"] for k, v in store.items()},
        "gate": {str(k.date()): v for k, v in S["decisions"].items()}, "sanity": san, "sensitivity": sens,
        "cpcv": cpcv, "kfold": kfold, "decay_table": dtab, "ic": ic, "rolling_ic": decay.rolling_ic(ic), "importance": importance,
        "news": news, "red_flags": flags, "runtime_sec": time.time() - t_start,
        "diag": pd.DataFrame(bt.diag).set_index("date") if bt.diag else None,
    }
    with (art / "research_result.pkl").open("wb") as fh:
        pickle.dump(result, fh)
    # Safe, portable API summary; the dashboard never unpickles report files.
    summary = {"config_hash": cfg.hash(), "metrics": final_metrics, "n_trials": n_trials,
               "red_flags": flags, "signal_decay": dtab.reset_index().to_dict(orient="records")}
    def json_default(value):
        return value.item() if hasattr(value, "item") else str(value)
    # Round-trip via pandas normalizes NaN to null for strict JSON clients.
    normalized = json.loads(pd.Series(summary).to_json(default_handler=json_default))
    (art / "validation_summary.json").write_text(json.dumps(normalized, indent=2))
    log(f"research done in {result['runtime_sec']:.0f}s")
    return result


def run_holdout(cfg: Config, bundle=None, force: bool = False, extra_portfolios: dict | None = None) -> dict:
    """Opens the holdout once. ``extra_portfolios`` must be the same pre-registered alternatives that were
    evaluated in research; they are scored inside this single opening, never chosen after it."""
    art = Path(cfg.paths.artifacts)
    spec_path = art / "final_spec.json"
    if not spec_path.exists():
        raise RuntimeError("run research first: no final_spec.json")
    spec = json.loads(spec_path.read_text())
    if spec["config_hash"] != cfg.hash():
        raise RuntimeError("config changed since research; the holdout must test the frozen research spec")
    full = bundle if bundle is not None else load_bundle(cfg)
    hstart = pd.Timestamp(spec["holdout_start"])
    lock = HoldoutLock(art).open(cfg.hash(), spec, force=force)
    log(f"HOLDOUT OPENED ({'CONTAMINATED - opened %d times' % lock['n_opens'] if lock['contaminated'] else 'first and only time'})")
    panel = build_features(full, cfg, make_scorer(cfg))
    h = int(spec["horizon"])
    labels = forward_returns(full, panel.dates, h, cfg.labels.benchmark, cfg.data.default_delisting_return, panel.universe)
    L = label_panel(labels, panel.X.index)
    X = panel.features(spec["features"])
    wf = walk_forward(X, L, spec["members"], _cfg_h(cfg, h), test_start=hstart)
    views = [model_view(wf.preds, m) for m in spec["members"]]
    preds = pd.DataFrame({"prob": np.mean([v["prob"] for v in views], 0), "score": np.mean([v["score"] for v in views], 0),
                          "er": np.mean([v["er"] for v in views], 0)}, index=wf.preds.index)
    preds = smooth_predictions(preds, float(cfg.portfolio.get("score_smoothing", 0.0) or 0.0))
    preds = preds[preds.index.get_level_values(0) >= hstart]
    prm = SizingParams.from_cfg(cfg, spec["keep"], spec["vol_target"])
    bt = monthly_backtest(preds, panel, labels, full, cfg, prm, regime_filter=cfg.portfolio.regime_filter.enabled)
    boot = dict(cfg.validation.bootstrap)
    met = _trial_metrics(bt, boot)
    extra_results = {}
    for name, ov in (extra_portfolios or {}).items():
        c2 = cfg.copy_with(ov)
        b2 = monthly_backtest(preds, panel, labels, full, c2,
                              SizingParams.from_cfg(c2, spec["keep"], spec["vol_target"] if c2.portfolio.mandate == "absolute"
                                                    else vol_target(c2)),
                              regime_filter=bool(c2.portfolio.regime_filter.enabled))
        extra_results[name] = {"overrides": ov, "metrics": _trial_metrics(b2, boot), "monthly": b2.monthly}
    Lp = L.reindex(preds.index)
    cls = classification_metrics(preds["prob"], Lp["y"], Lp["excess"], preds.index.get_level_values(0))
    eng = run_daily(full, bt.weights, cfg)
    reg = TrialRegistry(art / "trials", stage="research")
    dsr = deflated_sharpe((bt.net - bt.monthly["rf"]).to_numpy(), max(1, reg.n_trials("research")), reg.trial_sharpes())
    flags = red_flags(met, cls, dsr, {"pbo": spec.get("pbo")}, cfg, full.meta)
    if lock["contaminated"]:
        flags.insert(0, {"level": "critical", "message": f"HOLDOUT CONTAMINATED: opened {lock['n_opens']} times"})
    if len(bt.monthly) < 24:
        flags.append({"level": "warning", "message": f"holdout has only {len(bt.monthly)} months: wide uncertainty"})
    result = {"kind": "holdout", "config": cfg.to_dict(), "meta": full.meta, "hstart": hstart, "spec": spec, "lock": lock,
              "final_metrics": met, "cls": cls, "monthly": bt.monthly, "extra_portfolios": extra_results, "engine_daily": eng.daily,
              "engine_trades": eng.trades, "kill_events": eng.kill_events, "dsr": dsr, "red_flags": flags,
              "survivorship_warning": full.survivorship_warning()}
    with (art / "holdout_result.pkl").open("wb") as fh:
        pickle.dump(result, fh)
    return result
