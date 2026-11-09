"""Self-contained HTML report (charts embedded as PNG). Every success metric is shown together;
hit rate never appears alone. Red flags are listed first."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# palette (validated categorical slots 1-4 + neutrals)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
STATUS = {"critical": ("#c62828", "&#x2716;"), "serious": ("#d9480f", "&#x26A0;"),
          "warning": ("#b7791f", "&#x26A0;"), "info": ("#2a78d6", "&#x2139;"), "ok": ("#2b8a3e", "&#x2714;")}


def _style_axes(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return '<img alt="chart" src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '"/>'


def _label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.plot([x], [y], marker="o", markersize=3.5, color=color)


def chart_equity(monthly: pd.DataFrame, extra: dict | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    _style_axes(ax, None)
    series = {"Strategy (net)": (monthly["net"], C1), "Benchmark (buy & hold)": (monthly["bench"], C2)}
    for k, (s, c) in (extra or {}).items():
        series[k] = (s, c)
    for name, (r, c) in series.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, color=c, linewidth=1.6, label=name)
        _label_end(ax, eq.index[-1], eq.values[-1], f"{eq.values[-1]:.2f}x", c)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylabel("growth of $1 (log)", color=INK2, fontsize=8)
    return _png(fig)


def chart_drawdown(monthly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    _style_axes(ax)
    for name, r, c in [("Strategy", monthly["net"], C1), ("Benchmark", monthly["bench"], C2)]:
        eq = (1 + r.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=c, linewidth=1.4, label=name)
    ax.set_ylabel("drawdown %", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _png(fig)


def chart_hist(values, marks: dict, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    _style_axes(ax)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v):
        ax.hist(v, bins=min(30, max(5, len(v) // 2)), color=C1, alpha=0.85, edgecolor=SURFACE, linewidth=1.5)
    for (name, x), c in zip(marks.items(), [C2, INK2, C3]):
        if x is not None and np.isfinite(x):
            ax.axvline(x, color=c, linewidth=1.6)
            ax.annotate(name, (x, ax.get_ylim()[1] * 0.9), xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color=INK2)
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    return _png(fig)


def chart_small_multiples(df: pd.DataFrame, title_fmt="{}", ncols=4) -> str:
    cols = [c for c in df.columns if df[c].notna().sum() > 3]
    if not cols:
        return ""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 1.9 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, c in zip(axes.flat, cols):
        _style_axes(ax)
        ax.plot(df.index, df[c].values, color=C1, linewidth=1.3)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(title_fmt.format(c), fontsize=8, color=INK, loc="left")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return _png(fig)


def chart_costs(monthly: pd.DataFrame) -> str:
    comp = [c for c in ("cost_commission", "cost_spread", "cost_slippage", "cost_impact") if c in monthly]
    if not comp:
        return ""
    yearly = monthly[comp].groupby(monthly.index.year).sum() * 1e4
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    _style_axes(ax)
    bottom = np.zeros(len(yearly))
    for c, col in zip(comp, [C1, C2, C3, C4]):
        ax.bar(yearly.index, yearly[c].values, bottom=bottom, color=col, width=0.7, edgecolor=SURFACE, linewidth=1.5,
               label=c.replace("cost_", ""))
        bottom += yearly[c].values
    ax.set_ylabel("cost drag (bps / year)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    return _png(fig)


def chart_line(s: pd.Series, ylabel: str, hline: float | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8.6, 2.3))
    _style_axes(ax)
    ax.plot(s.index, s.values, color=C1, linewidth=1.4)
    if hline is not None:
        ax.axhline(hline, color=INK2, linewidth=0.8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    return _png(fig)


def chart_reliability(prob: pd.Series, y: pd.Series) -> str:
    df = pd.DataFrame({"p": prob, "y": y}).dropna()
    if len(df) < 100:
        return ""
    df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    _style_axes(ax)
    lo, hi = min(g["p"].min(), g["y"].min()) - 0.01, max(g["p"].max(), g["y"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8)
    ax.plot(g["p"], g["y"], color=C1, linewidth=1.6, marker="o", markersize=4)
    ax.set_xlabel("predicted probability (decile mean)", color=INK2, fontsize=8)
    ax.set_ylabel("realized frequency", color=INK2, fontsize=8)
    return _png(fig)


# ---------------------------------------------------------------------- formatting
def _f(v, kind="num"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "int":
        return f"{int(round(v)):,}"
    return f"{v:.2f}" if abs(v) >= 0.01 or v == 0 else f"{v:.4f}"


def table(df: pd.DataFrame, pct_cols=(), int_cols=(), index=True, max_rows=200) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>n/a</p>"
    df = df.head(max_rows)
    head = ("<th></th>" if index else "") + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        cells = "".join(
            f"<td>{_f(v, 'pct' if c in pct_cols else 'int' if c in int_cols else 'num') if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else html.escape(str(v))}</td>"
            for c, v in r.items())
        ix = html.escape(str(idx.date()) if isinstance(idx, pd.Timestamp) else str(idx))
        rows.append(f"<tr>{'<th>' + ix + '</th>' if index else ''}{cells}</tr>")
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


METRIC_ROWS = [
    ("1. Net return (OOS)", None, None), ("CAGR", "cagr", "pct"), ("Excess return vs benchmark (ann.)", "excess_ann", "pct"),
    ("Excess 95% CI low", "excess_ci_low", "pct"), ("Excess 95% CI high", "excess_ci_high", "pct"),
    ("P(excess > 0) (bootstrap)", "p_excess_positive", "pct"),
    ("2. Risk-adjusted", None, None), ("Sharpe", "sharpe", "num"), ("Sharpe 95% CI low", "sharpe_ci_low", "num"),
    ("Sharpe 95% CI high", "sharpe_ci_high", "num"), ("PSR (SR > 0)", "psr_0", "pct"),
    ("3. Drawdown", None, None), ("Max drawdown", "max_drawdown", "pct"), ("Calmar", "calmar", "num"),
    ("Sortino", "sortino", "num"), ("Annual volatility", "ann_vol", "pct"),
    ("4. Versus benchmark", None, None), ("Information ratio", "information_ratio", "num"),
    ("Tracking error", "tracking_error", "pct"), ("CAPM alpha (ann.)", "capm_alpha", "pct"), ("CAPM beta", "capm_beta", "num"),
    ("5. Secondary only", None, None), ("Hit rate (months beating benchmark)", "hit_rate_periods", "pct"),
    ("Avg win / avg loss (months)", "avg_win_loss_ratio", "num"), ("Expectancy per month", "expectancy_period", "pct"),
    ("Turnover (ann., one-way sum)", "turnover_ann", "num"), ("Cost drag (ann.)", "cost_drag_ann", "pct"),
    ("Months", "n_periods", "int"),
]


def metrics_table(cols: dict) -> str:
    head = "<th>Metric</th>" + "".join(f"<th>{html.escape(k)}</th>" for k in cols)
    rows = []
    for label, key, kind in METRIC_ROWS:
        if key is None:
            rows.append(f"<tr class='grp'><th colspan='{len(cols) + 1}'>{label}</th></tr>")
            continue
        rows.append("<tr><th>" + label + "</th>" + "".join(f"<td>{_f(m.get(key), kind)}</td>" for m in cols.values()) + "</tr>")
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def bench_metrics(m: dict) -> dict:
    return {"cagr": m.get("bench_cagr"), "sharpe": m.get("bench_sharpe"), "max_drawdown": m.get("bench_max_drawdown"),
            "calmar": m.get("bench_calmar"), "ann_vol": m.get("bench_vol"), "excess_ann": 0.0, "information_ratio": None,
            "n_periods": m.get("n_periods")}


def flags_html(flags: list) -> str:
    if not flags:
        return f"<div class='flag' style='border-color:{STATUS['ok'][0]}'>{STATUS['ok'][1]} <b>ok</b> no red flags</div>"
    out = []
    for f in flags:
        col, icon = STATUS.get(f["level"], STATUS["info"])
        out.append(f"<div class='flag' style='border-color:{col}'><span style='color:{col}'>{icon} <b>{f['level']}</b></span> "
                   f"{html.escape(f['message'])}</div>")
    return "".join(out)


CSS = f"""
:root {{ color-scheme: light; }}
body {{ background:{SURFACE}; color:{INK}; font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       margin:0; padding:0 16px; }}
main {{ max-width:980px; margin:0 auto; padding:24px 0 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 8px; border-bottom:1px solid {GRID}; padding-bottom:4px; }}
h3 {{ font-size:15px; margin:20px 0 6px; }}
.muted {{ color:{INK2}; }} .banner {{ background:#fff4e5; border:1px solid #f0c27b; padding:10px 12px; border-radius:6px; margin:12px 0; }}
.flag {{ border-left:4px solid; background:#fff; padding:6px 10px; margin:6px 0; border-radius:4px; }}
.tw {{ overflow-x:auto; }} table {{ border-collapse:collapse; font-size:12.5px; margin:6px 0 12px; }}
th, td {{ padding:4px 8px; border-bottom:1px solid {GRID}; text-align:right; white-space:nowrap; }}
th:first-child {{ text-align:left; }} thead th {{ color:{INK2}; font-weight:600; }}
tr.grp th {{ text-align:left; background:#f3f2ee; color:{INK}; }}
img {{ max-width:100%; height:auto; }} .kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin:12px 0; }}
.kpi {{ background:#fff; border:1px solid {GRID}; border-radius:8px; padding:10px 14px; min-width:150px; }}
.kpi .v {{ font-size:20px; font-weight:600; }} .kpi .l {{ color:{INK2}; font-size:12px; }}
code {{ background:#f3f2ee; padding:1px 4px; border-radius:3px; }}
"""


def kpi(label, value) -> str:
    return f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{html.escape(label)}</div></div>"


def write_report(res: dict, path: str | Path, recommendations: pd.DataFrame | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = res["config"]
    m = res["final_metrics"]
    monthly = res["monthly"]
    parts = [f"<main><h1>{html.escape(cfg['report']['title'])} — {res['kind']}</h1>",
             f"<p class='muted'>Data: <b>{html.escape(str(res['meta'].get('source')))}</b> · market {cfg['market']['name']} · "
             f"OOS months {m.get('n_periods')} ({monthly.index.min().date()} – {monthly.index.max().date()}) · holdout starts "
             f"{pd.Timestamp(res['hstart']).date()}</p>"]
    if res["meta"].get("source") == "synthetic":
        parts.append("<div class='banner'><b>Synthetic data.</b> This report runs the full pipeline on a simulated market "
                     "with planted, realistic-size anomalies. It validates the machinery, not any real-world edge.</div>")
    if res.get("survivorship_warning"):
        parts.append(f"<div class='banner'>{html.escape(res['survivorship_warning'])}</div>")

    # verdict
    dsr = res.get("dsr", {})
    parts.append("<h2>Verdict and red flags</h2>")
    beats = (m.get("excess_ci_low") or -1) > 0
    parts.append("<div class='kpis'>" + kpi("Beats buy & hold net (95% CI > 0)?", "YES" if beats else "NO")
                 + kpi("Net excess vs benchmark (ann.)", _f(m.get("excess_ann"), "pct"))
                 + kpi("Sharpe (net)", _f(m.get("sharpe")))
                 + kpi(f"DSR (N={dsr.get('n_trials')})", _f(dsr.get("dsr")))
                 + kpi("Alpha DSR (skill beyond market exposure)", _f((res.get("dsr_alpha") or {}).get("dsr")))
                 + kpi("PBO", _f(res.get("pbo", {}).get("pbo")))
                 + kpi("Max drawdown", _f(m.get("max_drawdown"), "pct")) + "</div>")
    parts.append(flags_html(res.get("red_flags", [])))

    # metrics
    parts.append("<h2>All success metrics (out-of-sample, net of costs)</h2>")
    cols = {"Strategy": m, "Benchmark": bench_metrics(m)}
    if "vol_matched_metrics" in res:
        vm = dict(res["vol_matched_metrics"])
        cols["vs vol-matched bench."] = vm
    if "engine_metrics" in res:
        cols["Daily engine (stops+kill)"] = res["engine_metrics"]
        cols["Daily engine (no kill)"] = res["engine_no_kill_metrics"]
    parts.append(metrics_table(cols))
    parts.append("<p class='muted'>The vol-matched column compares the strategy with the benchmark scaled to the same "
                 "volatility (cash at the risk-free rate). The strategy is vol-targeted to "
                 f"{_f(res['spec'].get('vol_target'), 'pct')} because of the {cfg['capital']['max_drawdown']:.0%} drawdown "
                 "tolerance, so raw-return comparisons with a 100%-invested index are structurally hard to win.</p>")
    cls = res.get("cls", {})
    parts.append("<h3>Model quality (direction classification, OOS)</h3>")
    parts.append(table(pd.DataFrame([cls]).T.rename(columns={0: "value"})))
    if res.get("trade_stats"):
        parts.append("<h3>Per-trade statistics (secondary)</h3>")
        parts.append(table(pd.DataFrame(res["trade_stats"]).T))

    # charts
    parts.append("<h2>Equity curve vs benchmark</h2>")
    extra = {}
    if "monthly_ew" in res and res["monthly_ew"] is not None and len(res["monthly_ew"]):
        extra["Equal-weight comparator"] = (res["monthly_ew"]["net"].reindex(monthly.index), C3)
    parts.append(chart_equity(monthly, extra))
    parts.append("<h2>Drawdown</h2>" + chart_drawdown(monthly))
    if "diag" in res and res["diag"] is not None and len(res["diag"]):
        d = res["diag"]
        cols_ = [c for c in ("gross", "vol_scale", "est_vol", "n_selected") if c in d]
        parts.append("<h3>Exposure after Kelly caps and vol targeting (yearly mean)</h3>")
        parts.append(table(d[cols_].groupby(d.index.year).mean(), pct_cols=("gross", "vol_scale", "est_vol")))

    # overfitting statistics
    if "trial_returns" in res:
        parts.append("<h2>Multiple testing: trials, DSR, PBO</h2>")
        tr = res["trial_returns"]
        srs = (tr.mean() / tr.std() * np.sqrt(12)).to_numpy() if tr.shape[1] else []
        parts.append(f"<p>{res['n_trials']} distinct configurations were tried and recorded (including failures and earlier "
                     f"runs); effective number of independent trials ≈ {res['n_eff']:.1f}. The deflated Sharpe compares the "
                     f"final Sharpe with the expected maximum Sharpe of {res['n_trials']} unskilled trials "
                     f"(SR₀ = {_f(dsr.get('sr0_annual'))} annualized).</p>")
        parts.append(chart_hist(srs, {"final": m.get("sharpe"), "SR0 (null max)": dsr.get("sr0_annual")},
                                "annualized Sharpe of every trial (net of rf)"))
        pbo_ = res.get("pbo", {})
        parts.append(f"<p>PBO (CSCV, {pbo_.get('n_combinations', '–')} combinations): <b>{_f(pbo_.get('pbo'))}</b> — "
                     "probability that the in-sample best configuration ranks below the median out of sample.</p>")
        if len(pbo_.get("logits", [])):
            parts.append(chart_hist(pbo_["logits"], {"0": 0.0}, "logit of OOS relative rank of the IS-best trial"))
        if res.get("dsr_eff"):
            parts.append(f"<p class='muted'>DSR using the effective number of trials: {_f(res['dsr_eff'].get('dsr'))}</p>")
        if res.get("dsr_alpha"):
            parts.append("<p>Alpha DSR deflates the <b>beta-adjusted</b> returns (appraisal ratio): a long-only strategy "
                         "earns a positive Sharpe from holding equities alone, so this is the test of stock-selection "
                         f"skill. All {res['n_trials']} trials: <b>{_f(res['dsr_alpha'].get('dsr'))}</b>; effective "
                         f"number of independent trials ({_f(res.get('n_eff_alpha'))}): "
                         f"<b>{_f((res.get('dsr_alpha_eff') or {}).get('dsr'))}</b>. The truth lies between the two.</p>")
        recs = res.get("trial_records", [])
        if recs:
            rows = [{**{k: r["config"].get(k) for k in ("model", "features", "horizon", "top_n", "buffer", "weighting",
                                                       "regime_filter")}, "status": r["status"],
                     "sharpe": r["metrics"].get("sharpe"), "excess_ann": r["metrics"].get("excess_ann"),
                     "max_dd": r["metrics"].get("max_drawdown"), "auc": r["metrics"].get("cls_auc"),
                     "accuracy": r["metrics"].get("cls_accuracy")} for r in recs]
            df = pd.DataFrame(rows).drop_duplicates(subset=["model", "features", "horizon", "top_n", "buffer", "weighting",
                                                            "regime_filter"], keep="last")
            parts.append("<h3>Trial log (latest run of each configuration)</h3>")
            parts.append(table(df.sort_values("sharpe", ascending=False), pct_cols=("excess_ann", "max_dd", "accuracy"),
                               index=False))
    if res.get("kfold") is not None and len(res["kfold"]):
        parts.append("<h3>Purged + embargoed K-fold (out-of-fold skill per model, research data)</h3>")
        parts.append(table(res["kfold"], pct_cols=("accuracy",)))
    if res.get("model_cls"):
        rows = {f"{k} · {m}": v for k, d in res["model_cls"].items() for m, v in d.items()}
        parts.append("<h3>Walk-forward OOS classification per model and feature set</h3>")
        parts.append(table(pd.DataFrame(rows).T[["accuracy", "auc", "log_loss", "brier", "ic_mean", "ic_tstat"]],
                           pct_cols=("accuracy",)))
    if res.get("gate"):
        g = pd.DataFrame(res["gate"]).T
        parts.append("<h3>Model gate (complex models admitted only when they beat the baseline OOS, net)</h3>")
        parts.append(table(g.astype(str)))
    if res.get("cpcv") and len(res["cpcv"].get("paths", [])):
        p = res["cpcv"]["paths"]
        parts.append(f"<h3>CPCV: {len(p)} backtest paths (members: {', '.join(res['cpcv']['members'])})</h3>")
        parts.append(chart_hist(p["sharpe"], {"final": m.get("sharpe")}, "Sharpe per CPCV path"))
        parts.append(table(p.describe().T))

    # sanity
    if res.get("sanity"):
        parts.append("<h2>Sanity checks</h2>")
        rows = {}
        for k, v in res["sanity"].items():
            rows[k] = {"passed": "PASS" if v.get("passed") else "FAIL", "rule": v.get("rule", ""),
                       "detail": ", ".join(f"{a}={_f(b)}" for a, b in v.items()
                                           if isinstance(b, (int, float)) and a not in ("passed",))}
            if k == "shuffled_labels":
                rows[k]["detail"] = (f"{v['n_perm']} permutations: mean AUC={_f(v['mean_auc'])}, mean IC={_f(v['mean_ic'])}, "
                                     f"max IC={_f(v['max_ic'])}, t(IC)={_f(v['ic_t_across_perms'])}; real IC={_f(v['true_ic'])}")
        parts.append(table(pd.DataFrame(rows).T))

    # robustness
    sens = res.get("sensitivity", {})
    if sens:
        parts.append("<h2>Robustness</h2>")
        pc = ("ret_ann", "bench_ann", "excess_ann", "max_dd", "bench_max_dd", "hit_rate_vs_bench")
        for key, title in [("regimes", "Crisis regimes"), ("structural_break", "Before / after the 2005 break"),
                           ("market_state", "Market states"), ("size", "Size robustness (no micro-cap dependence)"),
                           ("sectors", "Leave-one-sector-out"), ("costs", "Cost sensitivity")]:
            if key in sens and sens[key] is not None and len(sens[key]):
                parts.append(f"<h3>{title}</h3>" + table(sens[key], pct_cols=pc + ("excess_ann",)))
        if "parameters" in sens:
            parts.append("<h3>Parameter perturbations (look for a plateau, not a peak)</h3>")
            parts.append(table(sens["parameters"], pct_cols=pc, index=False))
        if "yearly" in sens:
            parts.append("<h3>Calendar years</h3>" + table(sens["yearly"], pct_cols=("ret", "bench", "excess")))

    # signals
    if "decay_table" in res:
        parts.append("<h2>Signals: decay, publication haircut, importance</h2>")
        parts.append("<p class='muted'>IC = monthly Spearman correlation of the sign-adjusted signal with next-month excess "
                     "return. Single-signal returns are long-only top-quintile capped value-weight, net of costs; "
                     f"published signals get a {cfg['signals']['haircut']:.0%} haircut.</p>")
        parts.append(table(res["decay_table"], pct_cols=("net_excess_ann", "after_haircut")))
        if "rolling_ic" in res:
            parts.append("<h3>Rolling 36-month IC per signal</h3>" + chart_small_multiples(res["rolling_ic"]))
        if res.get("importance"):
            parts.append("<h3>Mean |SHAP| of the latest ensemble (calibrated-logit units)</h3>")
            parts.append(table(pd.Series(res["importance"]).to_frame("mean_abs_shap")))
    if res.get("news"):
        n = res["news"]
        parts.append("<h2>News sentiment (daily strategy, separate and stricter costs)</h2>")
        info = n.get("info", {})
        parts.append(f"<p>Scorer: <b>{info.get('scorer')}</b>; knowledge cutoff: {info.get('cutoff')}; articles: "
                     f"{info.get('n_articles')}; dropped (on/before LLM cutoff): {info.get('n_dropped_cutoff')}. "
                     f"Daily long-short Sharpe gross {_f(n.get('gross_sharpe'))}, net {_f(n.get('net_sharpe'))}.</p>")
        if n.get("weight") is not None:
            parts.append(chart_line(n["weight"].resample("ME").mean(), "news weight (rolling-Sharpe based)", 0))
        if n.get("by_year") is not None:
            parts.append(table(n["by_year"]))

    # costs / trades
    parts.append("<h2>Turnover and costs</h2>" + chart_costs(monthly))
    if "engine_trades" in res and res["engine_trades"] is not None and len(res["engine_trades"]):
        t = res["engine_trades"]
        g = t.groupby("reason").agg(n=("ret", "size"), mean_ret=("ret", "mean"), mean_excess=("excess", "mean"),
                                    hit_rate=("excess", lambda x: (x > 0).mean()))
        parts.append("<h3>Daily engine exits by reason</h3>" + table(g, pct_cols=("mean_ret", "mean_excess", "hit_rate"),
                                                                      int_cols=("n",)))
    if res.get("kill_events"):
        parts.append("<h3>Kill-switch events</h3>" + table(pd.DataFrame(res["kill_events"]), pct_cols=("drawdown",),
                                                         index=False))

    if recommendations is not None and len(recommendations):
        parts.append("<h2>Recommendations</h2>" + table(recommendations, index=False))

    parts.append("<h2>Limitations</h2><ul>"
                 "<li>Results are estimates with wide confidence intervals; nothing here is a promise of profit.</li>"
                 "<li>Published anomalies decay after publication; a 50% haircut is applied but the true decay is unknown.</li>"
                 "<li>Costs are modeled, not observed; paper trading must confirm them.</li>"
                 "<li>LLM news signals cannot be backtested before the model's knowledge cutoff.</li>"
                 "<li>See docs/PLAN.md for what cannot be verified with the available data.</li></ul></main>")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>stocksel report</title><style>{CSS}</style></head>"
           f"<body>{''.join(parts)}</body></html>")
    path.write_text(doc, encoding="utf-8")
    return path
