"""Automatic red flags derived from the research brief's realistic expectations."""
from __future__ import annotations


def red_flags(final_metrics: dict, cls: dict, dsr: dict, pbo: dict, cfg, bundle_meta: dict,
              sanity: dict | None = None, dsr_alpha: dict | None = None) -> list[dict]:
    rf = cfg.validation.red_flags
    out = []

    def add(level, msg):
        out.append({"level": level, "message": msg})

    acc = cls.get("accuracy")
    if acc is not None and acc > rf.max_accuracy:
        add("critical", f"OOS directional accuracy {acc:.1%} > {rf.max_accuracy:.0%}: suspect leakage / overfitting")
    sr = final_metrics.get("sharpe")
    if sr is not None and sr > rf.max_sharpe:
        add("critical", f"Sharpe {sr:.2f} > {rf.max_sharpe}: hunt for bugs before going further")
    d = dsr.get("dsr")
    if d is None or d != d or d < rf.min_dsr:
        add("serious", f"DSR {d if d is None else round(d, 3)} < {rf.min_dsr}: not significant after "
                       f"{dsr.get('n_trials')} trials")
    da = (dsr_alpha or {}).get("dsr")
    if dsr_alpha is not None and (da is None or da != da or da < rf.min_dsr):
        add("serious", f"Alpha DSR {da if da is None else round(da, 3)} < {rf.min_dsr}: no significant stock-selection "
                       "skill once market exposure is removed (a positive Sharpe alone can come from holding equities)")
    p = pbo.get("pbo")
    if p is not None and p == p and p > rf.max_pbo:
        add("serious", f"PBO {p:.2f} > {rf.max_pbo}: the in-sample best trial tends to underperform out of sample")
    ex, lo = final_metrics.get("excess_ann"), final_metrics.get("excess_ci_low")
    if ex is None or ex <= 0:
        add("serious", "Does not beat buy-and-hold of the benchmark net of costs (default assumption holds)")
    elif lo is not None and lo <= 0:
        add("warning", "Net excess return is positive but its 95% CI includes zero: outperformance not established")
    if not bundle_meta.get("survivorship_free", False):
        add("critical", "Data is not survivorship-bias free: returns are biased upward")
    if not bundle_meta.get("pit_fundamentals", False):
        add("warning", "Fundamentals are not point-in-time: fundamental signals disabled")
    if bundle_meta.get("source") == "synthetic":
        add("info", "SYNTHETIC DATA: validates the pipeline only; says nothing about real markets")
    for name, s in (sanity or {}).items():
        if isinstance(s, dict) and s.get("passed") is False:
            add("serious", f"Sanity check failed: {name} ({s.get('rule', '')})")
    return out
