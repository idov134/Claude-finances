"""Conservative promotion gate shared by the CLI and dashboard.

Historical demo booleans are not evidence of real-market validation. A fresh,
uncontaminated holdout with positive excess-return CI is required alongside
the existing multiple-testing and leakage checks.
"""
from __future__ import annotations

import math


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def holdout_passed(result: dict, cfg) -> bool:
    meta, m = result.get("meta", {}), result.get("final_metrics", {})
    dsr = result.get("dsr", {}).get("dsr")
    dsr_alpha = (result.get("dsr_alpha") or {}).get("dsr")
    pbo = result.get("spec", {}).get("pbo")
    lower = m.get("excess_ci_low")
    gate = result.get("spec", {}).get("research_gate", {})
    return bool(
        meta.get("source") not in (None, "synthetic")
        and meta.get("survivorship_free") is True
        and meta.get("pit_fundamentals") is True
        and not result.get("lock", {}).get("contaminated", True)
        and m.get("n_periods", 0) >= 24
        and _finite(lower) and lower > 0
        and _finite(dsr) and dsr >= cfg.validation.red_flags.min_dsr
        and _finite(dsr_alpha) and dsr_alpha >= cfg.validation.red_flags.min_dsr
        and _finite(pbo) and pbo <= cfg.validation.red_flags.max_pbo
        and not result.get("red_flags")
        and gate.get("full_validation") is True
        and gate.get("sanity_passed") is True
        and gate.get("blocking_flags") == []
    )


def system_validated(spec: dict, cfg, meta: dict) -> bool:
    dsr, pbo, dsr_alpha = spec.get("dsr"), spec.get("pbo"), spec.get("dsr_alpha")
    gate = spec.get("research_gate", {})
    return bool(
        meta.get("source") not in (None, "synthetic")
        and meta.get("survivorship_free") is True
        and meta.get("pit_fundamentals") is True
        and spec.get("config_hash") == cfg.hash()
        and spec.get("validation_protocol") == "real_pit_holdout_v1"
        and spec.get("holdout_passed") is True
        and gate.get("full_validation") is True
        and gate.get("sanity_passed") is True
        and gate.get("blocking_flags") == []
        and _finite(dsr) and dsr >= cfg.validation.red_flags.min_dsr
        and _finite(dsr_alpha) and dsr_alpha >= cfg.validation.red_flags.min_dsr
        and _finite(pbo) and pbo <= cfg.validation.red_flags.max_pbo
    )
