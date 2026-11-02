"""Signal metadata: family, publication status (for the haircut) and the expected sign.

The expected sign is used only for single-signal diagnostics (IC, decay,
long-short tests). Models learn their own weights.
"""
from __future__ import annotations

EXPECTED_SIGN = {
    "mom_12_1": +1, "mom_6_1": +1, "rev_1m": -1, "illiq": +1, "vol_252": -1, "idio_vol": -1,
    "size": -1, "value_bm": +1, "value_ep": +1, "quality_gpa": +1, "quality_roe": +1,
    "sue": +1, "pead": +1, "revisions": +1, "news_sent": +1,
    "regime_trend": +1, "regime_vol": -1, "regime_vix": -1,
}

TIME_SERIES_FEATURES = {"regime_trend", "regime_vol", "regime_vix"}


def haircut_factor(features, cfg, importance: dict | None = None) -> float:
    """Fraction of a backtested edge to keep after the post-publication haircut.

    With importance weights (e.g. mean |SHAP|), the haircut applies to the share
    of the prediction coming from published signals; otherwise to all of it if
    any feature is published.
    """
    h = float(cfg.signals.haircut)
    reg = cfg.signals.registry
    if importance:
        tot = sum(abs(v) for v in importance.values()) or 1.0
        pub_share = sum(abs(v) for k, v in importance.items() if reg.get(k, {}).get("published", True)) / tot
    else:
        pub_share = 1.0 if any(reg.get(f, {}).get("published", True) for f in features) else 0.0
    return 1.0 - h * pub_share


def publication_year(name: str, cfg):
    return cfg.signals.registry.get(name, {}).get("year")
