"""Position sizing: selection with buffer, fractional Kelly caps, weight caps, vol targeting.

Pipeline for one rebalance date
1. rank candidates by model score; select top N; with ``buffer`` keep existing
   holdings while rank <= N * exit_rank_mult (fewer trades, lower costs);
   drop names whose haircut probability fell below ``min_prob`` (signal decay).
2. haircut probability  p_h = 0.5 + (p - 0.5) * keep   (keep = 1 - haircut share)
3. fractional Kelly cap f_i = k * mu_i / sigma_i^2 with
   mu_i = (2 p_h - 1) * E|r_i| = (2 p_h - 1) * sigma_i * sqrt(2/pi), sigma_i = monthly idio vol,
   k <= 0.25. f_i <= 0 => no position.
4. base weights (capped value-weight by default; value / equal / kelly variants),
   water-filled under per-name cap min(max_position, f_i) and sector caps; leftover -> cash.
5. portfolio vol (one-factor covariance) scaled to the vol target (never levered),
   optional regime-filter scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SQRT_2_PI = np.sqrt(2 / np.pi)


@dataclass
class SizingParams:
    top_n: int = 50
    buffer: bool = True
    exit_rank_mult: float = 1.6
    weighting: str = "capped_value"
    max_position: float = 0.05
    max_sector: float = 0.30
    max_gross: float = 1.0
    kelly_fraction: float = 0.25
    keep: float = 0.5                  # 1 - haircut
    vol_target: float = 0.10
    min_prob: float = 0.5
    regime_scale: float = 1.0          # < 1 in risk-off when the regime filter is on
    mandate: str = "absolute"          # absolute: Kelly caps + vol target | relative: fully invested

    @classmethod
    def from_cfg(cls, cfg, keep: float, vol_target: float, **over):
        p = cfg.portfolio
        kw = dict(top_n=p.top_n, buffer=p.buffer, exit_rank_mult=p.exit_rank_mult, weighting=p.weighting,
                  max_position=p.max_position, max_sector=p.max_sector, max_gross=p.max_gross,
                  kelly_fraction=min(0.25, p.kelly_fraction), keep=keep, vol_target=vol_target,
                  min_prob=p.min_prob, mandate=p.get("mandate", "absolute") or "absolute")
        kw.update(over)
        return cls(**kw)


def haircut_prob(p, keep):
    return 0.5 + (np.asarray(p, float) - 0.5) * keep


def kelly_cap(p_h, idio_vol_annual, fraction):
    sig = np.maximum(np.asarray(idio_vol_annual, float) / np.sqrt(12), 0.01)
    mu = (2 * np.asarray(p_h, float) - 1) * sig * SQRT_2_PI
    return fraction * mu / sig ** 2


def select(cands: pd.DataFrame, prev: pd.Series, prm: SizingParams) -> pd.Index:
    rank = cands["score"].rank(ascending=False, method="first")
    chosen = rank <= prm.top_n
    if prm.buffer and prev is not None and len(prev):
        held = cands.index.isin(prev[prev > 0].index)
        chosen |= held & (rank <= prm.top_n * prm.exit_rank_mult)
    if prm.mandate != "relative":   # the relative mandate ranks; it does not wait for an absolute edge
        ph = haircut_prob(cands["prob"], prm.keep)
        chosen &= ph > prm.min_prob
    return cands.index[chosen]


def waterfill(base: pd.Series, cap: pd.Series, sectors: pd.Series, max_sector: float, total: float,
              iters: int = 50) -> pd.Series:
    """Distribute ``total`` proportionally to ``base`` under per-name caps and sector caps (numpy)."""
    b = np.nan_to_num(base.to_numpy(dtype=float), nan=0.0).clip(min=0)
    cp = cap.to_numpy(dtype=float)
    codes, _ = pd.factorize(sectors.to_numpy())
    if b.sum() <= 0:
        return pd.Series(0.0, index=base.index)
    w = b / b.sum() * total
    frozen = np.zeros(len(w), bool)
    for _ in range(iters):
        over = w > cp + 1e-12
        excess = float((w[over] - cp[over]).sum())
        w[over] = cp[over]
        frozen |= over
        sec = np.bincount(codes, weights=w)
        for sidx in np.where(sec > max_sector + 1e-12)[0]:
            m = codes == sidx
            excess += float(w[m].sum() - max_sector)
            w[m] *= max_sector / w[m].sum()
            frozen |= m
        if excess <= 1e-10:
            break
        free = ~frozen & (b > 0)
        if not free.any():
            break
        w[free] += excess * b[free] / b[free].sum()
    return pd.Series(w.clip(min=0), index=base.index)


def portfolio_vol(w: pd.Series, beta: pd.Series, idio_ann: pd.Series, mkt_vol_ann: float) -> float:
    b = beta.reindex(w.index).fillna(1.0).to_numpy()
    s = idio_ann.reindex(w.index).fillna(idio_ann.median() if len(idio_ann) else 0.3).to_numpy()
    x = w.to_numpy()
    var = (x @ b) ** 2 * mkt_vol_ann ** 2 + np.sum(x ** 2 * s ** 2)
    return float(np.sqrt(max(var, 0)))


def target_weights(cands: pd.DataFrame, prev: pd.Series, prm: SizingParams, mkt_vol_ann: float) -> tuple[pd.Series, dict]:
    """cands index=ticker; columns: prob, score, mcap, sector, idio_vol, beta."""
    diag = {"n_candidates": len(cands)}
    if len(cands) == 0:
        return pd.Series(dtype=float), diag
    if prm.mandate not in ("absolute", "relative"):
        raise ValueError("portfolio.mandate must be 'absolute' or 'relative'")
    names = select(cands, prev, prm)
    c = cands.loc[names]
    ph = haircut_prob(c["prob"], prm.keep)
    f = pd.Series(kelly_cap(ph, c["idio_vol"].fillna(c["idio_vol"].median()), prm.kelly_fraction), index=c.index)
    if prm.mandate == "relative":
        # Benchmark-relative mandate: fully invested in the selected names; Kelly does not cap
        # weights and there is no absolute volatility target (drawdowns then resemble the index's).
        f = pd.Series(np.inf, index=c.index)
    c, f = c[f > 0], f[f > 0]
    diag["n_selected"] = len(c)
    if len(c) == 0:
        return pd.Series(dtype=float), diag
    if prm.weighting in ("capped_value", "value"):
        base = c["mcap"].clip(lower=0).fillna(0)
    elif prm.weighting == "equal":
        base = pd.Series(1.0, index=c.index)
    elif prm.weighting == "kelly":
        base = f.copy()
    else:
        raise ValueError(prm.weighting)
    pos_cap = 1.0 if prm.weighting == "value" else prm.max_position
    cap = np.minimum(f, pos_cap)
    w = waterfill(base, cap, c["sector"].fillna("Unknown"), prm.max_sector, prm.max_gross)
    diag["kelly_binding"] = int((np.isclose(w, f) & (f < pos_cap)).sum())
    diag["gross_pre_vol"] = float(w.sum())
    vol = portfolio_vol(w, c["beta"], c["idio_vol"], mkt_vol_ann)
    scale = min(1.0, prm.vol_target / vol) if vol > 0 and prm.mandate != "relative" else 1.0
    scale *= prm.regime_scale
    w = w * scale
    diag.update(est_vol=vol, vol_scale=scale, gross=float(w.sum()))
    return w[w > 1e-6], diag
