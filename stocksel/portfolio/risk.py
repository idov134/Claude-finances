"""Kill switch: drawdown breach, or live performance deviating from the backtest distribution."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KillDecision:
    kill: bool
    reason: str
    stats: dict


def drawdown_breach(equity: pd.Series, threshold: float) -> KillDecision:
    dd = float((equity / equity.cummax() - 1).iloc[-1]) if len(equity) else 0.0
    return KillDecision(dd <= -threshold, f"drawdown {dd:.1%} vs limit {-threshold:.0%}", {"drawdown": dd})


def deviation_test(live_monthly: pd.Series, bt_monthly: pd.Series, z_crit: float = -2.0, min_months: int = 3) -> KillDecision:
    """One-sided z-test of the live mean monthly return against the backtest distribution."""
    live = live_monthly.dropna()
    n = len(live)
    if n < min_months:
        return KillDecision(False, f"only {n} live months (< {min_months})", {"n": n})
    mu, sd = float(bt_monthly.mean()), float(bt_monthly.std(ddof=1))
    z = (float(live.mean()) - mu) / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return KillDecision(z < z_crit, f"live mean deviates: z={z:.2f} (limit {z_crit})",
                        {"z": z, "n": n, "live_mean": float(live.mean()), "bt_mean": mu, "bt_sd": sd})


def evaluate(equity: pd.Series, live_monthly: pd.Series, bt_monthly: pd.Series, cfg) -> KillDecision:
    from ..config import kill_drawdown
    ks = cfg.risk.kill_switch
    d1 = drawdown_breach(equity, kill_drawdown(cfg))
    if d1.kill:
        return d1
    d2 = deviation_test(live_monthly, bt_monthly, ks.deviation_z, ks.min_live_months)
    return d2 if d2.kill else KillDecision(False, "ok", {**d1.stats, **d2.stats})
