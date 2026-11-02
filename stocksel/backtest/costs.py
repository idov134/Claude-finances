"""Transaction cost model: commission + half-spread (liquidity dependent) + slippage + market impact.

impact follows the square-root law: k * sigma_daily * sqrt(trade_value / ADV).
All rates are fractions of traded notional. ``multiplier`` makes the model
harsher (used for the daily news strategy).
"""
from __future__ import annotations

import numpy as np


class CostModel:
    def __init__(self, costs_cfg, multiplier: float = 1.0):
        self.c = costs_cfg
        self.m = float(multiplier)

    def half_spread(self, adv_usd):
        adv_m = np.maximum(np.asarray(adv_usd, dtype=float), 1e5) / 1e6
        bps = np.maximum(self.c.min_half_spread_bps, self.c.spread_k / np.sqrt(adv_m))
        return bps / 1e4

    def components(self, trade_usd, adv_usd, daily_vol):
        trade = np.abs(np.asarray(trade_usd, dtype=float))
        adv = np.maximum(np.nan_to_num(np.asarray(adv_usd, dtype=float), nan=1e6), 1e5)
        vol = np.nan_to_num(np.asarray(daily_vol, dtype=float), nan=0.02)
        comm = np.full_like(trade, self.c.commission_bps / 1e4)
        spread = self.half_spread(adv)
        slip = np.full_like(trade, self.c.slippage_bps / 1e4)
        impact = self.c.impact_k * vol * np.sqrt(trade / adv)
        return {"commission": comm * self.m, "spread": spread * self.m, "slippage": slip * self.m,
                "impact": impact * self.m}

    def rate(self, trade_usd, adv_usd, daily_vol):
        comp = self.components(trade_usd, adv_usd, daily_vol)
        return sum(comp.values())

    def cost_usd(self, trade_usd, adv_usd, daily_vol):
        comp = self.components(trade_usd, adv_usd, daily_vol)
        trade = np.abs(np.asarray(trade_usd, dtype=float))
        return {k: v * trade for k, v in comp.items()}
