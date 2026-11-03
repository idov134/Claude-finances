"""Daily execution engine for the final strategy.

* target weights decided at a month-end close are executed at the next open;
* daily mark-to-market at the close; cash earns the risk-free rate;
* ATR stop (entry - k*ATR) and trailing stop (highest close - m*ATR) are checked
  at each close and executed at the next open; a stopped name is not re-bought
  until the next rebalance;
* delisting: the position is realized at the last close x (1 + delisting return);
* kill switch: drawdown beyond the limit liquidates at the next executable open
  and halts trading; optional pre-registered re-entry is disabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle, last_valid_positions
from .costs import CostModel


@dataclass
class EngineResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    kill_events: list = field(default_factory=list)
    reentry_events: list = field(default_factory=list)
    initial_equity: float | None = None

    def monthly(self) -> pd.DataFrame:
        eq = self.daily[["equity", "bench"]].resample("ME").last()
        first = self.daily[["equity", "bench"]].iloc[[0]].copy()
        if self.initial_equity is not None:
            first.loc[:, :] = self.initial_equity
        eq = pd.concat([first.set_axis([first.index[0] - pd.Timedelta(days=1)]), eq])
        m = eq.pct_change().dropna()
        m.columns = ["net", "bench"]
        return m


def _atr(bundle: DataBundle, n: int) -> np.ndarray:
    c, h, l = bundle.close.to_numpy(), bundle.high.to_numpy(), bundle.low.to_numpy()
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    tr = np.fmax(np.fmax(h - l, np.abs(h - prev)), np.abs(l - prev))
    return pd.DataFrame(tr).rolling(n, min_periods=max(3, n // 2)).mean().to_numpy()


def run_daily(bundle: DataBundle, targets: dict, cfg, use_stops: bool | None = None,
              kill_switch: bool | None = None, kill_threshold: float | None = None,
              cost_multiplier: float = 1.0) -> EngineResult:
    from ..config import kill_drawdown
    ex = cfg.execution
    use_stops = ex.use_stops if use_stops is None else use_stops
    kill_switch = cfg.risk.kill_switch.simulate_in_backtest if kill_switch is None else kill_switch
    kill_threshold = kill_drawdown(cfg) if kill_threshold is None else kill_threshold
    reentry = cfg.risk.kill_switch.get("reentry", {})
    allow_reentry = bool(reentry.get("enabled", False))
    cooldown = int(reentry.get("cooldown_days", 21))
    trend_days = int(reentry.get("trend_days", 126))
    recovery = float(reentry.get("recovery_fraction", 0.05))
    if allow_reentry and (cooldown < 1 or trend_days < 2 or not 0 <= recovery < 1):
        raise ValueError("reentry requires cooldown_days >= 1, trend_days >= 2, recovery_fraction in [0, 1)")
    cm = CostModel(cfg.costs, cost_multiplier)
    dates = bundle.dates
    tick = bundle.tickers
    O, C = bundle.open.to_numpy(), bundle.close.to_numpy()
    ATR = _atr(bundle, int(ex.atr_period))
    ADV = bundle.adv_usd(63).to_numpy()
    VOL = bundle.close.pct_change(fill_method=None).rolling(63, min_periods=21).std().to_numpy()
    LV = last_valid_positions(bundle.close)
    DR = bundle.delist_returns.reindex(tick).fillna(cfg.data.default_delisting_return).to_numpy()
    rf = bundle.rf_daily().to_numpy()
    bo, bc = bundle.benchmark["open"].to_numpy(), bundle.benchmark["close"].to_numpy()
    trend = bundle.benchmark["close"].rolling(trend_days, min_periods=trend_days).mean().to_numpy()

    dec = sorted(d for d in targets if d in dates)
    exec_day = {}
    for d in dec:
        p = dates.searchsorted(d, side="right")
        if p < len(dates):
            exec_day[p] = targets[d]
    if not exec_day:
        raise ValueError("no executable targets")
    t0 = min(exec_day)
    N = len(tick)
    shares = np.zeros(N)
    last_px = np.full(N, np.nan)
    cash = float(cfg.capital.initial)
    pos: dict[int, dict] = {}
    pending: dict[int, str] = {}
    blocked: set[int] = set()
    killed = False
    killed_day, killed_bench = None, None
    peak = cash
    kill_events, reentry_events, trades, rows = [], [], [], []
    k_atr, k_tr = float(ex.atr_stop_mult), float(ex.trailing_atr_mult)

    def trade(i, dsh, t, reason):
        nonlocal cash
        px = O[t, i]
        if not np.isfinite(px) or dsh == 0:
            return 0.0
        notional = abs(dsh) * px
        # Closing volume/volatility of execution day t are unavailable at its
        # open. Use only the last completed session's estimates.
        prior_adv = ADV[t - 1, i] if t > 0 else np.nan
        prior_vol = VOL[t - 1, i] if t > 0 else np.nan
        c = cm.cost_usd(np.array([notional]), np.array([prior_adv]), np.array([prior_vol]))
        cost = float(sum(v[0] for v in c.values()))
        cash -= dsh * px + cost
        if shares[i] == 0 and dsh > 0:
            pos[i] = {"ticker": tick[i], "entry_date": dates[t], "entry_px": px, "cost_in": notional + cost,
                      "cash_out": 0.0, "high": px, "atr_entry": ATR[t - 1, i] if t > 0 else np.nan,
                      "bench_entry": bo[t], "costs": cost}
        elif i in pos:
            p = pos[i]
            if dsh > 0:
                p["entry_px"] = (p["entry_px"] * shares[i] + px * dsh) / (shares[i] + dsh)
                p["cost_in"] += notional + cost
            else:
                p["cash_out"] += notional - cost
            p["costs"] += cost
        shares[i] += dsh
        if abs(shares[i]) < 1e-9:
            shares[i] = 0.0
            p = pos.pop(i, None)
            if p:
                trades.append(_close(p, dates[t], px, reason, bo[t]))
        return cost

    for t in range(t0, len(dates)):
        day_cost = 0.0
        # ---------------- open: rebalance / stop exits / kill
        open_px = np.where(np.isfinite(O[t]), O[t], last_px)
        held_val = np.nansum(shares * open_px)
        eq_open = cash + held_val
        # Re-entry is evaluated using t-1 data at an existing rebalance. Never
        # cancel an outstanding liquidation or revive positions without a quote.
        if (killed and allow_reentry and t in exec_day and not pending
                and not np.any(shares != 0) and killed_day is not None
                and t - killed_day >= cooldown and t > 0
                and np.isfinite(bc[t - 1]) and np.isfinite(trend[t - 1])
                and bc[t - 1] > trend[t - 1]
                and bc[t - 1] >= killed_bench * (1 + recovery)):
            killed = False
            peak = eq_open  # cycle risk budget; lifetime DD remains in equity
            reentry_events.append({"date": dates[t], "equity": eq_open,
                                   "previous_kill_date": dates[killed_day],
                                   "benchmark_prior_close": bc[t - 1]})
        target_sh = None
        if t in exec_day and not killed:
            blocked.clear()
            w = exec_day[t]
            target_sh = shares.copy() * 0
            idx = tick.get_indexer(w.index)
            ok = (idx >= 0) & np.isfinite(O[t, np.maximum(idx, 0)])
            target_sh[idx[ok]] = w.to_numpy()[ok] * eq_open / O[t, idx[ok]]
            # names we cannot trade today keep their current shares
            keep = ~np.isfinite(O[t]) & (shares != 0)
            target_sh[keep] = shares[keep]
        for i, why in list(pending.items()):
            if shares[i] != 0 and np.isfinite(O[t, i]):
                day_cost += trade(i, -shares[i], t, why)
                blocked.add(i)
                pending.pop(i, None)
            elif shares[i] == 0:
                pending.pop(i, None)
        if target_sh is not None:
            target_sh[list(blocked)] = 0.0
            delta = target_sh - shares
            for i in np.where(delta < -1e-9)[0]:          # sells first (frees cash)
                day_cost += trade(i, delta[i], t, "rebalance")
            for i in np.where(delta > 1e-9)[0]:
                day_cost += trade(i, delta[i], t, "rebalance")
        # ---------------- close: delistings, mark-to-market, stops, kill switch
        for i in np.where((shares != 0) & ~np.isfinite(C[t]))[0]:
            if LV[i] < t:  # delisted for good
                proceeds = shares[i] * last_px[i] * (1 + DR[i])
                cash += proceeds
                p = pos.pop(i, None)
                if p:
                    p["cash_out"] += proceeds
                    trades.append(_close(p, dates[t], last_px[i] * (1 + DR[i]), "delisted", bc[t], sold=False))
                shares[i] = 0.0
        last_px = np.where(np.isfinite(C[t]), C[t], last_px)
        cash *= 1 + rf[t]
        equity = cash + np.nansum(shares * last_px)
        peak = max(peak, equity)
        if use_stops and not killed:
            for i in list(pos):
                p = pos[i]
                if not np.isfinite(C[t, i]):
                    continue
                p["high"] = max(p["high"], C[t, i])
                a_e = p["atr_entry"] if np.isfinite(p["atr_entry"]) else ATR[t, i]
                stop = max(p["entry_px"] - k_atr * a_e, p["high"] - k_tr * ATR[t, i]) if np.isfinite(ATR[t, i]) else -np.inf
                p["stop"] = stop
                if C[t, i] < stop:
                    pending[i] = "atr_stop" if stop == p["entry_px"] - k_atr * a_e else "trailing_stop"
        if kill_switch and not killed and equity / peak - 1 <= -kill_threshold:
            killed = True
            killed_day, killed_bench = t, bc[t]
            kill_events.append({"date": dates[t], "equity": equity, "drawdown": equity / peak - 1})
            for i in np.where(shares != 0)[0]:
                pending[i] = "kill_switch"
        rows.append({"date": dates[t], "equity": equity, "cash": cash, "bench": bc[t],
                     "exposure": float(np.nansum(shares * last_px) / equity) if equity > 0 else 0.0,
                     "n_pos": int((shares != 0).sum()), "cost": day_cost, "killed": killed})
    daily = pd.DataFrame(rows).set_index("date")
    # Compare from the first execution open, retaining its intraday movement.
    daily["bench"] = daily["bench"] / bo[t0] * float(cfg.capital.initial)
    for i, p in pos.items():  # mark open positions
        trades.append(_close(dict(p, cash_out=p["cash_out"] + shares[i] * last_px[i]), dates[-1], last_px[i],
                             "open", bc[-1], sold=False))
    return EngineResult(daily=daily, trades=pd.DataFrame(trades), kill_events=kill_events,
                        reentry_events=reentry_events, initial_equity=float(cfg.capital.initial))


def _close(p: dict, date, px, reason, bench_px, sold: bool = True) -> dict:
    ret = p["cash_out"] / p["cost_in"] - 1 if p["cost_in"] > 0 else np.nan
    b = bench_px / p["bench_entry"] - 1 if p.get("bench_entry") else np.nan
    return {"ticker": p["ticker"], "entry_date": p["entry_date"], "exit_date": date, "entry_px": p["entry_px"],
            "exit_px": px, "reason": reason, "ret": ret, "bench_ret": b,
            "excess": (1 + ret) / (1 + b) - 1 if np.isfinite(b) else np.nan, "costs": p["costs"]}
