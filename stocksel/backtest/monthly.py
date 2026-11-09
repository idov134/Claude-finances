"""Fast monthly backtest (used for every trial): signal at month-end close, trade at next open,
hold to the next rebalance open; costs on actual turnover (after weight drift)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..portfolio.sizing import SizingParams, target_weights
from .costs import CostModel


@dataclass
class BacktestResult:
    monthly: pd.DataFrame
    weights: dict = field(default_factory=dict)
    contrib: pd.DataFrame | None = None
    diag: list = field(default_factory=list)

    @property
    def net(self) -> pd.Series:
        return self.monthly["net"]

    @property
    def bench(self) -> pd.Series:
        return self.monthly["bench"]


def market_vol(bundle, dates, window: int = 126) -> pd.Series:
    r = bundle.benchmark["close"].pct_change()
    return (r.rolling(window, min_periods=40).std() * np.sqrt(252)).reindex(dates).ffill().fillna(0.16)


def candidates_at(d, preds, panel, sectors: pd.Series) -> pd.DataFrame:
    """``preds`` is either the long prediction frame or a {date: frame} dict (faster in loops)."""
    if isinstance(preds, dict):
        c = preds.get(d)
        if c is None:
            return pd.DataFrame()
    else:
        if d not in preds.index.get_level_values(0):
            return pd.DataFrame()
        c = preds.xs(d, level=0)
    a = panel.aux
    data = {"prob": c["prob"].to_numpy(), "score": c["score"].to_numpy()}
    for k in ("mcap", "idio_vol", "beta"):
        data[k] = a[k].loc[d].reindex(c.index).to_numpy()
    data["sector"] = sectors.reindex(c.index).to_numpy()
    if "er" in c:
        data["er"] = c["er"].to_numpy()
    return pd.DataFrame(data, index=c.index)


def fee_reserved_weights(target: pd.Series, previous: pd.Series, adv: pd.Series, sigma: pd.Series, equity: float,
                         cost_model: CostModel):
    """Fund modeled costs from cash: a fully invested target is scaled down so cash never goes negative.

    Uses only signal-date liquidity inputs (``adv``/``sigma`` at the decision close). Returns the
    executable weights, the cost components as fractions of equity, and the traded weight.
    """
    index = target.index.union(previous.index)
    old = previous.reindex(index).fillna(0.0)
    wanted = target.reindex(index).fillna(0.0)
    a, v = adv.reindex(index).to_numpy(), sigma.reindex(index).to_numpy()

    def costs(w):
        usd = cost_model.cost_usd((w - old).abs().to_numpy() * equity, a, v)
        return {k: float(x.sum()) / equity for k, x in usd.items()}

    weights = wanted.copy()
    for _ in range(30):
        comp = costs(weights)
        fees = sum(comp.values())
        if weights.sum() + fees <= 1 + 1e-12:
            return weights[weights > 1e-10], comp, float((weights - old).abs().sum())
        weights = wanted * max(0.0, 1 - fees) / max(1.0, float(wanted.sum()))
    raise ValueError("could not reserve cash for modeled transaction costs")


def executable_targets(wanted: pd.Series, previous: pd.Series, entry_prices: pd.Series) -> tuple[pd.Series, int]:
    """A name with no executable opening price cannot be traded at the entry open.

    New buys stay in cash (no replacement name is picked after seeing the opens); an existing
    holding keeps its drifted weight. Uses only availability at execution, never exit returns.
    """
    index = wanted.index.union(previous.index)
    result = wanted.reindex(index).fillna(0.0)
    prices = entry_prices.reindex(index)
    unavailable = ~np.isfinite(prices) | (prices <= 0)
    result.loc[unavailable] = previous.reindex(index).fillna(0.0).loc[unavailable]
    n_unfilled = int((unavailable & (wanted.reindex(index).fillna(0.0) > 0)).sum())
    return result[result > 1e-10], n_unfilled


def monthly_backtest(preds: pd.DataFrame, panel, labels, bundle, cfg, prm: SizingParams,
                     regime_filter: bool = False, cost_multiplier: float = 1.0,
                     universe_mask: pd.DataFrame | None = None) -> BacktestResult:
    """Monthly portfolio simulation with strict accounting.

    * costs are paid from cash (fee reserve) using signal-date liquidity estimates;
    * an order without an executable next open is unfilled (cash), never a zero-return position;
    * a held name without a realized forward return is an error, not a zero;
    * weights drift on *net* equity; names delisted during the month leave as cash
      (their delisting proceeds are already in the forward return).
    """
    cm = CostModel(cfg.costs, cost_multiplier)
    by_date = {d: g.droplevel(0) for d, g in preds.groupby(level=0)}
    dates = [d for d in panel.dates if d in by_date and np.isfinite(labels.bench.get(d, np.nan))]
    mv = market_vol(bundle, pd.DatetimeIndex(dates))
    rf_ann = bundle.risk_free.reindex(pd.DatetimeIndex(dates), method="ffill").fillna(0.0) if bundle.risk_free is not None \
        else pd.Series(0.0, index=dates)
    if regime_filter:
        from ..features.regime import risk_off
        ro = risk_off(bundle, pd.DatetimeIndex(dates), cfg.portfolio.regime_filter.vix_threshold)
    V = float(cfg.capital.initial)
    w_prev = pd.Series(dtype=float)
    rows, weights, contrib, diags = [], {}, [], []
    for d in dates:
        cands = candidates_at(d, by_date, panel, bundle.sectors)
        if universe_mask is not None and len(cands):
            cands = cands[universe_mask.loc[d].reindex(cands.index).fillna(False).to_numpy()]
        p = prm
        if regime_filter and bool(ro.get(d, False)):
            p = SizingParams(**{**prm.__dict__, "regime_scale": cfg.portfolio.regime_filter.scale})
        wanted, diag = target_weights(cands, w_prev, p, float(mv.loc[d]))
        entry_px = bundle.open.loc[labels.entry.loc[d]] if pd.notna(labels.entry.loc[d]) else pd.Series(dtype=float)
        wanted, unfilled = executable_targets(wanted, w_prev, entry_px)
        w, costs, turnover = fee_reserved_weights(wanted, w_prev, panel.aux["adv"].loc[d], panel.aux["vol_d"].loc[d],
                                                  V, cm)
        diag.update(date=d, unfilled=unfilled)
        diags.append(diag)
        r = labels.fwd.loc[d].reindex(w.index)
        if r.isna().any():
            raise ValueError(f"{d.date()}: held names without a realized forward return: {list(r[r.isna()].index)[:5]}")
        rf_m = float(rf_ann.loc[d]) / 12
        gross = float((w * r).sum() + (1 - w.sum()) * rf_m)
        cost = sum(costs.values())
        net = gross - cost * (1 + rf_m)   # the fee reserve is spent at entry and earns no interest
        V *= 1 + net
        if V <= 0:
            raise ValueError("non-positive portfolio equity")
        rows.append({"date": d, "gross": gross, "cost": cost, "net": net, "bench": float(labels.bench.loc[d]),
                     "rf": rf_m, "turnover": turnover, "exposure": float(w.sum()), "n_pos": int(len(w)),
                     "unfilled": unfilled, **{f"cost_{k}": v for k, v in costs.items()}})
        weights[d] = w
        if len(w):
            contrib.append(pd.DataFrame({"date": d, "ticker": w.index, "weight": w.values,
                                         "ret": r.values, "contrib": (w * r).values}))
        w_prev = (w * (1 + r) / (1 + net)) if len(w) else w
        if len(w_prev) and pd.notna(labels.exit.loc[d]):
            exit_px = bundle.open.loc[labels.exit.loc[d]].reindex(w_prev.index)
            w_prev = w_prev[np.isfinite(exit_px) & (exit_px > 0)]   # delisted: proceeds already in r -> cash
    monthly = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["gross", "cost", "net", "bench", "rf", "turnover", "exposure", "n_pos"])
    return BacktestResult(monthly=monthly, weights=weights,
                          contrib=pd.concat(contrib, ignore_index=True) if contrib else None, diag=diags)


def holding_spells(res: BacktestResult, labels) -> pd.DataFrame:
    """Per-trade statistics from the weight history (a spell = consecutive months held)."""
    spells, open_ = [], {}
    dates = list(res.weights.keys())
    for d in dates:
        held = set(res.weights[d].index)
        for t in list(open_):
            if t not in held:
                spells.append(open_.pop(t))
        for t in held:
            r = labels.fwd.loc[d].get(t, np.nan)
            b = labels.bench.loc[d]
            s = open_.setdefault(t, {"ticker": t, "entry": d, "growth": 1.0, "bgrowth": 1.0, "months": 0})
            s["growth"] *= 1 + (0.0 if not np.isfinite(r) else r)
            s["bgrowth"] *= 1 + b
            s["months"] += 1
            s["exit"] = d
    spells.extend(open_.values())
    df = pd.DataFrame(spells)
    if len(df):
        df["ret"] = df["growth"] - 1
        df["excess"] = df["growth"] / df["bgrowth"] - 1
    return df


def trade_stats(spells: pd.DataFrame) -> dict:
    if spells is None or len(spells) == 0:
        return {"n_trades": 0}
    w, l = spells.loc[spells["excess"] > 0, "excess"], spells.loc[spells["excess"] <= 0, "excess"]
    return {"n_trades": int(len(spells)), "trade_hit_rate": float((spells["excess"] > 0).mean()),
            "trade_win_loss_ratio": float(w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else np.nan,
            "trade_expectancy": float(spells["excess"].mean()), "avg_holding_months": float(spells["months"].mean())}
