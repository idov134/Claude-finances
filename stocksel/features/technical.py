"""Classic technical indicators and simple long/flat trading rules, vectorized over assets.

Every rule returns a **position decided at the close** of each day: 1 = long, 0 = cash, NaN = not
enough history to compute it. The rule engine (``backtest/rules.py``) executes it at the next open,
so nothing here may look past the close of the day it is computed for. The tests check this by
changing future prices and verifying that past positions do not move.

Parameters are the textbook ones, not fitted here:

| Rule | Definition | Source of the thresholds |
|---|---|---|
| SMA150 | close > 150-day SMA | Weinstein (1988), 30-week MA |
| SMA200 | close > 200-day SMA | the classic long-term trend line |
| GOLDEN | 50-day SMA > 200-day SMA | "golden cross" |
| FABER10 | month-end close > 10-month SMA, held for the next month | Faber (2007) |
| MACD | MACD(12, 26) > its 9-day signal line | Appel |
| TEMPLATE | Minervini's trend template without the relative-strength rank | Minervini (2013) |
| RSI14 | enter RSI(14) < 30, exit RSI(14) > 70 | Wilder (1978) |
| RSI2 | enter close > SMA200 and RSI(2) < 10, exit close > 5-day SMA | Connors & Alvarez (2008) |
| BB_MR | enter close < lower Bollinger band (20, 2), exit close > middle band | Bollinger |
| BB_MR_TREND | BB_MR, entries only while close > SMA200 | Bollinger + trend filter |
| BB_BREAK | enter close > upper band, exit close < middle band | Bollinger breakout |
| HIGH52 | enter on a new 52-week closing high, exit close < 50-day SMA | 52-week-high breakout |
| TOM | long on the last trading day of a month and the first three of the next | Ariel (1987), Lakonishok & Smidt (1988) |
| HALLOWEEN | long November-April, cash May-October | Bouman & Jacobsen (2002) |
| VIX30 | long while the VIX closes below 30 | common "high fear" threshold |
| BREADTH | long while at least 5 of the 9 sector SPDRs close above their 150-day SMA | sector breadth, Weinstein-style |
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Frame = pd.DataFrame


# --------------------------------------------------------------------------- indicators
def sma(x: Frame, n: int) -> Frame:
    return x.rolling(n, min_periods=n).mean()


def ema(x: Frame, n: int) -> Frame:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(x: Frame, n: int = 14) -> Frame:
    """Wilder's RSI: exponentially smoothed average gain / loss with alpha = 1/n."""
    d = x.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + au / ad)
    out = out.mask((ad == 0) & (au > 0), 100.0).mask((ad == 0) & (au == 0), 50.0)
    return out.where(au.notna() & ad.notna())


def bollinger(x: Frame, n: int = 20, k: float = 2.0) -> tuple[Frame, Frame, Frame]:
    """(middle, upper, lower); the band width uses the population standard deviation, as Bollinger does."""
    mid = sma(x, n)
    sd = x.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def macd(x: Frame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Frame, Frame]:
    line = ema(x, fast) - ema(x, slow)
    return line, line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def rolling_high(x: Frame, n: int = 252) -> Frame:
    return x.rolling(n, min_periods=n).max()


def rolling_low(x: Frame, n: int = 252) -> Frame:
    return x.rolling(n, min_periods=n).min()


# --------------------------------------------------------------------------- helpers
def hold(entry: Frame, exit_: Frame, valid: Frame) -> Frame:
    """Long from an entry day until an exit day (state machine). Entry wins if both hold on a day."""
    state = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
    state = state.mask(exit_.fillna(False).astype(bool), 0.0).mask(entry.fillna(False).astype(bool), 1.0)
    return state.ffill().fillna(0.0).where(valid)


def flag(cond: Frame, valid: Frame) -> Frame:
    return cond.astype(float).where(valid)


def month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month. The final row is never a month end (month incomplete)."""
    m = pd.Series(index.month, index=index)
    return (m != m.shift(-1)) & m.shift(-1).notna()


def turn_of_month_window(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of a month and the first three trading days of the next."""
    s = pd.Series(index, index=index)
    ym = s.dt.year * 12 + s.dt.month
    pos = ym.groupby(ym).cumcount()                       # 0 = first trading day of the month
    last = month_end_mask(index)
    return last | (pos <= 2)


def broadcast(s: pd.Series, like: Frame) -> Frame:
    return pd.DataFrame(np.repeat(s.reindex(like.index).to_numpy()[:, None], like.shape[1], axis=1),
                        index=like.index, columns=like.columns)


# --------------------------------------------------------------------------- rules
@dataclass
class Market:
    """Inputs shared by the market-level rules (aligned to the asset dates by the caller)."""
    vix: pd.Series
    breadth: pd.Series          # number of sector SPDRs above their 150-day SMA (NaN in warm-up)


class Indicators:
    """Lazily computed, cached indicators of one close-price frame (dates x assets)."""

    def __init__(self, close: Frame):
        self.c = close
        self._cache: dict = {}

    def get(self, key: str):
        if key not in self._cache:
            c = self.c
            fn = {"sma5": lambda: sma(c, 5), "sma50": lambda: sma(c, 50), "sma150": lambda: sma(c, 150),
                  "sma200": lambda: sma(c, 200), "rsi14": lambda: rsi(c, 14), "rsi2": lambda: rsi(c, 2),
                  "bb": lambda: bollinger(c, 20, 2.0), "macd": lambda: macd(c),
                  "hi252": lambda: rolling_high(c, 252), "lo252": lambda: rolling_low(c, 252)}[key]
            self._cache[key] = fn()
        return self._cache[key]


def _faber10(c: Frame) -> Frame:
    me = month_end_mask(c.index)
    mc = c[me.to_numpy()]
    sig = (mc > mc.rolling(10, min_periods=10).mean()).astype(float).where(mc.rolling(10, min_periods=10)
                                                                            .mean().notna())
    out = sig.reindex(c.index).ffill()
    return out.where(c.notna())


def _template(ind: Indicators) -> Frame:
    c = ind.c
    s50, s150, s200 = ind.get("sma50"), ind.get("sma150"), ind.get("sma200")
    hi, lo = ind.get("hi252"), ind.get("lo252")
    cond = ((c > s150) & (c > s200) & (s150 > s200) & (s200 > s200.shift(21)) & (s50 > s150) & (s50 > s200)
            & (c > s50) & (c >= 1.30 * lo) & (c >= 0.75 * hi))
    valid = s200.shift(21).notna() & hi.notna() & c.notna()
    return flag(cond, valid)


def rule_positions(name: str, ind: Indicators, mkt: Market) -> Frame:
    """Position decided at each close for rule ``name`` (see module docstring)."""
    c = ind.c
    ok = c.notna()
    if name == "SMA150":
        s = ind.get("sma150"); return flag(c > s, s.notna() & ok)
    if name == "SMA200":
        s = ind.get("sma200"); return flag(c > s, s.notna() & ok)
    if name == "GOLDEN":
        a, b = ind.get("sma50"), ind.get("sma200"); return flag(a > b, b.notna() & ok)
    if name == "FABER10":
        return _faber10(c)
    if name == "MACD":
        line, sig = ind.get("macd"); return flag(line > sig, sig.notna() & ok)
    if name == "TEMPLATE":
        return _template(ind)
    if name == "RSI14":
        r = ind.get("rsi14"); return hold(r < 30, r > 70, r.notna() & ok)
    if name == "RSI2":
        r, s200, s5 = ind.get("rsi2"), ind.get("sma200"), ind.get("sma5")
        return hold((c > s200) & (r < 10), c > s5, r.notna() & s200.notna() & ok)
    if name in ("BB_MR", "BB_MR_TREND", "BB_BREAK"):
        mid, up, lo = ind.get("bb")
        valid = mid.notna() & ok
        if name == "BB_MR":
            return hold(c < lo, c > mid, valid)
        if name == "BB_MR_TREND":
            s200 = ind.get("sma200")
            return hold((c > s200) & (c < lo), c > mid, valid & s200.notna())
        return hold(c > up, c < mid, valid)
    if name == "HIGH52":
        hi, s50 = ind.get("hi252"), ind.get("sma50")
        return hold(c >= hi, c < s50, hi.notna() & s50.notna() & ok)
    if name == "TOM":
        nxt = turn_of_month_window(c.index).shift(-1).fillna(False).astype(bool)  # tomorrow is in the window
        return flag(broadcast(nxt, c).astype(bool), ok)
    if name == "HALLOWEEN":
        months = pd.Series(c.index.month, index=c.index).shift(-1)                # tomorrow's month
        nxt = months.isin([11, 12, 1, 2, 3, 4]) & months.notna()
        return flag(broadcast(nxt, c).astype(bool), ok)
    if name == "VIX30":
        v = mkt.vix.reindex(c.index).ffill()
        return flag(broadcast(v < 30, c).astype(bool), broadcast(v.notna(), c).astype(bool) & ok)
    if name == "BREADTH":
        b = mkt.breadth.reindex(c.index)
        return flag(broadcast(b >= 5, c).astype(bool), broadcast(b.notna(), c).astype(bool) & ok)
    raise KeyError(name)


RULES = ["SMA150", "SMA200", "GOLDEN", "FABER10", "MACD", "TEMPLATE", "RSI14", "RSI2", "BB_MR", "BB_MR_TREND",
         "BB_BREAK", "HIGH52", "TOM", "HALLOWEEN", "VIX30", "BREADTH"]
FAMILY = {"SMA150": "trend", "SMA200": "trend", "GOLDEN": "trend", "FABER10": "trend", "MACD": "trend",
          "TEMPLATE": "trend", "RSI14": "mean_reversion", "RSI2": "mean_reversion", "BB_MR": "mean_reversion",
          "BB_MR_TREND": "mean_reversion", "BB_BREAK": "breakout", "HIGH52": "breakout", "TOM": "calendar",
          "HALLOWEEN": "calendar", "VIX30": "regime", "BREADTH": "regime"}


def sector_breadth(etf_close: Frame, sector_etfs: list[str], n: int = 150) -> pd.Series:
    """Number of sector ETFs closing above their n-day SMA; NaN until every SMA exists."""
    c = etf_close[sector_etfs]
    s = sma(c, n)
    above = (c > s).sum(axis=1).astype(float)
    return above.where(s.notna().all(axis=1))


# --------------------------------------------------------------------------- cross-sectional features
def cross_sectional_features(ind: Indicators) -> dict[str, Frame]:
    """Continuous versions of the rules, for monthly cross-sectional rank IC."""
    c = ind.c
    mid, up, lo = ind.get("bb")
    line, sig = ind.get("macd")
    return {"rsi14": ind.get("rsi14"),
            "bb_pctb": (c - lo) / (up - lo).replace(0, np.nan),
            "dist_sma150": c / ind.get("sma150") - 1,
            "dist_sma200": c / ind.get("sma200") - 1,
            "macd_hist": (line - sig) / c,
            "prox_52w_high": c / ind.get("hi252")}
