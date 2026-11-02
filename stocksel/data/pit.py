"""Point-in-time joins: map event tables onto decision dates using *availability* time.

Rules (the core of leakage prevention for non-price data):
* date-only availability (e.g. a 10-Q ``publish_date``): usable at the close of
  day D only if ``publish_date + lag_days <= D``. With the default lag of 1 day,
  a filing dated D is first used at the close of D+1.
* timestamped availability (earnings ``announce_ts``, news ``timestamp``): usable
  at the close of day D only if the timestamp is strictly before D's close time
  in the market time zone. An 8-K at 16:30 on D is first used at the close of D+1.
"""
from __future__ import annotations

import pandas as pd


def decision_timestamps(dates: pd.DatetimeIndex, close_time: str = "16:00") -> pd.DatetimeIndex:
    hh, mm = (int(x) for x in close_time.split(":"))
    return pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=hh, minutes=mm)


def availability(series: pd.Series, tz: str, lag_days: int = 0) -> pd.Series:
    """Naive local-time availability of each event."""
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(tz).dt.tz_localize(None)
    return ts.dt.normalize() + pd.Timedelta(days=lag_days)


def asof_wide(events: pd.DataFrame, time_col: str, value_cols: list[str], decision_dates: pd.DatetimeIndex,
              tickers, tz: str = "America/New_York", close_time: str = "16:00", lag_days: int = 0,
              return_avail: bool = False) -> dict[str, pd.DataFrame]:
    """Latest event values known strictly before each decision time, as wide frames."""
    decision_dates = pd.DatetimeIndex(decision_dates)
    tickers = pd.Index(tickers)
    out: dict[str, pd.DataFrame] = {}
    empty = pd.DataFrame(index=decision_dates, columns=tickers, dtype=float)
    if events is None or len(events) == 0:
        return {c: empty.copy() for c in value_cols + (["avail"] if return_avail else [])}
    ev = events.loc[events["ticker"].isin(tickers), ["ticker", time_col] + value_cols].copy()
    ev["avail"] = availability(ev[time_col], tz, lag_days).astype("datetime64[ns]")
    ev = ev.dropna(subset=["avail"]).sort_values("avail", kind="stable")
    dts = decision_timestamps(decision_dates, close_time)
    left = pd.DataFrame({"dt": dts.repeat(len(tickers)).astype("datetime64[ns]"),
                         "date": decision_dates.repeat(len(tickers)),
                         "ticker": list(tickers) * len(decision_dates)})
    left = left.sort_values("dt", kind="stable")
    ev["ticker"] = ev["ticker"].astype(object)
    left["ticker"] = left["ticker"].astype(object)
    merged = pd.merge_asof(left, ev.drop(columns=[time_col]), left_on="dt", right_on="avail", by="ticker",
                           direction="backward", allow_exact_matches=False)
    cols = value_cols + (["avail"] if return_avail else [])
    for c in cols:
        w = merged.pivot(index="date", columns="ticker", values=c)
        out[c] = w.reindex(index=decision_dates, columns=tickers)
    return out
