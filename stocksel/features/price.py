"""Price/volume signals. Every value at date t uses closes up to and including t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle


def daily_price_frames(bundle: DataBundle) -> dict[str, pd.DataFrame]:
    c, h, l, v = bundle.close, bundle.high, bundle.low, bundle.volume
    r = c.pct_change(fill_method=None)
    rm = bundle.benchmark["close"].pct_change()
    out = {}
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    out["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    out["rev_1m"] = c / c.shift(21) - 1
    out["vol_252"] = r.rolling(252, min_periods=126).std() * np.sqrt(252)
    w = 63
    mp = 42
    rr = r.mul(rm, axis=0)
    m_i = r.rolling(w, min_periods=mp).mean()
    m_m = rm.rolling(w, min_periods=mp).mean()
    cov = rr.rolling(w, min_periods=mp).mean() - m_i.mul(m_m, axis=0)
    var_m = rm.rolling(w, min_periods=mp).var(ddof=0)
    beta = cov.div(var_m, axis=0)
    var_i = r.rolling(w, min_periods=mp).var(ddof=0)
    idio_var = (var_i - beta.pow(2).mul(var_m, axis=0)).clip(lower=0)
    out["idio_vol"] = np.sqrt(idio_var) * np.sqrt(252)
    out["beta"] = beta
    dollar = c * v
    out["illiq"] = np.log1p((r.abs() / dollar.replace(0, np.nan)).rolling(21, min_periods=15).mean() * 1e9)
    out["size"] = np.log(bundle.market_cap)
    prev = c.shift(1)
    tr = np.fmax(np.fmax((h - l).to_numpy(), (h - prev).abs().to_numpy()), (l - prev).abs().to_numpy())
    tr = pd.DataFrame(tr, index=c.index, columns=c.columns)
    out["atr"] = tr.rolling(14, min_periods=10).mean()
    out["vol_d"] = r.rolling(63, min_periods=21).std()
    return out


def price_features(bundle: DataBundle, decision_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    frames = daily_price_frames(bundle)
    return {k: v.loc[decision_dates] for k, v in frames.items()}
