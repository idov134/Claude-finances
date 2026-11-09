"""Feature pipeline: raw signals at decision dates -> cross-sectionally normalized long panel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..data.bundle import DataBundle
from ..data.universe import eligible
from .fundamental import fundamental_features
from .news import daily_sentiment, make_scorer, monthly_news_feature, news_weight
from .normalize import cross_sectional
from .price import price_features
from .regime import regime_features, regime_frame
from .residual import residual_momentum
from .registry import TIME_SERIES_FEATURES


@dataclass
class FeaturePanel:
    X: pd.DataFrame                      # MultiIndex (date, ticker) -> normalized features
    raw: dict                            # name -> wide raw frame (decision dates x tickers)
    universe: pd.DataFrame               # wide bool
    dates: pd.DatetimeIndex
    aux: dict                            # atr, close, mcap, adv, vol_d, idio_vol, beta (wide)
    regime: pd.DataFrame                 # trend / rvol / vix at decision dates
    news_info: dict = field(default_factory=dict)
    news_daily: Optional[pd.DataFrame] = None
    news_weight: Optional[pd.Series] = None

    def features(self, names) -> pd.DataFrame:
        return self.X[[n for n in names if n in self.X.columns]]


def to_long(frames: dict, universe: pd.DataFrame) -> pd.DataFrame:
    rows, cols = np.nonzero(universe.to_numpy())
    idx = pd.MultiIndex.from_arrays([universe.index[rows], universe.columns[cols]], names=["date", "ticker"])
    data = {k: v.reindex_like(universe).to_numpy()[rows, cols] for k, v in frames.items()}
    return pd.DataFrame(data, index=idx)


def build_features(bundle: DataBundle, cfg, scorer=None, dates: Optional[pd.DatetimeIndex] = None) -> FeaturePanel:
    dates = bundle.month_end_dates() if dates is None else pd.DatetimeIndex(dates)
    universe = eligible(bundle, cfg, dates)
    raw: dict[str, pd.DataFrame] = {}
    pf = price_features(bundle, dates)
    for k in ("mom_12_1", "mom_6_1", "rev_1m", "vol_252", "idio_vol", "illiq", "size"):
        raw[k] = pf[k]
    raw.update(residual_momentum(bundle, dates))
    raw.update(fundamental_features(bundle, cfg, dates))
    reg = regime_features(bundle, dates)
    raw.update(reg)

    scorer = scorer or make_scorer(cfg)
    ssum, scnt, info = daily_sentiment(bundle, scorer, cfg)
    news_daily, w = None, None
    if info.get("n_articles", 0) and scnt.to_numpy().sum() > 0:
        from ..backtest.daily import daily_news_strategy
        news_daily = daily_news_strategy(bundle, ssum, scnt, cfg)
        w = news_weight(news_daily["net"], cfg.news.rolling_sharpe_days, cfg.news.target_sharpe, cfg.news.min_obs)
        nf = monthly_news_feature(ssum, scnt, dates, cfg.news.lookback_days)
        raw["news_sent"] = nf.mul(w.reindex(dates).fillna(0.0), axis=0)
    else:
        raw["news_sent"] = pd.DataFrame(np.nan, index=dates, columns=bundle.tickers)

    disabled = set(cfg.features.get("disabled", []) or [])
    neutral = cfg.features.get("neutralize", "none") or "none"
    if neutral not in ("none", "sector"):
        raise ValueError("features.neutralize must be 'none' or 'sector'")
    groups = bundle.sectors if neutral == "sector" else None
    norm = {}
    for name, df in raw.items():
        if name in disabled:
            continue
        if name in TIME_SERIES_FEATURES:
            norm[name] = df.where(universe)
        elif name == "size":   # size is a risk control, not a peer-relative signal
            norm[name] = cross_sectional(df, universe, cfg.features.normalize, cfg.features.winsor_pct)
        else:
            norm[name] = cross_sectional(df, universe, cfg.features.normalize, cfg.features.winsor_pct, groups=groups)
    X = to_long(norm, universe).fillna(0.0)

    aux = {"atr": pf["atr"], "close": bundle.close.loc[dates], "mcap": bundle.market_cap.loc[dates],
           "adv": bundle.adv_usd(63).loc[dates], "vol_d": pf["vol_d"], "idio_vol": pf["idio_vol"],
           "beta": pf["beta"]}
    return FeaturePanel(X=X, raw=raw, universe=universe, dates=dates, aux=aux,
                        regime=regime_frame(bundle, dates), news_info=info, news_daily=news_daily, news_weight=w)
