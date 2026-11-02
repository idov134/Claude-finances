"""Data layer: canonical bundle, providers, synthetic market, universe, point-in-time joins."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from .bundle import DataBundle  # noqa: F401


def load_bundle(cfg, use_cache: bool = True) -> DataBundle:
    """Load data from the configured provider, with a local pickle cache keyed by the data config."""
    key = hashlib.sha256(json.dumps(cfg.to_dict()["data"], sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache = Path(cfg.data.cache_dir) / f"bundle_{cfg.data.provider}_{key}.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    p = cfg.data.provider
    if p == "synthetic":
        from .synthetic import from_config
        bundle = from_config(cfg)
    else:
        from .providers.csv_provider import CSVProvider
        from .providers.polygon import PolygonProvider
        from .providers.sharadar import SharadarProvider
        from .providers.tiingo import TiingoProvider
        from .providers.yfinance_provider import YFinanceProvider
        cls = {"csv": CSVProvider, "yfinance": YFinanceProvider, "sharadar": SharadarProvider,
               "polygon": PolygonProvider, "tiingo": TiingoProvider}[p]
        bundle = cls(cfg).load()
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle
