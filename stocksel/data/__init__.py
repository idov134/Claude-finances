"""Data layer: canonical bundle, providers, synthetic market, universe, point-in-time joins."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from .bundle import DataBundle  # noqa: F401


def load_bundle(cfg, use_cache: bool = True) -> DataBundle:
    """Cache canonical data by provider inputs, market identity and schema version.

    Market settings select the benchmark and interpretation of timestamps. A
    cache from another benchmark/timezone must never silently satisfy a load.
    Versioning also avoids reusing the former incomplete cache identity.
    """
    config = cfg.to_dict()
    identity = {"schema_version": 2, "data": config["data"], "market": config["market"]}
    if cfg.data.provider == "csv":
        # Local files can be rebuilt in place: key the cache on their size and modification time too,
        # so corrected data is never served from a stale bundle.
        folder = Path(cfg.data.csv_dir)
        identity["files"] = {f.name: [f.stat().st_size, f.stat().st_mtime_ns]
                             for f in sorted(folder.glob("*.csv"))} if folder.exists() else {}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
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
