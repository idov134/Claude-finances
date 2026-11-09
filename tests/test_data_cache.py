"""Changing benchmark/time interpretation must not reuse another dataset."""
from stocksel.data import load_bundle


def test_cached_bundle_tracks_benchmark_and_timezone(cfg, tmp_path, monkeypatch):
    calls = []

    def provider(config):
        identity = (config.market.benchmark, config.market.timezone)
        calls.append(identity)
        return {"provider_identity": identity}

    monkeypatch.setattr("stocksel.data.synthetic.from_config", provider)
    original = cfg.copy_with({"data.cache_dir": str(tmp_path)})
    changed_benchmark = original.copy_with({"market.benchmark": "QQQ"})
    changed_clock = original.copy_with({"market.timezone": "Asia/Jerusalem"})
    first = load_bundle(original)
    assert load_bundle(original) == first
    assert len(calls) == 1
    assert load_bundle(changed_benchmark)["provider_identity"][0] == "QQQ"
    assert load_bundle(changed_clock)["provider_identity"][1] == "Asia/Jerusalem"
    assert len(calls) == 3
    assert load_bundle(original) == first
    assert len(calls) == 3


def test_csv_cache_invalidated_when_files_change(tmp_path, monkeypatch):
    import os
    import pandas as pd
    import stocksel.data as data_mod
    from stocksel.config import load_config
    d = tmp_path / "csv"
    d.mkdir()
    (d / "prices.csv").write_text("date,ticker,open,high,low,close,volume,market_cap\n")
    cfg = load_config("config.yaml", {"data.provider": "csv", "data.csv_dir": str(d),
                                      "data.cache_dir": str(tmp_path / "cache")})
    calls = []

    class Fake:
        def __init__(self, cfg):
            pass

        def load(self):
            calls.append(1)
            return {"n": len(calls)}
    monkeypatch.setattr("stocksel.data.providers.csv_provider.CSVProvider", Fake)
    assert data_mod.load_bundle(cfg) == {"n": 1}
    assert data_mod.load_bundle(cfg) == {"n": 1}            # unchanged files -> cache hit
    (d / "prices.csv").write_text("date,ticker,open,high,low,close,volume,market_cap\n2020-01-02,A,1,1,1,1,1,1\n")
    assert data_mod.load_bundle(cfg) == {"n": 2}            # rebuilt file -> fresh load
