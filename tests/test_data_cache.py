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
