"""Provider capability flags must reflect the loaded dataset, not marketing."""
from stocksel.data.providers.polygon import PolygonProvider
from stocksel.data.providers.sharadar import SharadarProvider
from stocksel.validation.readiness import system_validated


def test_polygon_price_adapter_cannot_certify_real_data_readiness(cfg):
    meta = PolygonProvider(cfg).meta()
    spec = {"config_hash": cfg.hash(), "dsr": .99, "pbo": .01,
            "holdout_passed": True, "validation_protocol": "real_pit_holdout_v1",
            "research_gate": {"full_validation": True, "sanity_passed": True, "blocking_flags": []}}
    assert meta["survivorship_free"] is False
    assert meta["pit_fundamentals"] is False
    assert not system_validated(spec, cfg, meta)


def test_sharadar_tickers_table_filter_does_not_collide_with_endpoint(cfg, monkeypatch):
    provider = SharadarProvider(cfg)
    monkeypatch.setattr(provider, "api_key", lambda: "test-only-key")
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"datatable": {"columns": [{"name": "ticker"}], "data": [["TEST"]]}, "meta": {}}

    def get(url, params, timeout):
        calls.append((url, dict(params)))
        return Response()

    monkeypatch.setattr("stocksel.data.providers.sharadar.requests.get", get)
    frame = provider._table("TICKERS", table="SEP")
    assert calls[0][0].endswith("/SHARADAR/TICKERS.json")
    assert calls[0][1]["table"] == "SEP"
    assert list(frame.ticker) == ["TEST"]
