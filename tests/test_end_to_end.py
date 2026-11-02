"""Small end-to-end run: research (quick) -> report -> recommendations -> paper ledger."""
import json
from pathlib import Path

import pandas as pd
import pytest

from stocksel.config import load_config
from stocksel.data import synthetic

COLS = ["ticker", "calibrated_prob", "prob_after_haircut", "expected_excess_after_haircut", "position_pct",
        "entry_date", "planned_exit", "stop", "top3_signals_shap", "confidence"]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    art = tmp_path_factory.mktemp("e2e")
    cfg = load_config("config.yaml", {
        "paths.artifacts": str(art), "live.ledger_dir": str(art / "paper"), "report.output_dir": str(art / "out"),
        "data.start": "2002-01-01", "data.end": "2012-12-31", "models.n_jobs": 1,
        "models.candidates": ["logit", "ridge"], "trials.top_n": [30], "trials.buffer": [True],
        "trials.feature_sets": ["core_comp"], "portfolio.top_n": 30,
        "validation.walk_forward.min_train_months": 36, "validation.walk_forward.train_window_months": 60,
        "universe.min_market_cap": 1.0e9, "universe.min_adv_usd": 5.0e6,
    })
    b = synthetic.generate(start="2002-01-01", end="2012-12-31", n_initial=110, ipo_per_year=5, seed=21)
    from stocksel.pipeline import run_research
    res = run_research(cfg, bundle=b, quick=True)
    return cfg, b, res


def test_research_outputs(e2e):
    cfg, b, res = e2e
    m = res["final_metrics"]
    for k in ("cagr", "excess_ann", "sharpe", "max_drawdown", "calmar", "sortino", "information_ratio", "hit_rate_periods"):
        assert k in m
    assert res["dsr"]["n_trials"] >= 3 and "pbo" in res["pbo"]
    assert res["monthly"].index.max() < res["hstart"]  # research never touches the holdout
    spec = json.loads((Path(cfg.paths.artifacts) / "final_spec.json").read_text())
    assert spec["members"][0] == "logit"


def test_report_renders(e2e, tmp_path):
    from stocksel.reports.html import write_report
    _, _, res = e2e
    p = write_report(res, tmp_path / "r.html")
    txt = p.read_text()
    assert "Verdict and red flags" in txt and "DSR" in txt and "PBO" in txt
    assert "Hit rate" in txt and "Sharpe" in txt  # hit rate only appears inside the full table


def test_recommendations_and_paper(e2e):
    cfg, b, res = e2e
    from stocksel.live.paper import PaperLedger
    from stocksel.live.recommend import recommend
    asof = "2011-06-30"
    recs, meta = recommend(cfg, asof=asof, bundle=b)
    assert list(recs.columns[:8]) == COLS[:8] and set(COLS) <= set(recs.columns)
    assert (recs["position_pct"] <= cfg.portfolio.max_position * 100 + 1e-6).all()
    assert meta["gross_exposure"] <= 1.0 + 1e-9
    assert all(c.startswith("Low") for c in recs["confidence"])  # holdout not passed => capped at Low
    led = PaperLedger(cfg)
    led.submit(recs, meta["asof"])
    led.update(b.truncate("2011-08-31"))
    st = led.status()
    assert 1.5 < st["months"] < 3 and st["n_positions"] > 0
    assert st["live_ready"] is False  # fewer than live.paper_min_months of paper trading
