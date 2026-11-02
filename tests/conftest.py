import warnings

import numpy as np
import pandas as pd
import pytest

from stocksel.config import load_config
from stocksel.data import synthetic

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    art = tmp_path_factory.mktemp("artifacts")
    return load_config("config.yaml", {
        "paths.artifacts": str(art), "live.ledger_dir": str(art / "paper"),
        "data.start": "2002-01-01", "data.end": "2011-12-31", "models.n_jobs": 1,
        "validation.walk_forward.min_train_months": 36, "validation.walk_forward.train_window_months": 60,
        "universe.min_market_cap": 1.0e9, "universe.min_adv_usd": 5.0e6,
    })


@pytest.fixture(scope="session")
def bundle():
    return synthetic.generate(start="2002-01-01", end="2011-12-31", n_initial=120, ipo_per_year=5, seed=11)


@pytest.fixture(scope="session")
def strong_bundle():
    """Planted anomalies at 2x strength without post-2005 decay: tests statistical power of the pipeline."""
    return synthetic.generate(start="2002-01-01", end="2011-12-31", n_initial=200, ipo_per_year=8, seed=13,
                              alpha_scale=2.0, post_2005_decay=1.0, with_news=False)


@pytest.fixture(scope="session")
def null_bundle():
    return synthetic.generate(start="2002-01-01", end="2011-12-31", n_initial=200, ipo_per_year=8, seed=12,
                              alpha_scale=0.0, with_news=False)


@pytest.fixture(scope="session")
def panel(bundle, cfg):
    from stocksel.features.pipeline import build_features
    return build_features(bundle, cfg)
