"""The pipeline must detect a planted signal, find nothing in a null world, and fail on shuffled labels."""
import numpy as np
import pandas as pd

from stocksel.features.pipeline import build_features
from stocksel.models.labels import forward_returns, label_panel
from stocksel.models.train import model_view, walk_forward
from stocksel.validation.metrics import classification_metrics
from stocksel.validation.sanity import shuffle_within_date

FEATS = ["mom_12_1", "mom_6_1", "rev_1m", "vol_252", "idio_vol", "quality_gpa", "quality_roe", "sue", "pead", "revisions"]


def _oos(bundle, cfg, shuffle=False):
    p = build_features(bundle, cfg)
    lab = forward_returns(bundle, p.dates, 1)
    L = label_panel(lab, p.X.index)
    if shuffle:
        Ls = shuffle_within_date(L)
        wf = walk_forward(p.features(FEATS), Ls, ["logit"], cfg, keep_models=False)
    else:
        wf = walk_forward(p.features(FEATS), L, ["logit"], cfg, keep_models=False)
    v = model_view(wf.preds, "logit")
    Lt = L.reindex(v.index)
    return classification_metrics(v["prob"], Lt["y"], Lt["excess"], v.index.get_level_values(0))


def test_planted_signal_is_detected(strong_bundle, cfg):
    m = _oos(strong_bundle, cfg)
    assert m["ic_mean"] > 0 and m["ic_tstat"] > 2.0
    assert m["accuracy"] < 0.65  # anything higher would be a red flag for leakage


def test_null_world_has_no_edge(null_bundle, cfg):
    m = _oos(null_bundle, cfg)
    assert abs(m["ic_tstat"]) < 2.5
    assert abs(m["auc"] - 0.5) < 0.02


def test_shuffled_labels_fail(strong_bundle, cfg):
    m = _oos(strong_bundle, cfg, shuffle=True)
    assert abs(m["auc"] - 0.5) < 0.02
    assert abs(m["ic_tstat"]) < 2.5
