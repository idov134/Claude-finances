"""Purged + embargoed K-fold diagnostics: out-of-fold skill of each candidate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.train import fit_calibrated
from .metrics import classification_metrics
from .splits import PurgedKFold


def purged_kfold_scores(X: pd.DataFrame, L: pd.DataFrame, names: list[str], cfg) -> pd.DataFrame:
    pk = cfg.validation.purged_kfold
    cv = PurgedKFold(int(pk.n_splits), float(pk.embargo_pct))
    lab = L["y"].notna().to_numpy()
    Xl, Ll = X[lab], L[lab]
    d = Xl.index.get_level_values(0)
    rows = {}
    for n in names:
        oof = np.full(len(Xl), np.nan)
        for tr, te in cv.split(pd.Series(d), Ll["t1"]):
            if len(tr) < 500:
                continue
            fm = fit_calibrated(n, Xl.iloc[tr], Ll.iloc[tr], cfg, int(cfg.models.seed))
            oof[te] = fm.predict(Xl.iloc[te].to_numpy())["prob"]
        m = classification_metrics(oof, Ll["y"].to_numpy(), Ll["excess"].to_numpy(), d)
        rows[n] = {k: m.get(k) for k in ("auc", "accuracy", "log_loss", "brier", "ic_mean", "ic_tstat")}
    return pd.DataFrame(rows).T
