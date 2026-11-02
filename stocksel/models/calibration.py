"""Probability calibration (Platt / isotonic) and the probability -> expected-return map."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """Maps a raw logit-like score to a calibrated probability.

    ``slope`` (d calibrated-logit / d raw score) is always estimated with a Platt
    fit so that attributions can be expressed in calibrated-logit units even when
    the probabilities come from isotonic regression.
    """

    def __init__(self, method: str = "platt"):
        self.method = method
        self.platt = LogisticRegression(C=1e6, max_iter=1000)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98) if method == "isotonic" else None
        self.mu, self.sd = 0.0, 1.0

    def fit(self, score, y):
        s = np.asarray(score, float)
        self.mu, self.sd = float(np.mean(s)), float(np.std(s) or 1.0)
        z = ((s - self.mu) / self.sd).reshape(-1, 1)
        self.platt.fit(z, y)
        if self.iso is not None:
            self.iso.fit(s, y)
        return self

    @property
    def slope(self) -> float:
        return float(self.platt.coef_[0, 0] / self.sd)

    def transform(self, score):
        s = np.asarray(score, float)
        if self.iso is not None:
            return self.iso.predict(s)
        return self.platt.predict_proba(((s - self.mu) / self.sd).reshape(-1, 1))[:, 1]


class ExpectedReturnMap:
    """Monotone map from calibrated probability to expected excess return (fit on calibration data)."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)

    def fit(self, prob, excess, clip_q: float = 0.01):
        ex = np.asarray(excess, float)
        lo, hi = np.nanquantile(ex, [clip_q, 1 - clip_q])
        self.iso.fit(np.asarray(prob, float), np.clip(ex, lo, hi))
        return self

    def predict(self, prob):
        return self.iso.predict(np.asarray(prob, float))
