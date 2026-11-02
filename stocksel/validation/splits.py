"""Time-series cross-validation with purging and embargo (López de Prado, AFML ch. 7 & 12).

Each sample i has a start time (decision date) and an end time ``t1`` (label end).
* Purging removes training samples whose [start, t1] overlaps the test window.
* Embargo removes training samples that start within ``embargo`` after the test window
  (serial correlation leaks information backwards).
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _blocks(udates: np.ndarray, n: int) -> list[np.ndarray]:
    return [b for b in np.array_split(udates, n) if len(b)]


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, start: pd.Series, t1: pd.Series):
        start = pd.to_datetime(pd.Series(start)).reset_index(drop=True)
        t1 = pd.to_datetime(pd.Series(t1)).reset_index(drop=True)
        udates = np.sort(start.unique())
        span = (udates[-1] - udates[0])
        embargo = span * self.embargo_pct
        for block in _blocks(udates, self.n_splits):
            test = start.isin(block).to_numpy()
            t0, tmax = block[0], t1[test].max()
            overlap = (start <= tmax) & (t1 >= t0)
            emb = (start > tmax) & (start <= tmax + embargo)
            train = ~(test | overlap.to_numpy() | emb.to_numpy()) & t1.notna().to_numpy()
            yield np.where(train)[0], np.where(test)[0]


class CombinatorialPurgedCV:
    """CPCV: N groups, k test groups => C(N,k) splits and k/N*C(N,k) backtest paths."""

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01):
        self.N, self.k, self.embargo_pct = n_groups, n_test_groups, embargo_pct

    @property
    def n_paths(self) -> int:
        return comb(self.N, self.k) * self.k // self.N

    def split(self, start: pd.Series, t1: pd.Series):
        start = pd.to_datetime(pd.Series(start)).reset_index(drop=True)
        t1 = pd.to_datetime(pd.Series(t1)).reset_index(drop=True)
        udates = np.sort(start.unique())
        groups = _blocks(udates, self.N)
        embargo = (udates[-1] - udates[0]) * self.embargo_pct
        for test_groups in combinations(range(len(groups)), self.k):
            test = np.zeros(len(start), bool)
            train = t1.notna().to_numpy().copy()
            for g in test_groups:
                in_g = start.isin(groups[g]).to_numpy()
                test |= in_g
                t0, tmax = groups[g][0], t1[in_g].max()
                overlap = ((start <= tmax) & (t1 >= t0)).to_numpy()
                emb = ((start > tmax) & (start <= tmax + embargo)).to_numpy()
                train &= ~(overlap | emb)
            train &= ~test
            yield test_groups, np.where(train)[0], np.where(test)[0]

    def paths(self) -> list[list[tuple[int, tuple]]]:
        """For each path, the (group, split) pairs that compose it."""
        splits = list(combinations(range(self.N), self.k))
        counter = {g: 0 for g in range(self.N)}
        paths = [[None] * self.N for _ in range(self.n_paths)]
        for s in splits:
            for g in s:
                paths[counter[g]][g] = (g, s)
                counter[g] += 1
        return paths


def walk_forward_schedule(udates: pd.DatetimeIndex, min_train: int, window: int | None, step: int):
    """Yields (train_start, retrain_date, test_end_exclusive) over monthly decision dates."""
    udates = pd.DatetimeIndex(sorted(udates))
    out = []
    i = min_train
    while i < len(udates):
        start = udates[max(0, i - window)] if window else udates[0]
        end = udates[i + step] if i + step < len(udates) else udates[-1] + pd.Timedelta(days=1)
        out.append((start, udates[i], end))
        i += step
    return out
