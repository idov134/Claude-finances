"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (Bailey et al. 2017)."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def pbo(returns: pd.DataFrame, n_blocks: int = 16, max_combinations: int = 5000, seed: int = 0) -> dict:
    """returns: T x N matrix of per-period OOS returns of N trials (columns)."""
    M = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    T, N = M.shape
    if N < 2 or T < n_blocks * 2:
        return {"pbo": np.nan, "n_trials": N, "n_periods": T, "logits": np.array([])}
    blocks = np.array_split(np.arange(T), n_blocks)
    all_combos = comb(n_blocks, n_blocks // 2)
    rng = np.random.default_rng(seed)
    if all_combos > max_combinations:
        combos = {tuple(sorted(rng.choice(n_blocks, n_blocks // 2, replace=False))) for _ in range(max_combinations * 2)}
        combos = list(combos)[:max_combinations]
    else:
        combos = list(combinations(range(n_blocks), n_blocks // 2))
    X = M.to_numpy()
    logits, is_best_oos_rank = [], []
    for c in combos:
        is_idx = np.concatenate([blocks[i] for i in c])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in c])
        sr_is = X[is_idx].mean(0) / (X[is_idx].std(0, ddof=1) + 1e-12)
        sr_oos = X[oos_idx].mean(0) / (X[oos_idx].std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1) + 1
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_best_oos_rank.append(w)
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_trials": int(N), "n_periods": int(T),
            "n_combinations": len(combos), "logits": logits,
            "median_oos_rank_of_is_best": float(np.median(is_best_oos_rank))}
