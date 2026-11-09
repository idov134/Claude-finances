"""Performance metrics, Sharpe inference (PSR / DSR), block bootstrap.

Conventions: monthly return series (periods_per_year=12) unless stated. Sharpe
is computed on returns in excess of the risk-free rate when ``rf`` is given.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _clean(r) -> np.ndarray:
    r = np.asarray(r, float)
    return r[np.isfinite(r)]


def cagr(r, ppy=12):
    r = _clean(r)
    if len(r) == 0:
        return np.nan
    g = np.prod(1 + r)
    return g ** (ppy / len(r)) - 1 if g > 0 else -1.0


def ann_vol(r, ppy=12):
    return float(np.std(_clean(r), ddof=1) * np.sqrt(ppy))


def sharpe(r, rf=None, ppy=12, annualize=True):
    r = np.asarray(r, float)
    if rf is not None:
        r = r - np.asarray(rf, float)
    r = _clean(r)
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return np.nan
    sr = np.mean(r) / np.std(r, ddof=1)
    return float(sr * np.sqrt(ppy)) if annualize else float(sr)


def sortino(r, rf=None, ppy=12):
    r = np.asarray(r, float)
    if rf is not None:
        r = r - np.asarray(rf, float)
    r = _clean(r)
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return float(np.mean(r) / dd * np.sqrt(ppy)) if dd > 0 else np.nan


def drawdown_series(r) -> pd.Series:
    r = pd.Series(r).fillna(0.0)
    eq = (1 + r).cumprod()
    # Initial account equity is 1, even when the first measured period loses.
    return eq / eq.cummax().clip(lower=1.0) - 1


def max_drawdown(r):
    return float(drawdown_series(r).min())


def calmar(r, ppy=12):
    mdd = max_drawdown(r)
    return float(cagr(r, ppy) / abs(mdd)) if mdd < 0 else np.nan


def information_ratio(r, b, ppy=12):
    a = np.asarray(r, float) - np.asarray(b, float)
    return sharpe(a, ppy=ppy)


def tracking_error(r, b, ppy=12):
    return ann_vol(np.asarray(r, float) - np.asarray(b, float), ppy)


def capm(r, b, rf=None):
    r, b = np.asarray(r, float), np.asarray(b, float)
    if rf is not None:
        r, b = r - rf, b - rf
    m = np.isfinite(r) & np.isfinite(b)
    if m.sum() < 6:
        return np.nan, np.nan
    beta = np.cov(r[m], b[m])[0, 1] / np.var(b[m], ddof=1)
    alpha = np.mean(r[m]) - beta * np.mean(b[m])
    return float(alpha * 12), float(beta)


def sharpe_se(sr_period: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Std. error of a per-period Sharpe ratio (Lo 2002 / Mertens 2002, non-normal)."""
    return math.sqrt(max(1e-12, (1 - skew * sr_period + (kurt - 1) / 4 * sr_period ** 2) / max(n - 1, 1)))


def psr(r, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true per-period SR > sr_benchmark)."""
    r = _clean(r)
    n = len(r)
    if n < 6 or np.std(r, ddof=1) == 0:
        return np.nan
    sr = np.mean(r) / np.std(r, ddof=1)
    sk = stats.skew(r)
    ku = stats.kurtosis(r, fisher=False)
    return float(stats.norm.cdf((sr - sr_benchmark) / sharpe_se(sr, n, sk, ku)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] of n_trials unskilled strategies with cross-trial SR variance var_sr (per-period units)."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(r, n_trials: int, trial_sharpes_period: np.ndarray | None = None,
                    var_sr: float | None = None) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    ``trial_sharpes_period``: per-period (non-annualized) Sharpe of every trial in
    the registry, used for the cross-trial variance.
    """
    if var_sr is None:
        ts = _clean(trial_sharpes_period) if trial_sharpes_period is not None else np.array([])
        var_sr = float(np.var(ts, ddof=1)) if len(ts) > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return {"dsr": psr(r, sr0), "sr0_period": sr0, "sr0_annual": sr0 * math.sqrt(12),
            "n_trials": int(n_trials), "var_sr": var_sr}


def alpha_returns(r_ex, b_ex) -> pd.Series:
    """Beta-adjusted (CAPM alpha + residual) returns: r_ex - beta * b_ex, beta estimated on the overlap.

    Both inputs are returns in excess of the risk-free rate. The Sharpe ratio of the result is the
    appraisal ratio: reward per unit of risk the strategy took *beyond* its market exposure. For a
    long-only strategy this separates stock-selection skill from simply holding equities.
    """
    r = pd.Series(r_ex, dtype=float)
    b = pd.Series(b_ex, dtype=float).reindex(r.index)
    m = r.notna() & b.notna()
    if m.sum() < 6 or b[m].var() == 0:
        return r * np.nan
    beta = np.cov(r[m], b[m])[0, 1] / np.var(b[m], ddof=1)
    return r - beta * b


def effective_trials(returns: pd.DataFrame) -> float:
    """Effective number of independent trials from the eigenvalues of the trial correlation matrix."""
    R = returns.dropna(axis=1, how="all")
    R = R.loc[:, R.std() > 0]
    c = R.corr().fillna(0.0).to_numpy()
    if c.size == 0:
        return 0.0
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if (ev ** 2).sum() > 0 else 1.0


def stationary_bootstrap_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    p = 1.0 / block
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def bootstrap_ci(r, stat=np.mean, n: int = 2000, block: int = 6, alpha: float = 0.05, seed: int = 0):
    r = _clean(r)
    if len(r) < 6:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(r[stationary_bootstrap_idx(len(r), block, rng)]) for _ in range(n)])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)), float(np.mean(vals > 0))


def performance_table(net: pd.Series, bench: pd.Series, rf: pd.Series | None = None, ppy: int = 12,
                      boot: dict | None = None) -> dict:
    """All success metrics together (hit rate never alone)."""
    df = pd.concat({"r": net, "b": bench}, axis=1).dropna()
    r, b = df["r"], df["b"]
    rfa = rf.reindex(df.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=df.index)
    ex = r - b
    alpha, beta = capm(r, b, rfa)
    boot = boot or {}
    lo, hi, p_pos = bootstrap_ci(ex.to_numpy(), n=boot.get("n", 1000), block=boot.get("block", 6),
                                 seed=boot.get("seed", 0))
    sr_lo, sr_hi, _ = bootstrap_ci((r - rfa).to_numpy(), stat=lambda x: np.mean(x) / np.std(x, ddof=1) * np.sqrt(ppy),
                                   n=boot.get("n", 1000), block=boot.get("block", 6), seed=boot.get("seed", 0))
    wins, losses = r[r > 0], r[r < 0]
    return {
        "n_periods": int(len(r)),
        "cagr": cagr(r, ppy), "bench_cagr": cagr(b, ppy),
        "ann_vol": ann_vol(r, ppy), "bench_vol": ann_vol(b, ppy),
        "sharpe": sharpe(r, rfa, ppy), "sharpe_ci_low": sr_lo, "sharpe_ci_high": sr_hi,
        "bench_sharpe": sharpe(b, rfa, ppy),
        "psr_0": psr((r - rfa).to_numpy()),
        "sortino": sortino(r, rfa, ppy), "max_drawdown": max_drawdown(r), "bench_max_drawdown": max_drawdown(b),
        "calmar": calmar(r, ppy), "bench_calmar": calmar(b, ppy),
        "excess_ann": float(ex.mean() * ppy), "excess_ci_low": lo * ppy, "excess_ci_high": hi * ppy,
        "p_excess_positive": p_pos,
        "tracking_error": tracking_error(r, b, ppy), "information_ratio": information_ratio(r, b, ppy),
        "capm_alpha": alpha, "capm_beta": beta,
        "hit_rate_periods": float((ex > 0).mean()),
        "avg_win_loss_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
        "expectancy_period": float(r.mean()),
    }


def classification_metrics(prob, y, excess=None, dates=None) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    p = np.asarray(prob, float)
    yy = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(yy)
    p, yy = p[m], yy[m]
    out = {"n": int(m.sum()), "accuracy": float(((p > 0.5) == (yy > 0.5)).mean()) if m.sum() else np.nan}
    if m.sum() > 10 and len(np.unique(yy)) == 2:
        out["auc"] = float(roc_auc_score(yy, p))
        out["log_loss"] = float(log_loss(yy, np.clip(p, 1e-6, 1 - 1e-6)))
        out["brier"] = float(brier_score_loss(yy, p))
        out["base_rate"] = float(yy.mean())
    if excess is not None and dates is not None:
        df = pd.DataFrame({"p": np.asarray(prob)[m], "e": np.asarray(excess)[m], "d": np.asarray(dates)[m]})
        ic = df.groupby("d").apply(lambda g: g["p"].corr(g["e"], method="spearman") if len(g) > 5 else np.nan,
                                   include_groups=False).dropna()
        out["ic_mean"] = float(ic.mean()) if len(ic) else np.nan
        out["ic_tstat"] = float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else np.nan
        # out-of-sample R^2 of a probability-implied ranking is not defined; report monthly cross-sectional R^2
        out["xs_r2"] = float(np.nanmean(ic.to_numpy() ** 2)) if len(ic) else np.nan
    return out
