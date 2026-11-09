# Algorithm research v3 — pre-registration (mandate frontier + gate diagnostic)

Registered **9 November 2026**, committed before the run (runner: `stocksel/experiments/algo_v3.py`).
Synthetic data only.

## Why a v3

v2 ([results](../algo_research_v2/RESULTS.md)) found no stock-selection change that generalised to
fresh worlds. The one robust effect came from the **mandate**: the fully invested relative
mandate (E5) beat the benchmark in all four worlds, with confidence intervals above zero, but with
index-like drawdowns of −35% to −44%. v3 asks one question about that lever. It adds no new signals
and searches no parameters.

## Candidate

**M1:** `portfolio.mandate: relative` + `portfolio.regime_filter.enabled: true`, with the regime
filter's existing defaults, unchanged since the first version of the config. When the benchmark is
below its 210-day average **and** the VIX proxy is above 30, exposure is halved. The trend filter
follows Faber (2007).

References in every world: **B0** (the default, absolute mandate) and **E5** (relative mandate
without the filter).

## Worlds

- Development world: seed 7, truncated at 2023-12-29 (the used 2024–2025 holdout is never touched).
- Fresh confirmation worlds: **seeds 404, 505, 606**, never generated before (v2 used 101/202/303).
- Engine, portfolio parameters and costs as in v2. One code change since v2: the elastic-net
  solver's random seed is now fixed (reproducibility, effect ~0.01%/yr).

## Adoption rule (M1 becomes the default only if all hold)

1. Pooled net excess return over the benchmark across the three confirmation worlds is > 0, with
   P ≥ 0.90 (stationary bootstrap per world, 2,000 draws, block 6, seed 3; statistic = mean of the
   worlds' means).
2. Month-end max drawdown within the configured tolerance (`capital.max_drawdown`, 20%) in
   **every** world, development included.
3. No red flag (accuracy > 65% or Sharpe > 2).

If rule 2 fails, M1 is reported as a point on the risk/return frontier between B0 and E5. The user
chooses a mandate; the code does not change the default.

## Diagnostic D1b (not a candidate)

The v2 gate never admitted the tuned LightGBM, even in the world with a planted interaction
(seed 7, `interaction: 1.0`). D1b runs standalone logit and standalone `lgbm_tuned` there through
the same walk-forward and reports:
- OOS AUC, IC and log-loss of each;
- net performance of each one's own portfolio;
- the gate's bootstrap P(lgbm_tuned beats logit) at every retrain date.

This separates "the tree model learned nothing extra" from "the gate is too strict to notice".

All runs are logged in this folder's `registry.jsonl`. Development-world returns join the full
trial history used for DSR/PBO. The run is executed once:
`python -m stocksel.experiments.algo_v3`.
