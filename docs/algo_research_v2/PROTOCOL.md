# Algorithm research v2 — pre-registration

Registered **9 November 2026**, committed to git **before any run on the research data**
(the commit that adds this file also adds the runner `stocksel/experiments/algo_v2.py`; results are
committed afterwards). Author: Claude. Synthetic data only: this protocol can show that a change
extracts a planted signal better and generalises across fresh draws of the same simulator. It
cannot show real-market value.

## Question

Which of five literature-motivated single changes improve the **out-of-sample, net-of-cost** result
of the default strategy, and does the combination of the individually successful ones generalise
to **fresh, independent market realisations** it was never developed on?

Success is ranked as in the original brief: net return, then DSR, then drawdown/Calmar/Sortino,
then excess return and IR. Hit rate is reported as a secondary column only.

## Data and engine

- **Development world:** the default synthetic market (seed 7), physically truncated at
  **2023-12-29**. The 2024–2025 holdout was opened on 2 Nov 2026; it is used and is never touched.
- **Confirmation worlds:** three fresh synthetic markets, **seeds 101, 202, 303**, never generated
  before in this project. They use the same simulator, config and truncation. They are run **once**,
  only after the development decision has been committed. The runner refuses a second run unless
  forced, and a forced run is recorded.
- **Engine:** the corrected monthly engine. Costs are paid from cash, an order without an executable
  open stays unfilled (cash), a held name without a return is an error, and weights drift on net
  equity. Configured portfolio: top 50 with buffer, capped value weights, quarter Kelly after the
  50% haircut, 10% vol target, 5% name / 30% sector caps, 1× costs.

## Configurations

**B0 (baseline)** is the effective default: `core_comp` features, rank normalisation, binary
target (excess > 0), and an online-gated ensemble of {logit, ridge, enet}. In the original run the
gate rejected LightGBM and the MLP in all 21 windows, so they are left out here to save compute.
E6 tests a tuned tree model through the same gate.

| ID | Single change vs B0 | Why (prior evidence, not this data) |
|---|---|---|
| E1 | `features.neutralize: sector`: every signal except size ranked within its sector | Industry-relative signals strip sector bets that are mostly noise (Asness, Porter & Stevens 2000) |
| E2 | `models.feature_set: core_comp_resid`: residual momentum replaces 12-1 / 6-1 total-return momentum | Blitz, Huij & Martens (2011); Gutierrez & Pirinsky (2007): higher Sharpe, smaller crashes |
| E3 | `labels.target: tails`, `tail_frac: 0.3`: classifiers learn the top vs bottom 30% only; calibration still uses all rows and the full binary event | The middle of the return distribution is mostly noise around zero |
| E4 | `portfolio.score_smoothing: 0.5`: each stock's prediction is exponentially smoothed across months | Lower turnover and costs for the same signal (Novy-Marx & Velikov 2016; Gârleanu & Pedersen 2013) |
| E5 | `portfolio.mandate: relative`: fully invested, no vol target or Kelly caps | Isolates the market-exposure (beta) drag in "beat buy-and-hold". **Mandate alternative, not adoptable**: its drawdowns resemble the index's (−47% in 2008), which breaks the configured 20% tolerance. Reported for the user's decision |
| E6 | Candidates {logit, ridge, enet, `lgbm_tuned`} through the existing online gate. The tuned LightGBM picks num_leaves ∈ {7, 15, 31} × min_child_samples ∈ {200, 800} by inner purged 3-fold log-loss on each training window (nested; never on test months) | Did poor hyper-parameters cause the earlier gate rejection? |

No parameter is searched: tail fraction 0.3, smoothing 0.5 and the grid are fixed above.

**Diagnostic D1 (not a candidate).** The default simulator is additive-linear, which structurally
disadvantages nonlinear models. D1 runs B0 and E6 on a world with one planted interaction (seed 7,
`interaction: 1.0`: momentum works mainly in low-volatility stocks). It reports whether the gate
admits the tuned tree model when real nonlinearity exists. It checks the machinery's power and
plays no part in adoption.

## Development pass rule (E1, E2, E3, E4, E6 each vs B0, same months)

1. Annualised mean paired net-return difference > 0 **and** stationary-bootstrap
   P(difference > 0) ≥ **0.90** (2,000 draws, expected block 6 months, seed 3).
2. Sharpe ≥ B0's Sharpe.
3. Month-end max drawdown not worse than B0's by more than **2 percentage points**.
4. No red flag: OOS directional accuracy > 65% or Sharpe > 2.

**C1** applies all passing changes together, with no further tuning. It is reported on the
development world. If nothing passes, there is no C1 and B0 stays the default.

## Confirmation and adoption rule (run once)

On each confirmation world, run B0, C1 and E5. **C1 becomes the new default only if:**

- the pooled P(C1 − B0 > 0) ≥ **0.90**. Each draw resamples every world's paired-difference series
  independently (stationary bootstrap, same settings), and the statistic is the mean of the three
  worlds' means;
- the mean difference is > 0 in **at least 2 of 3** worlds;
- no red flag in any world.

Otherwise the defaults stay unchanged and the candidates remain available as config options. E5
results on the confirmation worlds are reported for the user's mandate decision.

## Multiple testing and records

Every planned, completed and failed configuration goes to `registry.jsonl` in this folder.
Development-world returns (net − rf, monthly, decision-date index) go to `returns.csv` and join the
full history: the original 48 trials (`docs/research_history`), Codex's re-entry (2) and comparison
(33) experiments, and this protocol. DSR is charged with the number of distinct trial IDs in that
whole history, failed and planned ones included. PBO is computed on the full decision-indexed
history matrix. Confirmation and D1 trials count toward N; their returns are stored separately
because they are different markets.

## Known limitation stated in advance

The same author designed the simulator and these experiments. In the simulator the planted
momentum is idiosyncratic and sector returns are unpredictable noise, so E1 and E2 gains are
**partly by construction**. Fresh seeds test generalisation across draws of that simulator, not
across reality. Real point-in-time data and a new untouched window remain the only route to a
financial claim.

Reproduce: `python -m stocksel.experiments.algo_v2 dev`, commit, then
`python -m stocksel.experiments.algo_v2 confirm`.
