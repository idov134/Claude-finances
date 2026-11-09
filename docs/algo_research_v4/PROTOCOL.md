# Algorithm research v4 — pre-registration: the cash-collapse fix

Registered **9 November 2026**, committed before the run (runner: `stocksel/experiments/algo_v4.py`).

## Where the question comes from

The real-data first look ([v1](../real_firstlook/PROTOCOL.md)) showed the default absolute
mandate sitting **100% in cash for 21 of 23 holdout months**, and through 2021–22. A stock is held
only if its haircut probability of beating the index exceeds 0.5. The calibrated probabilities
inherit the recent cross-sectional base rate. In a narrow, mega-cap-led market most stocks lag the
index, so every probability falls below 0.5. Under the absolute mandate the alternative to a stock
is cash, not the index, so this is an unintended market-timing rule based on breadth.

**The real data that revealed the flaw cannot test the fix.** v4 therefore tests it on synthetic
markets it has not seen. The real research window is reported once at the end, labelled post hoc
and non-decisive. The real holdout is not used.

## Candidate

**E7:** `portfolio.prob_centering: cross_section`. On each decision date, before the haircut, the
Kelly caps and the `min_prob` gate, the calibrated probabilities are shifted so that their
cross-sectional mean is 0.5. Ranking and selection order are unchanged; only the market-wide base
rate is removed. Baseline **B0** is the v2/v3 default (`prob_centering: none`).

## Worlds (all fresh, never generated before)

- **Standard worlds** (no-harm test): seeds **1101, 1202, 1303**. The development world (seed 7)
  is reported alongside, not used in the rule.
- **Breadth worlds** (mechanism test): seeds **2404, 2505, 2606** with `breadth: 2.0`. Persistent
  mega-cap regimes push the monthly share of stocks beating the index to about 0.39–0.53 by year,
  similar to real markets. The knob was set by looking only at those base rates, on seed 2101, which
  is excluded from the test. No strategy results were viewed.
- Truncated at 2023-12-29 as before; engine, costs and portfolio defaults as in v2/v3.

## Adoption rule (E7 becomes the default only if all hold)

1. **No harm:** on the standard worlds, the 95% CI lower bound of the pooled E7 − B0 net
   difference is above **−1.0%/yr** (stationary bootstrap per world, 2,000 draws, block 6, seed 3).
2. **Mechanism:** on the breadth worlds, pooled P(E7 − B0 > 0) ≥ **0.90**, and the mean difference
   is positive in at least **2 of 3** worlds.
3. Max drawdown no worse than B0's by more than **3 percentage points** in any world. More
   exposure is the point, so the margin is wider than v2's 2 points.
4. No red flag (accuracy > 65% or Sharpe > 2) in any world.

Also reported: months spent essentially in cash (exposure < 10%), mean exposure, and every v2
metric.

**Post hoc, non-decisive:** E7 vs default on the real research window (2005–2024, corrected v1.1
data), logged as a trial in the real-data registry, never on the real holdout.
