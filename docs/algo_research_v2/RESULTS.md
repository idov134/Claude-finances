# Algorithm research v2 — results

Pre-registration: [PROTOCOL.md](PROTOCOL.md), committed in `695b841` before any run. The
development results were committed in `a8eac89` before the confirmation stage ran. Raw outputs:
`dev_results.json`, `confirm_results.json`, `registry.jsonl`, `returns.csv`, `*_net_returns.csv`.
**Synthetic data only.** Nothing here is evidence of real-market returns.

## Development world (seed 7, 239 OOS months, Dec 2003 – Oct 2023)

Net of modeled costs, corrected monthly engine. The benchmark is the synthetic cap-weighted index
(9.11% CAGR, −46.9% max drawdown).

| Config | CAGR | Excess vs bench (95% CI) | Sharpe | Max DD | β | Turnover/yr | AUC | IC | Δ vs B0 /yr | P(Δ>0) | Pass? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **B0** baseline | 12.48% | +1.99% (−3.8, +6.5) | 1.094 | −22.1% | 0.47 | 8.0× | 0.524 | 0.049 | — | — | — |
| E1 sector-neutral | 12.39% | +1.89% (−3.8, +6.5) | 1.110 | −20.6% | 0.46 | 7.8× | 0.523 | 0.049 | −0.11% | 0.44 | no |
| E2 residual momentum | 10.92% | +0.53% (−5.8, +5.5) | 1.003 | −22.3% | 0.40 | 7.7× | 0.522 | 0.046 | −1.46% | 0.01 | no |
| E3 tails target | 13.28% | +2.69% (−3.3, +7.5) | 1.188 | −21.2% | 0.46 | 7.6× | 0.523 | 0.048 | +0.69% | 0.94 | **yes** |
| E4 smoothing 0.5 | 10.71% | +0.36% (−5.8, +5.4) | 0.962 | −22.8% | 0.43 | 5.3× | 0.520 | 0.040 | −1.63% | 0.00 | no |
| E6 tuned LightGBM via gate | 12.47% | +1.98% (−3.9, +6.5) | 1.093 | −22.1% | 0.47 | 8.0× | 0.524 | 0.049 | −0.01% | 0.23 | no (never admitted) |
| E5 relative mandate* | 16.01% | **+6.23% (+4.6, +7.9)** | 0.827 | **−44.1%** | 0.99 | 9.1× | 0.524 | 0.049 | +4.23% | 0.96 | n/a (mandate) |

\* Not adoptable: its drawdown breaks the configured 20% tolerance.

What the failures say (synthetic world):
- **Sector-neutral ranks (E1):** slightly better Sharpe and drawdown, no net gain. Sector moves are
  pure noise in this simulator, and the cross-sectional ranks were already dominated by stock-level
  information.
- **Residual momentum (E2) was worse,** contrary to my prior. The published definition regresses
  with an intercept over 36 months, so it measures *recent change* in idiosyncratic performance.
  The planted momentum drift is persistent (monthly autocorrelation 0.92), and total-return
  momentum captures its level better. Real-data evidence for residual momentum (Blitz et al.)
  remains; this simulator does not reproduce it.
- **Smoothing (E4)** cut turnover by a third (8.0× to 5.3×) and costs by 0.2%/yr, but delayed the
  fast signals (PEAD, revisions, reversal) by more than that. At a USD 100k account, costs are
  too small for smoothing to pay.
- **Tuned LightGBM (E6)** was never admitted by the gate (0 of 21 windows).

Multiple testing at the end of development: **93 distinct trials** in the full history (original
48, Codex 2 + 33, v2 10). DSR charged with all of them: B0 0.980, E3 0.994. PBO over the full
decision-indexed history matrix (239 × 67): **0.07**.

## Confirmation (fresh worlds 101, 202, 303 — run once)

| World | B0 Sharpe | B0 excess (CI) | C1 = E3, Δ vs B0 /yr (P) | E5 relative: excess (CI), max DD |
|---|---:|---|---|---|
| 101 | 1.018 | +2.25% (−2.7, +7.0) | +0.11% (0.65) | +8.90% (+7.1, +10.7), −42.4% |
| 202 | 1.016 | +3.00% (−1.9, +7.5) | −0.73% (0.05) | +9.49% (+7.3, +11.9), −35.3% |
| 303 | 0.926 | +1.05% (−4.1, +5.7) | −0.44% (0.11) | +9.49% (+7.7, +11.4), −43.7% |

**Decision: C1 not adopted.** Pooled C1 − B0 = −0.35%/yr (95% CI −0.76, +0.03), P(Δ > 0) = 0.03,
positive in 1 of 3 worlds. The development pass of E3 (P = 0.94) was a **false positive**.
Five single changes each faced a P ≥ 0.90 threshold, roughly a 10% false-pass rate per change. So
even with no real effect, the chance of at least one false pass was up to about 40%. That is why
the confirmation stage exists.

What generalised:
- **The baseline itself:** Sharpe 0.93–1.09 and IC ≈ 0.05 in all four worlds, but its excess over
  the raw index is never significant.
- **The relative mandate:** pooled +7.2%/yr vs B0 (CI +4.6, +9.8), significant excess over the
  index in every world, at drawdowns of −35% to −44%.

The bottleneck to "beating buy-and-hold" in this system is the **market exposure** chosen to
respect the 20% drawdown tolerance (β ≈ 0.5), not stock selection. That led to
[v3](../algo_research_v3/RESULTS.md).
