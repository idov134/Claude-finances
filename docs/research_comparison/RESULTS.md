# First-release research comparison results

Run date: **5 November 2026**. All results below are **synthetic engineering diagnostics**. No real-market advantage, brokerage fills or vendor entitlement is established. No candidate is promoted and no used holdout was reopened.

## Reproduction and boundaries

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_research_comparison.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m stocksel.experiments.comparison
```

The final run completed in **102.28 seconds**, within the predeclared 300-second compute budget. Seven focused diagnostic tests passed. The run used the unchanged [pre-registration](PROTOCOL.md), default seed 7 and 1,004 synthetic tickers. Data was physically truncated to **2023-12-29** before features and labels; **2024-01-01 onward was excluded**. It produced 77,493 purged walk-forward predictions, with 76,729 available realized labels. The 239 complete monthly decision dates are 2003-12-31 through 2023-10-31; each return runs from the following opening price to the next rebalance opening price. Detailed training/calibration boundaries are in `comparison.json` → `folds`.

- Configuration SHA identifier: `882fae5f0d64fd08`.
- Permitted input content SHA256: `79372dbe60dd55b0adb0a3ced9510724e2593db12a5f84d27ca605a64325aecb`.
- Research source SHA256 at execution: `b71e62b4799d11189fd43b44ab49b2d05d4a6023a87398639c990bd5f3913a39`.
- Full configuration, dependency versions and exact performance conventions are included in [comparison.json](comparison.json). Returns and the append-only registry are exported beside it.

The current-logit portfolio uses the configured monthly logistic model, haircut, buffer and sizing. The transparent factor uses a fixed momentum/low-volatility ranking with equal weights among the top configured N. The all-name equal-weight portfolio shares the same eligible universe, caps and ex-ante volatility target. Cash receives the supplied synthetic risk-free rate. The benchmark is the synthetic cap-weighted total-return approximation; it is **not a verified investable ETF**. The comparison has no intramonth stops or kill switch and is not the separate re-entry experiment.

## A. Software and calculation evidence

The diagnostic checks fee reserves, next-open execution availability, strict missing-return handling, probability/rank separation, drawdown duration, economic break-even edge cases and the holdout firewall. Costs use signal-date inputs. Weights drift using net equity. Next-open buys with no opening price remain unfilled cash, without choosing replacement winners. Terminal delisting proceeds already included in forward returns become cash and are not sold again.

Earlier attempts are retained: eleven failure records came from a pandas date/index ambiguity, followed by one strict missing-return failure exposing a selected name without a next-open price. The former was corrected without changing the hypothesis; the latter was resolved with explicit unfilled-order accounting, not a zero-return substitution. No configuration was tuned to improve its financial outcome. Registry timestamps span approximately 32 and 32 seconds for those failed attempts, plus 102 seconds for the completed run: approximately 166 seconds of observed run windows, excluding idle time between attempts. The first failed revision did not yet record a research-code hash; its original records are preserved rather than retrospectively inventing one.

There are **33 distinct implementation/configuration identifiers** across the three code revisions (planned attempts count), **12 recorded failures**, and **11 completed configurations in the final run**. Earlier planned configurations that did not reach evaluation remain planned. The prior documented research/re-entry history adds at least 50 attempts. The original full return matrix is unavailable, so **combined DSR and combined PBO are unavailable**. The 11 completed-column PBO diagnostic is 0.000 across 5,000 combinations; it excludes failed/unrealized trials, includes correlated cost scenarios, and cannot certify that selection risk is absent or improved.

## B. Forecast quality

| Metric | Current logit | Constant 0.5 probability |
|---|---:|---:|
| Brier loss (lower is better) | 0.249340 | 0.250000 |
| Log loss (lower is better) | 0.691827 | 0.693147 |
| AUC | 0.523713 | 0.500000 |
| Accuracy (not an investment criterion) | 52.07% | 51.66% |

The probability event is positive stock excess return versus the same-window benchmark over one month, next open to next rebalance open. Monthly cross-sectional rank IC is **0.04782** for the logit and **0.02865** for the fixed factor. No 65% accuracy or Sharpe>2 leakage flag was triggered. These modest forecast differences on planted synthetic signals do not establish real-market skill.

| Forecast probability bin | N | Mean probability | Observed event fraction |
|---|---:|---:|---:|
| (0.3, 0.4] | 150 | 38.87% | 39.33% |
| (0.4, 0.5] | 55,586 | 47.49% | 47.46% |
| (0.5, 0.6] | 20,991 | 51.21% | 50.76% |
| (0.6, 0.7] | 2 | 60.52% | 100.00% |

Empty calibration bins remain null. The two observations above 0.6 cannot support a high-confidence calibration claim. Rank quintile returns (gross, not a net executable portfolio) are:

| Score | Lowest quintile monthly excess | Highest quintile monthly excess |
|---|---:|---:|
| current_logit | -0.66% | 0.65% |
| transparent_factor | -0.40% | 0.28% |

## C. Investment usefulness

| Rule, base costs | Net CAGR | Annual mean excess | Excess 95% CI | Sharpe | Month-end max DD | Annual turnover | Mean exposure |
|---|---:|---:|---|---:|---:|---:|---:|
| current_logit_cost1 | 12.07% | 1.65% | [-4.47%, 6.24%] | 1.043 | -22.61% | 8.02× | 67.33% |
| transparent_factor_cost1 | 10.94% | 0.67% | [-5.17%, 5.37%] | 0.900 | -19.79% | 4.22× | 76.35% |
| equal_weight_cost1 | 6.45% | -3.54% | [-9.59%, 1.07%] | 0.528 | -23.78% | 1.04× | 71.42% |
| cash | 1.45% | -8.86% | [-17.38%, 0.41%] | Unavailable | 0.00% | 0.00× | 0.00% |
| synthetic_benchmark | 9.11% | 0.00% | [0.00%, 0.00%] | 0.496 | -46.91% | 0.00× | 100.00% |

Net CAGR is compounded growth annualized at 12 periods/year. Annual mean excess is `12 × mean(strategy − benchmark)`, not the difference of CAGRs. Sharpe uses monthly strategy return minus the supplied annual synthetic risk-free rate/12, divided by its sample standard deviation and multiplied by √12. Cash has zero excess-return variance and therefore undefined Sharpe; runtime warnings for that division were retained and the output is null. Confidence intervals use 500 stationary-block-bootstrap draws, expected block six months, seed 3. Drawdown is month-end only; gaps and intramonth losses can be worse.

**All three portfolio rules have benchmark-excess intervals containing zero.** The logit point estimate beats this synthetic benchmark, but the experiment does not establish outperformance even on this simulated path. Concentration, drawdown duration, worst month, fifth-percentile loss, cumulative return, terminal equity and fixed subperiod metrics are available per configuration in the JSON.

| Rule | Net CAGR at 1× costs | At 2× | At 3× | Cost drag at 1× per year |
|---|---:|---:|---:|---:|
| current_logit | 12.07% | 11.42% | 10.78% | 0.59% |
| transparent_factor | 10.94% | 10.58% | 10.22% | 0.33% |
| equal_weight | 6.45% | 6.36% | 6.28% | 0.08% |

Ordinary and risk-matched comparisons use the same dates. Risk matching scales the benchmark using observed full-window volatility and invests the remainder in cash; it is an **ex-post descriptive comparison**, not a prospective trading rule.

### Trading costs versus ownership costs

The assumed annual USD600 subscription/compute expense is an **illustrative configurable scenario, not a verified vendor price**. It is separate from modeled trading costs and is not deducted from the strategy-return table above. At USD100,000 initial capital it is 0.60% of initial capital/year.

| Rule | Total break-even trading cost, bps/traded dollar | Additional headroom beyond modeled costs | Capital to cover USD600/year if point estimate persisted |
|---|---:|---:|---:|
| current_logit_cost1 | 27.890 | 20.527 | USD36,446 |
| transparent_factor_cost1 | 23.628 | 15.885 | USD89,588 |
| equal_weight_cost1 | Unavailable | Unavailable | Unavailable |

Break-even total bps is `gross annual mean benchmark excess / annual gross turnover × 10,000`. Additional headroom uses net excess instead. Illustrative capital is `fixed annual USD cost / net annual mean excess`, only for a positive point estimate. These linear calculations ignore tax, account-size-dependent market impact, FX and changing costs. Because the excess intervals cross zero, **none is evidence that a particular account size makes the strategy economically viable**. Negative-edge alternatives remain unavailable, not zero-cost successes.

## D. Product and operational evidence

This module supplies portable JSON/CSV evidence and a complete retained trial log; it does not demonstrate real provider access or UI usability. See the release validation report for browser/task checks. Real survivorship-free point-in-time data, a verified investable total-return reference, actual entitlement/corporate-action reconciliation and prospective paper evaluation remain prerequisites for financial claims.

The legacy `research` monthly engine is a separate implementation; its historical reports are not relabeled as these corrected comparison results. The fixed diagnostic engine intentionally uses stricter missing-fill and fee-reserve behavior. The next research task should reconcile that legacy path before certifying new production research runs. The existing used 2024–2025 lock is preserved, and future evaluation requires a genuinely new pre-registered window.

**Decision: retain the current defaults, keep guarded re-entry/news disabled, and ship the accounting and research tooling.** No synthetic-only winner is promoted.
