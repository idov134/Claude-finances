# GPT algorithm review

All experiments in this document use synthetic data. They validate research machinery,
not a tradable advantage or profits in real markets. The previously reported
2024–2025 holdout has been used and is not reopened or used for parameter choices.

## Pre-registered question

Does a conservative re-entry rule improve the permanent kill switch's out-of-sample,
net-of-cost returns without worsening evidence of overfitting? The rule is optional
and disabled by default. No parameter search is planned.

The one candidate waits at least 21 trading sessions after a kill, then permits
entry only on an already scheduled rebalance, once positions are liquidated,
the prior benchmark close is above its trailing 126-session average, and it has
recovered at least 5% from its close at the kill. Only information available
before the entry open is used. Restarting creates a new risk-budget high-water
mark; lifetime drawdown is still measured from the original account equity.
This can permit cumulative losses beyond a single 20% kill threshold, so it is
an experimental simulation option, not an automatic live trading policy.

Reference and candidate will share one set of purged walk-forward logistic-model
predictions, features, universe, sizing, transaction costs, and execution stops.
The last 24 months are physically truncated before feature construction. The
comparison uses the corrected daily execution engine, rather than comparing
against historical report numbers produced by a different engine.

Every configuration, including failures, is written to the append-only trial
registry. The two configurations are compared with block bootstrap confidence
intervals, CAGR, annualized net excess, Sharpe/DSR, drawdown, Calmar, Sortino and
information ratio. The known original research search of at least 48 trials is
charged in DSR. The original full trial-return matrix is not committed, so a
combined PBO comparison cannot be certified and the candidate cannot become the
default solely from this experiment.

## Correctness findings

1. Opening execution costs used the same day's closing volatility and dollar
   volume. These are unavailable at the open. Use the previous completed day's
   estimates and test that changing the execution day's future close/volume does
   not change the opening transaction cost.
2. Pending kill/stop exits were discarded when an opening price was unavailable.
   Retain the instruction until an executable opening price is available.
3. Drawdown calculations omitted initial account equity from the high-water mark,
   hiding a loss in the first period. Include an initial equity of 1 in the peak.
4. The daily engine's monthly return conversion started from its first post-trade
   close, dropping opening costs and the first day's P&L. Anchor returns to the
   initial account equity and benchmark returns to the first execution open.
   The monthly metrics' drawdown also misses intramonth losses; the comparison
   below therefore uses daily lifetime drawdown and corresponding Calmar.

## Other ideas and validation plan

Targets, industry neutralization, model hyperparameters, turnover optimization,
and news changes are not tried in this round. Trying several at once would
increase selection risk and make attribution unclear. No alpha claim is based
on synthetic planted signals. Historical LLM news evaluation still requires a
documented knowledge cutoff earlier than the evaluated articles.

Next, obtain survivorship-free prices, point-in-time index membership, corporate
actions and filings from an authorized provider. Version and freeze the dataset,
pre-register the research search and an untouched future test window, and retain
all trial returns. Any model choice must use purged training-window validation.
Evaluate net returns against both raw and volatility-matched benchmarks; do not
choose parameters using the already opened synthetic holdout. Follow with at
least 3–6 months of prospective paper trading, including measured spread, impact,
fill availability, data gaps, and stop/re-entry behavior before considering live
capital.

## Results

The default synthetic market (seed 7, 1,004 tickers observed before the firewall)
was physically cut at **2023-12-29**, excluding all data from **2024-01-01**.
The shared logistic model produced 77,493 stock-month predictions using purged,
rolling walk-forward fits. The daily comparison covers 240 monthly observations
(2004–2023), after training history, with execution costs and configured ATR stops.
No simulator parameters or thresholds were selected by searching this result.

| Metric | Permanent halt reference | Guarded re-entry candidate |
|---|---:|---:|
| Net CAGR | 2.19% | 11.81% |
| CAGR 95% block-bootstrap CI | −0.57% to 5.19% | 6.87% to 16.83% |
| Benchmark CAGR | 9.18% | 9.18% |
| Annualized mean net excess | −8.12% | +1.22% |
| Excess 95% CI | −16.66% to +0.12% | −4.87% to +6.25% |
| Sharpe | 0.173 | 1.129 |
| Sharpe 95% CI | −0.433 to 0.626 | 0.621 to 1.655 |
| Daily lifetime max drawdown | −20.44% | −20.73% |
| Month-end max drawdown (for comparison) | −19.45% | −19.45% |
| Calmar using daily drawdown | 0.107 | 0.570 |
| Sortino | 0.291 | 2.202 |
| Information ratio | −0.485 | 0.103 |
| DSR diagnostic, charged at least 50 trials | <0.000001 | 0.033 |
| Kill events / re-entry events | 1 / 0 | 1 / 1 |
| Total modeled transaction costs | $5,024 | $52,736 |

Candidate-minus-reference annualized mean net return improvement has a 95%
stationary block-bootstrap interval of **+5.18 to +13.58 percentage points**
(2,000 resamples, expected block length six months, seed 3). This measures
recovery from a permanent halt on this simulated path; it does not establish
outperformance of an investable benchmark. The candidate's excess-return
interval against the benchmark includes zero, and its lifetime drawdown is
slightly worse. A next-open liquidation can exceed the configured threshold
because of gaps and transaction costs; the threshold is not a loss guarantee.

**Trial accounting:** two distinct configurations were attempted; deterministic
reruns append records without hiding or resetting previous attempts. The
two-configuration PBO is **0.0108** over 5,000 CSCV combinations. This is a
shared pair-level statistic, not a separate PBO for each configuration. The
previous 48-configuration return matrix is unavailable, so neither combined PBO
nor a claim that PBO did not increase over the original research can be made.
The DSR diagnostic charges at least **50** trials and uses the larger of current
cross-trial Sharpe variance and a nonzero sampling-variance floor; unavailable
historical trial variance prevents full inferential certification. It also fails
the stated DSR >0.95 target. The reference is a pre-registered logistic model,
not the old report's ensemble, and both runs use the corrected engine, so these
numbers should not be compared as an algorithm improvement over the old
report's different 1.09 Sharpe.

**Decision: keep re-entry disabled by default.** Correctness fixes are adopted;
the candidate is available as an explicit simulation option. Synthetic-only
improvement, uncertain benchmark excess, higher lifetime drawdown, and incomplete
historical multiple-testing evidence do not meet the adoption rule.

Reproduce from the repository root with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m stocksel.experiments.reentry
```

The command never calls `holdout`, modifies a holdout lock, or starts live trades.
It records planned, successful and failed attempts in
`artifacts/gpt_review/trials/registry.jsonl`; results are in
`artifacts/gpt_review/review.json`. An evidence export, including the exact
configuration, trial log and return matrix, is retained under
[`reentry_review/`](reentry_review/) for audit. Accuracy above 65% or Sharpe above
2 aborts the experiment for investigation; ordinary low DSR or uncertain excess
is reported as a failed research objective rather than hidden.
