# First working release validation

Date: **2026-11-05**. Base: clean local and remote `GPT` at
`14c95a03387f7288da18b5e095aeeadbf1f90f2f`. Release branch:
`improvement/research-paper-v1`. No reset, deleted research history, reopened
holdout, paid subscription, brokerage execution or public deployment occurred.

## Verified software and calculations

Commands below ran from `/workspace/Claude-finances` with Python in
`/workspace/stocksel-venv`. Equivalent activated-venv commands are in README.

| Check | Observed result |
|---|---|
| Baseline `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest` | 79 passed in 59.14 s before edits |
| Intermediate integrated suite | 104 passed in 63.09 s |
| Intermediate expanded suite | 114 passed in 180.12 s; concurrent cold scanner/browser work affected wall time |
| Final full release suite | **116 passed in 88.34 s**, zero failures/skips |
| Final affected API, workflow, feed integration and PIT suites | **23 passed in 11.10 s**, after the final quote-order/currency/mark-clock changes |
| `python -m pip check` | No broken requirements |
| `python -m compileall -q stocksel` | Passed |
| `node --check stocksel/web/static/app.js` and `workflow.js` | Passed |
| `git diff --check` | Passed |
| `python -m stocksel app --demo --port 8002` and local snapshot request | Ready; 378 candidates, signal date 2023-11-30; separate persistent demo account |

The full suite emitted a Starlette deprecation notice about its httpx TestClient,
and two scipy constant-input correlation warnings in a deliberately constant
forecast fixture. They were not suppressed to claim success. Undefined
correlations remain unavailable. No leakage or sanity assertion was disabled.

Regression coverage includes fee-inclusive purchases/sales and weighted basis,
cash affordability, actual sells before buys, retained unfilled orders, scoped
stops, known synthetic delisting settlement exactly once, refusal of undated real
settlement, incomplete legacy basis remaining null, file/restart/concurrent retry
behavior, stale-writer refusal, source/adjustment incompatibility and unsupported
currency. Future-input perturbations cover opening costs, after-close events,
early exchange closes and delayed quarterly aggregates. Cache identity now
includes benchmark and market clock.

The paper API is local and synthetic-only for this release. Preview hashes bind
the account/configuration; expirations and changed state require a new preview.
Persisted ledger idempotency recovers a successful submission even if a separate
HTTP receipt was lost. Wrong-origin requests, missing mutation headers, invalid
hosts, invalid targets and unpreviewed requests fail. No brokerage endpoint is
activated. These tests do not constitute a public-hosting penetration test.

## Actual application and browser checks

The application was started with `python -m stocksel app --port 8000` for the
baseline and port 8001 for the integrated release. Both produced **378 eligible
synthetic candidates**. The stock-detail API charts only data through the replay
event, labels the event/horizon of its probability, and preserves unavailable
history gaps. Prices refreshing does not refit the model or change its signal
date. The demo reserves a later research month for replay and excludes 2024–2025.

Chromium (`/usr/bin/chromium`, driven by Playwright) exercised the real API at
**1440×1000 desktop** and **390×844 phone** widths. The scenario completed:

1. Filter candidates, save a sector view and reload it; star a watchlist entry
   and verify that it survives reload.
2. Compare up to three stocks; retain table filters/sort/scroll while opening
   a linked stock detail with indexed stock/benchmark history.
3. Enter a thesis/counter-thesis; refresh quotes without losing unsent text.
4. Preview target exposure/costs, confirm a paper decision, repeat the same
   submission and verify that it creates no duplicate order.
5. Advance one session; verify later-opening fills, nonnegative cash, persisted
   trade history/journal and JSON export. Restart retained the original account.
6. Export the full filtered CSV; change display time zone and reload; inspect
   all five screens on the phone layout and toggle light/dark themes.

There were no JavaScript errors or phone body-width overflow. The actual
diagnostic account retained two decisions/two trades through 2023-12-05; the
last checked order ID was `4ef3dc13d5e49f813f4e9fac`. These are synthetic test
decisions, not live executions or investment recommendations. Local demo state
is ignored by Git; screenshots and measurement records contain no credentials.

Verified browser failures were fixed: saved sector initialization lost its
selection, a post-action refresh could return before an in-flight poll completed,
and both-missing chart rows were dropped so lines crossed unavailable periods.
Workspace imports are shape-checked, CSV string formulas escaped, and pending
paper request retries keep their idempotency key.

## Measured responsiveness

Budgets were declared before optimization: API p95 <500 ms, ready-state initial
render <2 s, filter median <50 ms, full-filter render <100 ms. These are local
interaction budgets, not a guarantee of production capacity. Both measurements
used the same environment and 378-candidate fixture; the final fixture also
contained recorded paper history. The table is paginated at 75 visible rows,
while filtering/export operate on the complete eligible set.

| Measurement | Baseline | Final | Assessment |
|---|---:|---:|---|
| Snapshot API median, ten requests | 131.50 ms | 167.66 ms | Added state/provenance costs; no speedup claimed |
| Final API maximum, ten requests | Not recorded | 307.30 ms | All sampled requests below 500 ms |
| Ready-state first loaded render | 930.32 ms | 1753.20 ms | Below 2 s budget; slower than baseline |
| Alternating filter median, twenty inputs | 54.85 ms | 10.10 ms | Below 50 ms budget |
| Full-filter rebuild | Approximately 94–174 ms | Maximum 31.20 ms | Below 100 ms full-render budget |
| Worst filter sample | Not retained as an aggregate | 750.60 ms | Narrow-filter tail outlier remains; no tail-latency certification |

Raw samples and task checks are in
[`screenshots/final-metrics.json`](screenshots/final-metrics.json). The cold
scanner exceeded a 60-second browser wait and was still loading at approximately
2m42 during concurrent validation after cache invalidation. Its exact cold-ready
time was not instrumented. The loading UI remained responsive and eventually
completed. Ready-state timings explicitly exclude model startup. Incremental
feature/model snapshots and job progress/cancellation remain roadmap work.

Screenshots: [baseline desktop](screenshots/baseline-desktop.png),
[baseline phone](screenshots/baseline-mobile.png),
[final screener](screenshots/final-desktop.png),
[final phone](screenshots/final-mobile.png), and
[stock detail/paper preview](screenshots/final-detail.png).

## Forecast quality and investment usefulness

The preregistered comparison completed in **102.28 s**, within its 300-second
budget, on physically truncated synthetic research data ending 2023-12-29.
There are 239 OOS monthly periods and 11 completed final configurations:
current logit, a fixed transparent momentum/low-volatility rule, equal-weight
eligible stocks, cash and a same-window synthetic benchmark, with cost scenarios.
All attempts remain in the registry: **33 implementation/configuration IDs and
12 recorded failures**, including earlier correctness failures. No favorable
result was obtained by erasing those records or searching new parameters.

| Evidence | Result and limitation |
|---|---|
| Forecast | Brier 0.249340 versus 0.250000 constant probability; log loss 0.691827 versus 0.693147; logit rank IC 0.04782 |
| Current logit net CAGR | 12.07% versus simulated benchmark 9.11%; synthetic only |
| Annual mean net benchmark excess | +1.65 percentage points, 95% CI **−4.47 to +6.24**; does not establish outperformance |
| Factor/equal-weight/cash CAGR | 10.94% / 6.45% / 1.45%; same permitted comparison window |
| Multiple testing | Combined DSR/PBO unavailable because original full trial returns are missing; subset PBO is a diagnostic, not certification |
| Economic viability | Illustrative ownership-cost/break-even scenarios reported separately; intervals crossing zero prevent a viability claim |

Full conventions, risk/cost/subperiod metrics, forecast bins, ranking buckets,
training boundaries, source/config/input hashes and retained failure records are
in [the comparison results](research_comparison/RESULTS.md) and adjacent JSON/CSV.
The benchmark is a synthetic cap-weighted total-return approximation, **not a
verified investable ETF**. No candidate becomes the default. Re-entry and news
remain disabled. The original used holdout remains used.

The legacy `research` monthly engine is still a separate implementation with
missing-return substitution and gross-equity weight drift. The corrected bounded
comparison does not certify or relabel those legacy results. Aligning that path
is a prerequisite before certified new production research, and is explicitly
deferred rather than silently treating it as verified.

## Source, data, environment and release limits

Official competitor websites were blocked by the configured proxy (CONNECT403),
not checked through a paid account. QuantConnect's official GitHub documentation
was accessible and independently inspected. Other supplied workflow patterns,
current portfolio claims and current vendor prices remain unverified. Details
and data-option tradeoffs are in `COMPETITOR_RESEARCH.md`.

No data credentials were configured. Actual live-provider entitlement, PIT
coverage, exchange completeness, raw shares/splits/dividends, real settlement
dates, security identifier changes and real fill costs remain blocked checks.
Web paper execution stays synthetic-only. USD is the only supported web valuation
currency; Israel/NY/UTC display clocks do not imply FX conversion. ILS translation,
cash-flow IRR, Israeli taxes and full Hebrew/RTL are deferred. Core accounting
has no LLM or paid-service dependency.

`runtime_manifest.json` records tested top-level dependency versions and their
installed license metadata. `pip check` establishes dependency consistency,
not vulnerability clearance. No complete transitive vulnerability/license audit,
exact dependency lock, Windows runtime test, screen-reader study or public
authentication/deployment test was performed. No repository LICENSE file was
found; distribution licensing requires an owner decision. Local startup works
from the checkout; a standalone packaged installer is deferred.

The next highest-value work is to align the legacy research engine, obtain a
versioned authorized real PIT dataset and corporate-action contract, and run
prospective paper/reference reconciliation against a verified total-return
benchmark. The profitable conclusion may still be simple benchmark ownership.
