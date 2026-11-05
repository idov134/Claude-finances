# First release: research to paper decisions

Audit date: **2026-11-05**. Starting branch `GPT` was clean at
`14c95a03387f7288da18b5e095aeeadbf1f90f2f`; the remote ref matched it. Work is on
`improvement/research-paper-v1`. Existing source, trial records, old reports and
the used 2024–2025 synthetic holdout are preserved. The release retains Python,
FastAPI and the existing plain-JavaScript frontend. It adds no brokerage path,
paid subscription, public deployment or autonomous live trading.

## Baseline and bounded release

Baseline command `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/workspace/stocksel-venv/bin/python -m pytest` completed **79 tests in 59.14 s**.
The existing app started with `python -m stocksel app --port 8000`, producing
378 eligible synthetic candidates. Before interface edits, Chromium inspection
at 1440×1000 and 390×844 showed no page-width overflow. The measured baseline
snapshot median was 131.50 ms over ten requests; first loaded render was
930.32 ms; alternating filter median was 54.85 ms, with full rebuilds around
94–174 ms. These are local engineering measurements, not user-study findings.
Screenshots are retained under `docs/screenshots/baseline-*.png`.

The budget set before optimization is snapshot API p95 below 500 ms, initial
render below 2 s, filter median below 50 ms and full-filter rebuild below 100 ms
on the same local fixture/hardware. Final measurements, actual browser task
steps and any exceptions belong in `VALIDATION_REPORT.md` and the screenshot
metrics export; baseline values are not substituted for final results.

The first release completes one workflow: inspect status → screen/filter/save a
view → inspect a stock → compare alternatives → preview a bounded paper weight
change → confirm it once with a research note → advance a later simulated
session → inspect persisted orders, fills, costs and the account. Synthetic
replay advances explicitly so refresh/reload does not consume a different
trading decision. Real-feed accounting remains blocked pending corporate-action
reconciliation. Broader dashboard customization and new alpha models are deferred.

## Prioritized audit and acceptance matrix

Code evidence identifies the affected functions in the baseline/current modules,
rather than relying on line numbers that move during implementation.

| Priority | Issue and code evidence | Consequence | Release fix / boundary | Acceptance check |
|---|---|---|---|---|
| P0 | `live/paper.py:PaperLedger.update` used current-session ADV at the opening fill | Future closing volume/price altered an earlier execution cost | Lag ADV and volatility one session; label fallback estimates | Mutate future close/volume; opening cost unchanged |
| P0 | `PaperLedger.update` spent pre-cost equity and ordered trades by target weights | Negative cash; buys could precede funding sales | Reserve costs in affordability solve; actual negative notionals first | Full-capital purchase stays unlevered; sell-before-buy fixture |
| P0 | Missing opens discarded by removing the whole order | Risk reduction or intended purchase vanished | Preserve per-symbol remaining instructions, defer buys awaiting missing sales | Missing-open liquidation later executes exactly once |
| P0 | Position `entry_px` remained its first price; no released basis/realized P&L | Misleading reported gains after additions/sales | Weighted gross entry and fee-inclusive average cost basis | USD 1,000 buy/sell trace reconciles to USD 1,089 |
| P0 | Plain JSON write and repeated submissions appended new orders | Corruption or duplicate simulated trades | Atomic replacement, prior-version backup, locking, schema migration and persisted idempotency key | Retry/restart/concurrent submissions create one decision; stale writer refused |
| P0 | `data/pit.py:decision_timestamps` hardcoded 16:00; `availability` normalized naive intraday timestamps | Post-close earnings/news leaked into early-close signals | Exchange close times, preserved intraday local timestamps, consistent lag | Early-close, DST and exact-close leakage tests |
| P0 | `features/fundamental.py:_ttm` rolled quarter values before publication-aware join | A delayed quarter leaked into an earlier-published TTM row | Delay aggregate availability to the latest constituent publication | Change unavailable Q3; February feature unchanged |
| P0 | `data/__init__.py:load_bundle` cache omitted market config | Benchmark/timezone switch could silently reuse wrong data | Cache includes market+data settings and schema version | Changed benchmark/clock uses distinct bundle; unchanged config reuses cache |
| P0 | `providers/polygon.py` certified PIT/survivorship though it loaded prices/current share counts only | Unsupported quality flags could clear promotion gates | Flags false for this adapter; Sharadar table-keyword collision fixed separately | Offline capability and request-shape tests; paid/live access still unverified |
| P0 | Raw quotes could differ from adjusted historical/ledger units | False P&L and stop comparisons across splits | Retain block; only compatible synthetic marks are used in this release | Raw/adjusted and source-change rejection; no quote substitution |
| P1 | Known delisted holdings could stay at a stale mark forever | Equity overstated; pending liquidation unresolved | Once-only explicit synthetic settlement; real undated settlements block | No settlement before known event; retry does not credit twice |
| P1 | `paper.py` full target stop order included unrelated positions | Triggered stop could alter other holdings | Scope stop instruction to one holding | Other position's quantity/trade count unchanged |
| P1 | Duration alone set `live_ready` | Mere elapsed time implied economic validation | Separate duration diagnostics; live readiness remains false | Account maturity cannot promote a strategy |
| P1 | UI had transient ticker filter, modal detail and no preview-to-ledger path | Repeated research setup, lost context, no complete decision record | Persisted screens/watchlists/columns, comparison and linked detail, explicit preview/confirm/journal | Reload restores settings; find/compare/preview/confirm/advance workflow completes |
| P1 | Source/as-of/event definition absent from several displayed estimates | Probability, rank and confidence could be confused | Named forecast event/horizon, separate timestamps, evidence and units | Stock detail/API disclose definitions; unavailable values remain null |
| P1 | Research results lacked a bounded transparent comparison/economics view | More complex models could appear useful without fair references | One preregistered synthetic comparison with logit/factor/equal/cash/benchmark and cost sensitivity | Every attempt registered, same permitted window, no promotion from simulation |
| P2 | Current vendor pricing/competitor claims could not be fetched | Unverifiable cost/performance claims | Mark unknown; verify accessible official QuantConnect source only | Source matrix distinguishes independent evidence from supplied hypotheses |

The implemented checks establish arithmetic and workflow properties. They do not
certify vendor coverage, profitable forecasts, successful public deployment,
live fill quality or long-term usability. `FINANCIAL_CORRECTNESS.md` is the
accounting/data contract; `COMPETITOR_RESEARCH.md` records source access; the
validation report records actual runs rather than intended acceptance alone.

## One stock traced through the system

`load_bundle` supplies a canonical date×ticker panel and availability-stamped
events. `data/universe.py:eligible` applies price, capitalization, liquidity,
history and membership filters. `features/pipeline.py` joins only available
fundamental/news inputs and normalizes the permitted cross-section.
`models/train.py:fit_calibrated` splits training/calibration in time with label
purging, then `live/recommend.py` maps its calibrated excess-return event to
haircut probability/expected excess and `portfolio/sizing.py` applies position,
sector, exposure and risk limits. The web preview keeps the chosen signal record
and current account state, estimates a target change and asks for an explicit
paper confirmation. `PaperLedger.submit` stores the order and research journal;
`update` fills only a later usable open, charges modeled costs and records units,
cash, basis and realized/unrealized accounting. The UI reads that same account;
it has no separate balance model. The deterministic arithmetic trace and test
names are in `FINANCIAL_CORRECTNESS.md`; full scanner/browser execution is in the
validation report. No single synthetic stock trace establishes real alpha.

## Separate acceptance scorecards

| Scorecard | Evidence collected | What would strengthen it |
|---|---|---|
| A — calculations/software | Existing tests plus hand-checked accounting, availability, provider flags, cache and persistence regressions | Dated real corporate-action fixtures, audited vendor contract, Windows CI |
| B — forecast quality | Purged OOS logit versus constant probability; Brier/log loss, calibration bins and rank IC in the new comparison | Locked real PIT data, repeatable independent validation window |
| C — investment usefulness | Synthetic net/cost/risk/subperiod/economics diagnostics; failures and old trial uncertainty retained | Investable total-return benchmark, full prior trial matrix, real implementation costs and prospective paper results |
| D — usability/reliability | Actual local desktop/mobile task execution, persistence and measured latency | User task study, slow-network/accessibility testing, longitudinal provider failures |

## Deferred roadmap, ordered by value

Effort is a relative engineering estimate, not a vendor quote or commitment.

| Next improvement | Benefit | Main risk/dependency | Effort | Cost boundary |
|---|---|---|---|---|
| Versioned real PIT prices/universe/events with verified entitlements | Makes statistical evaluation economically meaningful | Coverage, delistings, changed identifiers and licensing | High | Vendor price unknown until official quote/access verified |
| Raw-share corporate-action and dated cash settlement reconciliation | Unlocks real-feed P&L/stops safely | Split factors, ex/pay dates, mergers, tax treatment | High | Licensed event data likely; no purchase authorized |
| Prospective same-period paper/reference reconciliation | Separates selection, exposure, timing and implementation differences | Requires real consistent data and months of observations | Medium implementation; months elapsed | Provider and compute costs measured separately |
| Full trial provenance across prior versions | Enables defensible combined DSR/PBO | Original 48-trial return matrix absent; cannot reconstruct by relabeling | Medium if originals supplied | Storage/compute; no new parameter search |
| Security identifiers, restatement versions and incremental vendor caching | Reduces stale/mismatched data and repeated downloads | Vendor identifiers and availability semantics | High | Depends on provider quotas/license |
| Dated USD/ILS reporting and explicit external flows/IRR | Useful Israel-based account reporting | FX source, timestamps and flow timing; translation separate from return | Medium | FX licensing unknown; no guessed tax rules |
| Hebrew/RTL localization and broader keyboard/screen-reader study | Improves accessibility for the user's locale | Human terminology review and chart/table directionality | Medium | Translation/review time; no financial API dependency |
| Live price/benchmark attribution and persistent historical score snapshots | Explains outcomes and rating changes across time | Immutable source/version records | Medium | Storage growth; bounded retention/export needed |
| Cancellation/progress, diagnostics and incremental refits | Better behavior for long data/model jobs | Safe cancellation and immutable job snapshots | Medium | Measured compute budgets, no heavy queue required yet |
| Authentication, secrets management and deployment controls | Enables deliberate multi-user hosting | Threat modeling and access controls | High | Hosting/security cost; deployment outside this task |
| Optional grounded narrative summaries | Reduces reading effort | Citations, hallucination, privacy and token cost | Low–medium | Explicit opt-in and capped budget; core remains offline |

Complex draggable dashboards, additional indicators, new model families,
aggressive sizing and enabling failed news/re-entry strategies are not first
release goals. The profitable outcome may remain a simple benchmark after costs.
