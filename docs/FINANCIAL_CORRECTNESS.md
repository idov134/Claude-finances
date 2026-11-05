# Financial correctness and data contract — first release

Audit date: 5 November 2026. Baseline: `GPT` at
`14c95a03387f7288da18b5e095aeeadbf1f90f2f`. The existing synthetic research and
used 2024–2025 holdout remain historical evidence, not a demonstrated market edge.

## One account model and one valuation basis

The application and CLI use `stocksel.live.paper.PaperLedger`; there is no second
account engine for the web workflow. The account is long-only, unlevered, and
supports fractional **canonical adjusted-series units**. They are research units,
not a promise that they equal a broker's current raw share count. `DataBundle`
prices are specified as split/dividend adjusted, with a total-return benchmark.
The ledger executes at the next available open after the recorded decision date.
It never fills a decision at the same day's earlier opening price.

Corporate actions are deliberately bounded in this release:

- Split and dividend effects already present in the canonical adjusted series
  are consumed through those prices. No dividend cash is credited again and no
  second split is applied to these adjusted units.
- Separate cash dividend income is **unavailable (`null`)**, not zero. Ex-dividend,
  payment-date, withholding-tax and raw-share accounting remain deferred.
- Explicit raw-price bundles are rejected. Changing a held security's historical
  mark in a refreshed adjusted series is rejected until reconciliation. A source
  change cannot silently continue the same account.
- Raw real-time quotes remain unsuitable for adjusted-unit marks or stop checks
  without verified corporate-action reconciliation. This release does not add
  such reconciliation or turn the existing safety block off.
- Missing prices do not imply delisting or a zero return. The last known mark is
  retained with its date; unavailable-open instructions remain pending. In the
  synthetic contract, an explicit `delist_returns` entry settles once on the
  represented session after its last valid price: proceeds are adjusted units ×
  last close × (1 + supplied delisting return), with zero additional modeled
  settlement fee. Cost basis is released to realized P&L and affected pending
  instructions are resolved. The return becomes available through the existing
  synthetic truncation firewall; no missing price alone triggers settlement.
  Real adapters do not carry dated settlement metadata: a known held delisting
  with absent prices therefore blocks ledger advancement for reconciliation.
  Real delisting cash dates, identifier changes and mergers remain deferred.

The first supported historical currency is USD. The ledger persists its currency
and original capital. A changed configuration cannot silently re-denominate an
existing account. Data whose currency differs from the account is rejected;
missing currency metadata in existing US-oriented adapters retains their USD
contract. USD/ILS translation requires dated FX observations and is not available.
No Israeli tax assumptions or external deposits/withdrawals are introduced.

## Deterministic accounting definitions

| Quantity | Definition / convention |
|---|---|
| Cash | Prior cash minus signed execution notional minus modeled transaction cost |
| Position market value | Adjusted units × latest compatible recorded adjusted close |
| Equity | Cash plus the sum of position market values |
| Purchase cost basis | Prior remaining basis plus purchase notional plus purchase costs |
| Gross average entry price | Quantity-weighted purchase prices; distinct from fee-inclusive basis |
| Sale basis release | Remaining basis × units sold / units before sale |
| Realized P&L | Sale proceeds minus sale costs minus released fee-inclusive basis |
| Unrealized P&L | Current position value minus remaining fee-inclusive basis |
| Total P&L | Equity minus persisted initial capital; equals realized plus unrealized for complete new accounts |
| `fees_paid` | Sum of all modeled trading-cost dollars, including commission, spread, slippage and impact; not just broker commission |
| Cumulative / time-weighted return | Equity / initial capital − 1; equivalent here because external cash flows are unsupported |
| Investor money-weighted return | Unavailable; no deposits/withdrawals or IRR calculation implemented |
| Paper duration | Elapsed calendar days / 30.44, a duration diagnostic only |
| Live readiness | Never inferred from paper duration; `live_ready` remains false |

The initial-capital denominator is persisted in schema v2. Legacy accounts with
complete trade records can reconstruct it as current cash plus signed historic
notionals and trading costs. If legacy records cannot establish basis or initial
capital, affected values remain `null` with an accounting warning. No guessed
basis is presented as a verified fact. Existing positions, cash, trades and
equity records are preserved by migration.

### Hand-checkable trace

The deterministic regression fixture begins with USD 1,000, constant USD 10
prices, and a 1% commission with other cost terms set to zero. A 50% decision at
the first close buys 50 adjusted units at the next open:

1. Notional USD 500, purchase cost USD 5, cash USD 495.
2. Remaining basis USD 505 and equity USD 995; unrealized P&L is −USD 5.
3. A later zero-weight decision sells 50 units at USD 12 on the next open.
4. Proceeds USD 600, sale cost USD 6, realized P&L = 600 − 6 − 505 = USD 89.
5. Final cash/equity USD 1,089, total modeled costs USD 11, cumulative return 8.9%.

This verifies arithmetic and order timing; these prices are a hand-constructed
engineering fixture. They make no investment performance claim. The existing
end-to-end test additionally follows a synthetic stock from point-in-time
features through calibrated prediction, target sizing, recommendation, paper
submission and next-open execution.

## Verified defects fixed in this release

| Baseline evidence | Consequence | Fix / acceptance regression |
|---|---|---|
| `paper.py` used `adv.loc[d]` at opening fill | Closing volume/price unavailable at open affected execution cost | Lag ADV and volatility one session; perturb future close/volume without changing opening cost |
| Target sizing spent full pre-cost equity | Cash could go negative after costs | Solve affordable purchase notional including impact/costs; full-capital test never borrows |
| Orders sorted by target weight | A lower-weight purchase could precede a larger-weight reduction | Sort actual signed trade notionals; sell-before-buy test |
| Missing open skipped and whole order removed | A requested liquidation or purchase disappeared | Per-symbol remaining instructions persist until executable; held missing sales fund later purchases |
| Retries appended another identical order | Duplicate execution and journal entries | Persist idempotency key/payload hash and receipt; concurrent retries produce one decision |
| Entry price never updated on additional purchases | Unrealized/realized P&L and cost basis incorrect | Weighted gross entry and fee-inclusive average-cost basis; hand-check tests |
| Stop used a new full-portfolio target dictionary | Unrelated holdings could be rebalanced/erased | Stop instruction scope contains only the triggered holding |
| Direct `write_text` persistence | Interrupted saves could corrupt the only ledger | Same-directory temporary file, flush/fsync and atomic replace; previous valid file retained as `.bak` |
| Duration alone set `live_ready` | Time spent paper trading appeared to imply validated skill | Separate duration flags; keep readiness false |
| Update iterated data before the first decision | Long flat pre-decision history obscured the monitoring period | History begins at the decision date; no pre-decision fills or equity rows |
| An explicit delisted holding could retain its last mark indefinitely | Equity overstated and missing liquidation never resolved | Settle explicit synthetic delistings once; block undated real settlements; retain ordinary missing-price instructions |

Order target dollar amounts are fixed at the first execution opportunity. A
partial order retains only unfilled symbols. Missing funding sales defer buys;
a fully priced basket that cannot afford all desired purchases is capped to
available cash and records that limitation on the order. Costs are estimates,
not guaranteed broker fills. `cost_inputs_estimated` marks the documented
CostModel fallbacks when insufficient lagged ADV or volatility exists.

Stops are observed at a completed close and queued for a later available open.
A gap, suspended market or cost can produce a loss beyond the stop threshold;
there is no guaranteed maximum loss. Absent open prices are not fabricated.

## Persistence, concurrency and recovery

Schema v2 includes account identity, original capital, order IDs, payload hashes,
journal entries, accounting totals and position basis. A per-ledger thread lock
plus OS file lock serializes CLI/API instances during read-modify-write. Each
mutation reloads disk state under the lock. Explicit `save()` rejects stale
object revisions. POSIX `flock` is exercised here; the Windows `msvcrt` branch
is implemented but not exercised on this Linux host.

Every mutation serializes strict JSON before replacing the primary file.
`ledger.json.bak` contains the immediately preceding valid version. An unreadable
primary fails visibly rather than silently discarding account state. Recovery is
an explicit local operation: stop writers, preserve the unreadable file, inspect
both versions, and restore the desired valid backup. A backup is not a complete
historical archive; export the account for longer-term retention.

`submission(idempotency_key)` recovers a stored receipt after an API response or
separate preview-receipt write was interrupted. Same key plus different decision
content is rejected. A second distinct decision is blocked while a previous
order is pending, avoiding ambiguous overlapping target portfolios.

## Verified versus deferred

Verified by focused deterministic regressions: buys, purchase costs, sales,
weighted basis, realized/unrealized reconciliation, cash limits, actual
sell-before-buy, missing opens, future-input invariance, scoped stops, retry and
concurrency deduplication, restart lookup, strict atomic serialization, stale
writer refusal, legacy byte preservation, missing legacy basis and raw/revised
basis rejection. Existing research leakage tests remain in place.

Deferred: real-provider corporate-action reconciliation, raw dividends/taxes,
external flows and investor IRR, actual broker fill/slippage calibration,
changing security identifiers, real-provider delisting settlements, dated FX,
Windows runtime verification, and independently verified point-in-time vendor
coverage. Synthetic fixtures verify machinery only. Probability calibration and
investment usefulness require separate out-of-sample scorecards and real data.

## Research availability and dataset identity

The NYSE 16:00 strategy clock now uses exchange-calendar closes, including
13:00 early closes. An event timestamp equal to or later than the decision close
is deferred. Timezone-aware events convert to exchange-local time; naive
intraday timestamps are interpreted as local and retain their hours/minutes.
The configured publication lag applies to both forms. Synthetic business-day
grids may contain exchange holidays; those simulated rows retain the configured
clock and are not a claim of a valid real exchange session.

Trailing-twelve-month fundamental aggregates are delayed until every constituent
quarter has been published. A late Q3 filing cannot influence a Q4 aggregate
that appeared earlier. The whole joined row is delayed conservatively, including
its balance-sheet fields; first-report semantics are retained and restatements
are not reconstructed as a full version history. Dedicated tests mutate the
future quarter and prove the earlier feature is unchanged.

Bundle cache identity now includes the complete data and market configuration
(including benchmark and timezone), plus cache schema version 2. Changing a
benchmark or interpretation clock cannot reuse a previous market's bundle.
This is configuration identity, not immutable vendor-file content versioning:
CSV file changes at the same path still require `use_cache=False`/a refresh, and
vendor historical revisions must be separately versioned and reconciled.

Polygon's price-only adapter no longer declares survivorship-free or
point-in-time fundamental coverage. Sharadar's table-filter keyword collision
is corrected with an offline request-shape test. Neither correction establishes
live entitlement, vendor completeness or market-data quality; those remain
unverified without authorized provider access.
