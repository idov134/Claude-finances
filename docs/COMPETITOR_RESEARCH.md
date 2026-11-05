# Competitor workflows and data options

Research date: **2026-11-05**. Scope: official public documentation and public
documentation assets, plus a read-only audit of Stocksel's adapters. No paid
product, brokerage account, subscription, or competitor portfolio was operated.

## Evidence and access limits

The supplied task specification lists prior research findings. They are useful
design hypotheses, but do not establish that a feature or disclosure is current.
The direct official website requests below all raised `requests.ProxyError` in
this environment; the representative TradingView diagnostic was **HTTPS CONNECT
403 Forbidden**. This is a proxy response, not an HTTP response from the origin
page. The restricted network policy does not list these product domains. TLS
verification and the configured proxy were retained; no restriction was bypassed.

One independent official-source route worked: QuantConnect publishes its own
documentation in the public `QuantConnect/Documentation` GitHub repository.
Its source was inspected at commit
[`72caf31cdb1db511120c60955852e00c1f388855`](https://github.com/QuantConnect/Documentation/tree/72caf31cdb1db511120c60955852e00c1f388855),
committed 2026-11-04. That is independently verified **documentation**, not a
hands-on test of the live service. Its public reconciliation screenshot was
also inspected. The other products' current features, pricing and methodological
disclosures remain blocked checks, explicitly identified below.

| ID | Official source attempted | Result and evidence boundary |
|---|---|---|
| T1 | [TradingView screener walkthrough](https://www.tradingview.com/support/solutions/43000718885-tradingview-screeners-walkthrough/) | ProxyError; CONNECT 403 independently extracted; no origin page read |
| T2 | [TradingView watchlists](https://www.tradingview.com/support/solutions/43000745825-mastering-the-tradingview-watchlists/) | ProxyError; page not read |
| K1 | [Koyfin My Dashboards](https://www.koyfin.com/help/mydashboards-myd/) | ProxyError; page not read |
| K2 | [Koyfin stock screener](https://www.koyfin.com/features/stock-screener/) | ProxyError; page not read |
| S1 | [Seeking Alpha Quant Ratings help](https://help.seekingalpha.com/premium/what-are-quant-ratings-and-how-do-i-use-them) | ProxyError; page not read |
| S2 | [Seeking Alpha Quant performance](https://seekingalpha.com/performance/quant) | ProxyError; current claim and applicable disclosures unavailable |
| S3 | [Seeking Alpha terms](https://about.seekingalpha.com/terms) | ProxyError; current terms unavailable |
| S4 | [Alpha Picks about](https://seekingalpha.com/alpha-picks/about) | ProxyError; current strategy-specific claim unavailable |
| Q1 | [QuantConnect reconciliation](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/reconciliation) | Website ProxyError; official GitHub source below read successfully |
| P1 | [Portfolio123 rank performance tests](https://portfolio123.customerly.help/en/articles/13770-how-to-run-a-rank-performance-test) | ProxyError; page not read |

### What was verified from QuantConnect

The official [reconciliation introduction](https://github.com/QuantConnect/Documentation/blob/72caf31cdb1db511120c60955852e00c1f388855/Resources/live-trading/reconciliation/introduction.html)
describes an out-of-sample backtest running over the same time period as a live
deployment and overlaying its equity curve with the deployment curve. It frames
deviations as matters to investigate, rather than proof that the model has alpha.

The accompanying official source documents three groups of differences:

- [Data](https://github.com/QuantConnect/Documentation/blob/72caf31cdb1db511120c60955852e00c1f388855/Resources/live-trading/reconciliation/differences-from-data.php): availability timestamps, custom-data delays, live event timing, split/dividend adjustment of indicators, and vendor revisions.
- [Modeling](https://github.com/QuantConnect/Documentation/blob/72caf31cdb1db511120c60955852e00c1f388855/Resources/live-trading/reconciliation/differences-from-modeling.html): fees, slippage, stale fills, and market impact. At this source revision the documentation says market impact is not modeled by default; this is a source-specific disclosure, not a statement about every user's custom LEAN model.
- [Brokerage](https://github.com/QuantConnect/Documentation/blob/72caf31cdb1db511120c60955852e00c1f388855/Resources/live-trading/reconciliation/differences-from-brokerage.html): allocation and fill differences among other practical effects.

The public [reconciliation screenshot](https://github.com/QuantConnect/Documentation/blob/72caf31cdb1db511120c60955852e00c1f388855/single-page/images/reconciliation-4.png)
shows a shared date axis, equity units, visible curve differences and tabs for
equity and out-of-sample backtest evidence. It is illustrative UI evidence with
dates approximately August 2019–March 2020; it does not identify a complete,
investable strategy or provide an audited return. No performance number was
extracted from the pixels. No competitor artwork is copied into Stocksel.

## Adoption matrix

In this table, **supplied pattern** means a finding described in the user's
specification whose current official page could not be independently read.
Effort estimates are engineering estimates, not vendor quotes.

| Product / category | User problem solved | Documented method or supplied pattern | UI pattern | Reported results / evidence type | Limitations | What Stocksel should adopt | Effort / dependencies |
|---|---|---|---|---|---|---|---|
| TradingView / screening and charting platform | Narrow a large universe and revisit a research shortlist | Supplied T1/T2 pattern: saved filters, customizable sortable columns, watchlists | Screen → watchlist → detail; table/chart views | No platform-level investment return is meaningful; current docs blocked | Current entitlements and exact feature behavior unverified; no hands-on testing | Saved screens, persistent watchlist, sortable columns, linked detail that preserves context, CSV export | Small–medium; versioned local preferences, stable ticker IDs, accessible controls |
| Koyfin / research and analytics platform | Reuse a research layout and compare companies | Supplied K1/K2 pattern: dashboard templates, configurable tables, linked stock/chart widgets | Reusable workspace with consistent context | Workflow evidence supplied only; no single platform strategy return | Paid layouts, current templates, pricing and widget limitations unverified | A consistent screener/detail/compare workflow; reuse a saved view | Medium; history API and comparison state. Defer drag/drop/resizable widget builder |
| Seeking Alpha Quant Ratings / ratings service | Understand why a stock ranks highly relative to peers | Supplied S1 pattern: sector-relative factor grades, underlying metrics, rating history | Summary rating with drill-down evidence and change history | Current Quant performance claim unavailable; no figure adopted | Proprietary model, universe, revision behavior and applicable performance disclosure unverified | Name our probability event/horizon; show the model's own signals, source dates, missing data and reasons for low confidence | Medium; honest feature metadata now; prospective score history and sector-relative research later |
| Seeking Alpha Alpha Picks / recommendation portfolio service | Follow a defined stock-selection portfolio | S4 content unavailable; do not substitute Quant Ratings assumptions for this separate service | Selection rationale and portfolio history are requirements to investigate, not observed screens | Current portfolio returns, dates and methodology unknown | No access to current strategy-specific terms, dividends, turnover, costs, taxes or audit evidence | Keep a decision journal and later outcomes; require a portfolio-specific methodology record before comparing returns | Small–medium journal; substantial real-data/provider work for comparable performance |
| QuantConnect / algorithm research and deployment platform | Diagnose why realized or paper results differ from modeled results | Q1 official GitHub source independently verified: same-period OOS/live comparison and data/model/brokerage reconciliation | Aligned equity curves and explicit source of differences; public screenshot inspected | Documented method and illustrative UI; no universal platform return | No live account tested; model defaults are not universal and may omit market impact | Align comparison dates/units, distinguish modeled versus paper fills, retain missing-value gaps, expose fees and state/version provenance | Medium for paper evidence; high for genuine live reconciliation, which remains outside this release |
| Portfolio123 / ranking and portfolio-research platform | Determine whether a ranking separates useful outcomes | Supplied P1 pattern: rank buckets, benchmark-relative and subperiod results, slippage assumptions | Ranking diagnostics next to portfolio results | Method described in supplied findings; current source blocked | Current ranking-test defaults, costs, universe and result definitions unverified | Rank-bucket and calibration diagnostics, cost sensitivity, fixed test boundaries and reproducible evidence exports | Medium; shared eligible universe, forward labels, trial accounting and uncertainty estimates |

These requirements solve concrete tasks independently of competitors' exact
current screen designs. The release keeps Stocksel's Python engine and plain
JavaScript frontend. No proprietary ranking formula, branding or asset is
recreated. UI implementation and actual acceptance results are recorded in
`IMPROVEMENT_PLAN.md` and `VALIDATION_REPORT.md`, rather than assumed from this matrix.

## Portfolio-performance claims: deliberately not comparable yet

Research platforms such as TradingView, Koyfin, QuantConnect and Portfolio123
can host many different strategies. A single return for the platform would be
misleading. Recommendation services require a strategy-specific record. The
historical description in the supplied prompt of hypothetical Seeking Alpha
portfolios with zero transaction costs is **not verified here as the disclosure
for any current claim**.

| Required item | Seeking Alpha Quant performance (S2) | Alpha Picks (S4) |
|---|---|---|
| Exact strategy and portfolio construction | Unknown from accessible current evidence | Unknown from accessible current evidence |
| Start/end dates and current measurement cutoff | Unknown | Unknown |
| Eligible universe and historical membership | Unknown | Unknown |
| Benchmark identity and timing | Unknown | Unknown |
| Dividends / reinvestment convention | Unknown | Unknown |
| Turnover and rebalance/fill assumptions | Unknown | Unknown |
| Transaction costs / spread / slippage | Unknown for this current claim | Unknown for this current claim |
| Subscription costs and taxes | Unknown | Unknown |
| Backtested / forward simulated / real-money | Unknown for this current claim | Unknown for this current claim |
| Independent audit evidence | Not verified | Not verified |
| Headline return used in Stocksel comparison | None | None |

No competitor headline return is an acceptance target. Comparison requires
compatible dates, total-return conventions, eligible universes and realistic
implementation costs. Missing fields above are pending evidence, not zero costs
or an assertion that the service lacks a disclosure.

## Data options: capability, economics and verification

Official pricing/documentation endpoints were requested on 2026-11-05. Each
raised `ProxyError`, so **no current subscription price, redistribution license
or entitlement was verified**. No subscription was purchased. Vendor product
names are candidate sources to evaluate; the table separates what Stocksel's
code does from facts about a current vendor offering.

| Option / official source attempted | Current cost verified? | PIT / delistings / actions in this app | Freshness and coverage in this app | Licensing / dependency and decision |
|---|---|---|---|---|
| [Alpaca data](https://alpaca.markets/data), [market-data API docs](https://docs.alpaca.markets/docs/about-market-data-api) | Unknown; both requests blocked | Quote adapter only; does not reconcile holdings with splits/dividends or supply a PIT research dataset | Code supports `iex` or `sip`; IEX must not be labeled consolidated US coverage. Credentials, entitlements and live transport unverified | Verify user's current subscription, exchange entitlements and intended-use license before use; useful for price monitoring after history is correctly sourced |
| [Sharadar SEP pricing on Nasdaq Data Link](https://data.nasdaq.com/databases/SEP/pricing) | Unknown; request blocked | Code attempts SEP/SF1/SP500/ACTIONS plus DAILY/TICKERS and SFP benchmark; filing dates and membership mapping need vendor-data fixtures. Delisting losses currently use a configurable estimate, not measured settlement | Historical daily research path; live adapter not end-to-end verified | Confirm all required table entitlements and export rights. The adapter's historical flags are intended inputs, not independent vendor certification |
| [Massive pricing](https://massive.com/pricing) (existing adapter uses Polygon endpoints) | Unknown; request blocked | Existing adapter loads provided symbols, adjusted bars and current share counts; no financial statements or historical membership loaded. Delisted coverage unverified | Current shares create biased historical market-cap estimates; no validated live/research integration | Confirm current provider branding/API compatibility and license. Release corrects falsely positive survivorship/PIT flags; do not use this adapter as proof of unbiased research |
| [Tiingo pricing](https://www.tiingo.com/about/pricing) | Unknown; request blocked | Adapter loads adjusted EOD prices but no market caps/PIT fundamentals; survivorship certification remains false | Missing market caps fail the existing size filter; it is not a working full-universe pipeline without additional data | Verify dataset/license and join dated market-cap/membership records first; pricing and delisted coverage unverified |
| [Yahoo exchange/provider information](https://help.yahoo.com/kb/finance-for-web/exchanges-data-providers-yahoo-finance-sln2310.html) via optional yfinance client | Unknown; request blocked; no claim of commercial redistribution rights | Research adapter lacks delisted/PIT fundamentals and estimates historical caps from current shares | Polling adapter applies a conservative 900-second delay assumption and preserves timestamps; this is an app assumption, not verified vendor SLA | Convenience/demo exploration only until coverage/license and raw/adjusted basis are resolved; unsuitable as certification of an unbiased backtest |
| User-licensed CSV exports / canonical CSV adapter | User's actual contract, not inferred | Can represent the required canonical adjusted data and dated events, but flags must be justified by provenance and tests | Historical frequency and missing sessions are explicit in files; no automatic entitlement guarantee | Strong first integration path: versioned dataset, checksums, corporate-action policy and verified usage rights |
| Synthetic provider / local generator | No market-data subscription; compute cost depends on host | Planted signals and known simulated events; never real-market evidence | Offline deterministic daily replay; historical timestamps remain simulated | Keep for engineering/regression work; report actual runtime separately from hypothetical cloud charges |

The provider audit also found and fixed a local Sharadar keyword collision:
`_table("TICKERS", table="SEP")` previously passed `table` twice. Offline tests
now verify the endpoint name and query filter are distinct. This confirms local
request construction, not live authentication, schema compatibility or coverage.

### Economic decision rule

Trading commissions, spread, slippage and impact belong in portfolio returns.
Recurring data/software subscriptions and measured compute belong in a separate
ownership-cost view. Let `F` be annual recurring cost in the reporting currency
and `a` a plausible annual net advantage after trading costs. The arithmetic
minimum capital to cover fixed costs is `F / a` only for `a > 0`; it is unavailable
when the estimated advantage is nonpositive, and is not a recommendation when
its uncertainty interval includes zero. User-entered fee scenarios must be
labeled assumptions, not vendor prices. No assumed USD/ILS FX rate or Israeli
tax rule belongs in this calculation.

## Follow-up evidence needed

1. Revisit the blocked official product pages through an authorized environment;
   record retrieval date, exact applicable disclosure and screenshots where public.
2. Verify prices and entitlements for the actual candidate dataset, including
   historical membership, identifiers, delistings, corporate actions, API limits
   and personal versus commercial/display/export rights.
3. Compare software workflows using task steps and measured response time, then
   compare investment outcomes only with complete strategy-specific evidence.

Access limits did not block offline engineering. They do prevent claiming this
document is a complete, current commercial-product or pricing survey.
