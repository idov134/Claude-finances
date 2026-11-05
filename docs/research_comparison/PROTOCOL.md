# First-release comparison pre-registration

Registered 5 November 2026 before execution. Engineering diagnostics on synthetic
data only; no claim of investable returns, no strategy promotion, no holdout opening.

Budget: one configured synthetic dataset (default seed 7), one purged walk-forward
logistic fit, no hyperparameter search, at most five minutes of compute. Physically
exclude the final 24 months before features/labels; also cap the research end before
2024-01-01 because the original 2024–2025 holdout has already been used.

The eleven predeclared configurations are three monthly portfolios at costs
1×/2×/3×, plus two references:

1. Current configured logistic features, calibration, haircut, buffer and sizing.
2. Equal-weight top N by `(rank(momentum_12_1) - rank(volatility_252)) / 2`,
   using the configured position/sector caps and ex-ante volatility target.
   This is a ranking rule, with no manufactured calibrated probabilities.
3. Equal-weight all eligible names, with the same caps and volatility target.
4. Cash at the supplied synthetic annual risk-free rate divided by twelve.
5. The bundle's cap-weighted synthetic total-return benchmark, frictionless.

All portfolios use the same eligible universe, next-open to next-rebalance-open
label windows, fractional shares, signal-date cost inputs and explicit fee cash
reserve. Missing held returns fail the comparison rather than becoming zero.
There are no intramonth stops in this bounded monthly comparison; it does not
claim to reproduce the separate daily execution engine. The benchmark is a
simulation reference, not a verified investable total-return ETF. Real-data
benchmark verification remains blocked by absent licensed point-in-time data.

Report model Brier/log loss versus constant 0.5, calibration bins, monthly rank IC
and rank quintiles; annualized net return/excess with stationary-bootstrap
uncertainty (500 draws, expected block six months, seed 3); risk, exposure,
concentration, costs, tail losses and fixed subperiods. Risk-matched benchmark
results are ex-post descriptive comparisons, not a tradable timing policy.

Log each planned configuration and every success/failure to the existing trial
registry. Preserve the previous 48-plus-2 known attempts: the old full return
matrix is unavailable, so combined DSR/PBO certification is unavailable. A limited
current-matrix PBO diagnostic must be labeled as such. No winner is selected.

Ownership costs are separate from trading costs. The default illustrative
USD 600/year is a user-editable scenario, not a quoted vendor price. Break-even
capital/bps use linear point-estimate approximations, ignore taxes and varying
market impact at other account sizes, and are unavailable for nonpositive edge.
An excess confidence interval containing zero cannot establish economic viability.
