# Real data + free point-in-time fundamentals — pre-registration

Registered **15 November 2026**, committed before the run (runner:
`stocksel/experiments/real_fundamentals.py`; data: `stocksel/data/build_sec_fundamentals.py`).

## Question

On real S&P 500 data, price and volume signals alone showed **no** out-of-sample skill (AUC 0.494;
see [the first look](../real_firstlook/RESULTS.md)). Do **point-in-time fundamental and earnings
signals** add skill? The synthetic results depended on exactly these signals.

## Data

- Prices: the corrected Yahoo dataset of the first look (v1.1). It is survivor-only, so results
  are an upper bound.
- Fundamentals: SEC EDGAR XBRL company facts. For each quarter the **first-filed** value is used,
  and it becomes available only from its **filing date**. Quarterly flows come from 3-month facts or
  year-to-date differences; Q4 = FY − 9M. Fields: total assets, book equity, gross profit, net
  income, revenue, diluted EPS. XBRL starts in 2009, and all filers report from 2011.
- Earnings surprise: there is no analyst consensus in free data, so **SUE uses the seasonal random
  walk** (EPS change vs the same quarter last year, scaled by its own volatility). It is dated at the
  10-Q/10-K filing, which is later than the press release, so PEAD measured this way is conservative.
- There is no analyst-revision signal.
- Data-quality fixes made while building the data, before any result was seen, each covered by a test:
  1. Trailing-year sums need four **consecutive** quarters, because real filings have gaps.
  2. The seasonal SUE compares a quarter only with the **same quarter** a year earlier.
  3. **Late comparatives:** an older quarter first reported after a newer one does not overwrite it
     in the point-in-time join. Early XBRL filings restate prior years as comparatives.
  4. **Re-registered companies:** a company that moved to a new SEC registrant (holding-company
     reorganization or redomicile; for example Exxon in 2026, BlackRock in 2024, Disney in 2019) keeps its
     earlier history. The predecessor is merged only if the successor's comparative net income matches
     it: at least 2 quarters, and ≥ 75% within 1%. Candidates that fail are not used. These include old
     vs new News Corp, and EI du Pont vs DowDuPont, which really are different companies, and Alphabet vs
     Google Inc. and Eaton, rejected conservatively.

  None of the fixes changes the synthetic results; this was checked.

### Data checked before the run (properties of the inputs, not results)

- 716 tickers, 684 mapped to an SEC registrant, 660 with US-GAAP facts; 42,747 company-quarters;
  first filing 2009-04-15; median filing lag 37 days. 12 predecessor registrants were verified and
  merged, and 6 were rejected (`sec_info.json`).
- Share of eligible index members with a value, by year:

| | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 |
|---|---|---|---|---|---|---|---|---|
| value_bm | 0.69 | 0.83 | 0.91 | 0.90 | 0.84 | 0.90 | 0.92 | 0.91 |
| value_ep / quality_roe | 0.43 / 0.41 | 0.81 / 0.80 | 0.90 / 0.89 | 0.91 / 0.88 | 0.86 / 0.82 | 0.94 / 0.89 | 0.97 / 0.91 | 0.96 / 0.90 |
| quality_gpa | 0.22 | 0.47 | 0.52 | 0.52 | 0.54 | 0.59 | 0.59 | 0.58 |
| sue / pead | 0.00 | 0.21 | 0.62 | 0.73 | 0.66 | 0.69 | 0.86 | 0.82 |

  Missing values are scored as neutral (0) after cross-sectional ranking. Gross profit is missing for
  banks, insurers and utilities, which do not report it.
- Spot checks: Apple, Microsoft, Exxon and BlackRock values match their reported financials.

## Configurations (no search)

Model: the effective default, an online-gated ensemble of {logit, ridge, enet}. The gate admitted
only these on the first look, as on all synthetic worlds. Walk-forward, costs, sizing: defaults.

| ID | Features |
|---|---|
| F0 | `price_only`: mom_12_1, mom_6_1, rev_1m, illiq, vol_252, idio_vol, size |
| F1 | `price_fund`: F0 + value_bm, value_ep, quality_gpa, quality_roe, sue, pead |

Mandates: the default **absolute** (primary) and **relative** (secondary). The relative mandate
removes cash and beta effects, so it shows selection more directly.

## Windows (fixed now)

- **Primary: Jan 2014 – Sep 2024.** By then the walk-forward training window contains several
  years of XBRL-era fundamentals.
- Full out-of-sample 2005–2024, reported for completeness. Before 2010 the fundamental features are
  empty (neutral).
- **Nov 2024 – Sep 2026** is reported but **is not a clean holdout**: it was viewed in the first
  look.

## Decision (on the primary window)

"Fundamentals add stock-selection skill" requires all of:
1. F1 rank IC t-statistic ≥ **2.0** and AUC > **0.505**;
2. paired net difference F1 − F0 (absolute mandate) > 0 with stationary-bootstrap P ≥ **0.90**;
3. F1 **alpha DSR** ≥ **0.95**, charged with every real-data trial (first look v1 + v1.1: 52, plus
   this study).

Reported outcome: **skill established** (1+2+3); **signal present, not significant after testing**
(1, not 3); or **no skill**. Whatever the outcome, defaults do not change from this study alone:
it has no clean holdout. A positive result would justify paid survivorship-free data and a new
pre-registered window.

All configurations go to `registry.jsonl` in this folder. Single-signal IC for every fundamental
signal on real data is reported as well.
