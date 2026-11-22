# Replication of the technical-rule leads on markets never used — pre-registration

Registered **22 November 2026**. It follows the [technical rules study](../technical_rules/RESULTS.md),
which was committed in `98161c4`, and is committed before any international data is run through the
rules. Code:
- Runner: `stocksel/experiments/technical_replication.py`
- Rules: `stocksel/features/technical.py`
- Engine: `stocksel/backtest/rules.py`

All three are unchanged from the first study except for the new runner.

## Why

The first study tested 16 rules on 15 US markets. It found no confirmed edge, but two leads:
1. **Short-term dip-buying in an uptrend** (Bollinger mean reversion with a 200-day filter, BB_MR_TREND;
   Connors RSI(2), RSI2). Alpha was positive in both halves in 10 of 11 and 8 of 11 ETFs. The best
   two cases failed only the deflated Sharpe.
2. **Trend rules as drawdown control** (SMA200, GOLDEN, FABER10). On energy they cut drawdowns
   roughly in half. The random-timing p-values were 0.014–0.034, just missing the pre-registered
   0.01.

Leads found by looking at many results must be tested on data that played no part in finding them.
No new US history exists, so this test uses **other countries**.

## Markets (new data)

There are 16 iShares MSCI country ETFs that have traded since March 1996, so they have full history
before the April 2000 start:
- Asia-Pacific: EWA Australia, EWH Hong Kong, EWJ Japan, EWS Singapore
- Americas: EWC Canada, EWW Mexico
- Europe: EWD Sweden, EWG Germany, EWI Italy, EWK Belgium, EWL Switzerland, EWN Netherlands,
  EWO Austria, EWP Spain, EWQ France, EWU United Kingdom

These prices were never loaded in any earlier study. Their price histories include every company
they held, so there is no survivorship bias. Country ETFs trade in US hours while most of their
stocks trade abroad. That creates short-term price reversals between the ETF and its holdings,
which a dip-buying rule could pick up. It is tradable in the ETF, but spreads were wide in the
early years. Costs are therefore **10 bps per side**, more than three times the US ETF assumption,
with a **2× check (20 bps)**.

Windows are the same as the first study:
- **Discovery:** 3 Apr 2000 – 31 Dec 2012
- **Confirmation:** 2013 – end of data
- **Full:** both together

Execution is the same: signal at the close, trade at the next open, cash at the T-bill rate.

## Hypotheses and decision rules (fixed now)

**H1 — dip-buying alpha replicates.** It is tested separately for BB_MR_TREND and RSI2, each
against each market's buy-and-hold. The *pooled* alpha is the monthly alpha return, beta-adjusted
per market, averaged across the 16 markets. Pooling handles the correlation between markets,
because the test runs on one time series. A rule **replicates** if all of these hold:
- pooled alpha over 2000–2026 has a Newey-West t ≥ 2.0, net of 10 bps;
- the pooled alpha is positive in each half;
- at least 11 of 16 markets have positive full-period alpha;
- the pooled alpha stays positive at 20 bps.

The family **replicates** if both rules do, and **partly replicates** if one does.

**H2 — trend rules as drawdown control replicate.** It is tested for SMA200, GOLDEN and FABER10, each
per market. The per-market criterion is the first study's risk-control test at the 0.05 level:
- maximum drawdown no worse than 2/3 of buy-and-hold's in both halves;
- Sharpe ≥ buy-and-hold's in both halves;
- random-timing drawdown p ≤ 0.05 (500 circular shifts).

On random walks this flag appears in about 1 of 176 trials, so about 0.1 of 16 markets by chance.
A rule **replicates** if it is met in a majority (≥ 9) of the 16 markets, and **partly** if in 5–8.

**H3 — a usable portfolio (practical test).** Rotate capital across the markets that currently have a
dip signal. One portfolio uses BB_MR_TREND and one uses RSI2:
- each active market gets min(1/number active, 25%);
- the rest earns T-bills;
- the benchmark is equal-weight buy-and-hold of all 16 markets;
- costs are 10 bps per side on every weight change.

A portfolio is **usable** if all of these hold:
- full-period Sharpe ≥ the benchmark's Sharpe;
- full-period alpha t ≥ 2;
- alpha > 0 in both halves;
- max drawdown no worse than 2/3 of the benchmark's.

**Thin trading.** In thinly traded markets, closes bounce between bid and ask, which by itself looks
like profitable dip-buying. Several guards are in place:
- execution at the next open, not at the close that triggers the signal;
- 10 bps per side, with a 20 bps check;
- a positive result required in 2013–2026, when these ETFs were liquid;
- a descriptive report of the pooled H1 alpha on the **8 most liquid markets** only, ranked by
  average dollar volume in 2013–2026.

**Also reported:**
- each market's details;
- results at 20 bps;
- an *in-sample reference*: the same rotation on the 9 US sector SPDRs, where the lead was found
  (flagged in-sample, no decision).

**Multiple testing.** There are 2 + 3 + 2 pre-specified hypotheses on new data. The trials (16 × 5
per market + 2 portfolios = 82) are added to the registry, which then holds **379** real-data trials.

**No default of the system changes from this study.** A replicated result would become a
pre-registered candidate for paper trading, as a separate sleeve or an overlay, with its own
holdout: the months after it starts.
