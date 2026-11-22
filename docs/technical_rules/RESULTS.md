# Simple technical rules by market and segment — results

Protocol: [PROTOCOL.md](PROTOCOL.md), committed and pushed in `4ffa4a9` before the single run on
22 November 2026. Outputs:
- [`trials.csv`](trials.csv): all 240 trials with every metric, per window;
- [`results.json`](results.json): decisions, global statistics, cross-sectional IC, buy-and-hold by segment;
- [`registry.jsonl`](registry.jsonl) and [`returns.csv`](returns.csv): the trial registry.

Raw prices are not committed.

## Outcome

| Pre-registered question | Result | Random-walk reference (5 worlds) |
|---|---|---|
| **Edge confirmed** (beats its own buy-and-hold in both halves, survives DSR, 2× costs, random timing, top-days removal) | **0 of 240** | 0 of 1,200 |
| Alpha t ≥ 2 in both halves (before the DSR) | 2 ETF trials (XLY + RSI2, XLI + Bollinger-in-uptrend) | 0 of 1,200 |
| **Risk control** (drawdown ≤ 2/3 of buy-and-hold and Sharpe ≥ buy-and-hold in both halves, timing p ≤ 0.01) | **1 of 176** (XLU + VIX30) | 1 of 880 |
| Persistence of gross alphas between halves (Spearman) | ETF **0.24**; stocks 0.19 | ETF −0.13 to 0.17; stocks −0.04 to 0.33 |
| Promising cross-sectional technical feature | **0 of 24** | 0 of 120 |

**None of the 16 textbook rules beats buy-and-hold in any of the 15 markets under the pre-registered
test.** That includes moving averages (150/200-day, golden cross, Faber), RSI, Bollinger Bands,
MACD, Minervini, 52-week highs, turn of month, Halloween, VIX and breadth. Three things are worth
knowing beyond that:

1. **Short-term dip-buying is the one family with a consistent sign.** Bollinger mean reversion
   within an uptrend has positive alpha in both halves in **10 of 11** ETFs; Connors RSI(2) in
   **8 of 11**. Two individual cases pass every check except the multiple-testing deflation (table
   below). But these rules are invested only ~11% of the time. On their own they return about half
   of buy-and-hold (4–5% vs 8% a year over 2000–2026). Their value is as an *addition* to a
   portfolio, not a replacement for holding the market.
2. **Trend rules control drawdowns in energy, but not significantly enough to pass.** On XLE, the
   golden cross, Faber's 10-month rule and the 200-day rule cut buy-and-hold's drawdowns of −57%
   (2000–12) and −71% (2013–26) to about −30% and −30% to −43%. Their returns were similar (8.7–9.4%
   vs 9.1%, then 7.6–9.3% vs 7.5%). The random-timing drawdown p-values were 0.014–0.034, so they
   missed the pre-registered 0.01. Seven ETF trials pass at a looser 0.05, against about one
   expected by chance. This is a lead, not a result, and it is not pre-registered.
3. **What worked in 2000–2012 mostly stopped working.** The best rule of the discovery period in
   each focus market ranked 10th, 5th, 13th, 6th, 14th, 13th, 12th and 4th of 16 in 2013–2026. Most
   of these picks were calendar rules:
   - Halloween: +3.6% a year of alpha in 2000–12 → −0.9% in 2013–26.
   - Turn of the month: +1.0% → −0.4%.
   - Faber's 10-month rule on SPY: +4.4% (t 2.1) → −0.3%.

   Choosing the best backtest would not have helped.

## The two near-misses

| Market + rule | Alpha 2000–12 (t) | Alpha 2013–26 (t) | at 2× costs | random timing p | top 0.1% days removed | **alpha DSR** | CAGR vs buy-and-hold | max DD 2000–12 / 2013–26 (vs buy-and-hold) | invested |
|---|---|---|---|---|---|---|---|---|---|
| XLY RSI2 | +3.2% (2.1) | +5.8% (3.9) | +3.9% | 0.002 | yes | **0.27** | 7.0% vs 9.2% | -14.2% / -11.5% vs -59.0% / -39.5% | 11% |
| XLI BB_MR_TREND | +2.7% (2.5) | +6.5% (4.3) | +4.6% | 0.002 | yes | **0.65** | 7.0% vs 8.5% | -7.6% / -9.9% vs -62.0% / -39.7% | 11% |

Both are significant in each half, survive double costs, and beat 500 random-timing versions of
themselves. They fail because they are the best 2 of **297 real-data trials**: the deflated Sharpe
(0.27, 0.65) does not reach 0.95. Neither is in the focus markets (tech, energy, SPY, QQQ).

## The dip-buying family vs the rest (11 ETFs, net of costs)

| Rule | mean alpha 2000–12 | mean alpha 2013–26 | positive in both halves | time invested | CAGR 2000–26 | buy-and-hold |
|---|---|---|---|---|---|---|
| BB_MR_TREND | +1.3% | +2.1% | 10/11 | 12% | 4.5% | 8.0% |
| RSI2 | +2.3% | +1.3% | 8/11 | 11% | 4.2% | 8.0% |
| BB_MR | +1.0% | +2.7% | 7/11 | 21% | 5.1% | 8.0% |
| RSI14 | -0.2% | +1.5% | 4/11 | 40% | 5.3% | 8.0% |
| FABER10 | +2.2% | +0.5% | 6/11 | 72% | 7.0% | 8.0% |
| GOLDEN | +2.1% | -0.7% | 3/11 | 72% | 6.9% | 8.0% |
| SMA200 | -0.3% | -0.5% | 4/11 | 73% | 5.4% | 8.0% |
| SMA150 | -0.6% | -0.7% | 2/11 | 71% | 5.0% | 8.0% |
| TEMPLATE | -0.8% | -1.0% | 2/11 | 18% | 2.1% | 8.0% |
| HIGH52 | -0.8% | -1.9% | 0/11 | 31% | 2.1% | 8.0% |
| BB_BREAK | -2.3% | -0.6% | 0/11 | 32% | 1.9% | 8.0% |
| MACD | -2.9% | +0.1% | 0/11 | 51% | 3.5% | 8.0% |
| TOM | +1.0% | -0.4% | 2/11 | 19% | 3.6% | 8.0% |
| HALLOWEEN | +3.6% | -0.9% | 4/11 | 48% | 6.5% | 8.0% |
| VIX30 | -1.5% | -1.1% | 1/11 | 91% | 5.6% | 8.0% |
| BREADTH | -2.1% | -2.0% | 0/11 | 76% | 3.4% | 8.0% |

The ranking of rule families is consistent: dip-buying positive, breakouts and MACD negative. That is
why the persistence statistic (0.24) is above the random-walk range. For a trend follower, the
sobering line is SMA150/SMA200: about 5% a year against 8%.

## Focus markets: SPY, QQQ, tech (XLK), energy (XLE)

**SPY** — buy-and-hold: 1.6%/yr, max DD -55.4% (2000–12); 14.0%/yr, max DD -32.0% (2013–26)

| Rule | 2000–12: CAGR / max DD | α (t) | 2013–26: CAGR / max DD | α (t) | time invested |
|---|---|---|---|---|---|
| SMA150 | 2.3% / -26.0% | +0.5% (0.2) | 10.0% / -20.6% | +0.8% (0.5) | 73% |
| SMA200 | 2.3% / -24.5% | +0.5% (0.2) | 10.3% / -20.0% | +1.3% (0.8) | 74% |
| GOLDEN | 6.4% / -17.0% | +4.3% (2.2) | 9.6% / -32.0% | -0.0% (-0.0) | 74% |
| FABER10 | 6.4% / -20.4% | +4.4% (2.1) | 8.4% / -25.3% | -0.3% (-0.1) | 75% |
| RSI14 | 1.2% / -55.4% | -0.5% (-0.2) | 9.0% / -27.0% | +1.1% (0.5) | 37% |
| RSI2 | 4.3% / -14.2% | +2.2% (1.7) | 4.5% / -11.4% | +1.8% (1.4) | 10% |
| BB_MR | 1.1% / -35.9% | -0.6% (-0.3) | 7.6% / -27.0% | +2.3% (0.9) | 20% |
| BB_MR_TREND | 3.0% / -10.3% | +0.9% (1.1) | 5.4% / -27.0% | +0.9% (0.4) | 11% |
| MACD | 2.0% / -24.4% | +0.2% (0.1) | 6.6% / -16.0% | -0.8% (-0.5) | 50% |
| HALLOWEEN | 5.0% / -35.0% | +3.0% (1.4) | 6.8% / -32.0% | -2.1% (-1.0) | 48% |

**QQQ** — buy-and-hold: -3.2%/yr, max DD -81.5% (2000–12); 18.6%/yr, max DD -36.7% (2013–26)

| Rule | 2000–12: CAGR / max DD | α (t) | 2013–26: CAGR / max DD | α (t) | time invested |
|---|---|---|---|---|---|
| SMA150 | -2.4% / -63.1% | -2.5% (-0.5) | 17.2% / -22.4% | +4.2% (1.9) | 73% |
| SMA200 | 0.5% / -45.6% | +0.1% (0.0) | 14.3% / -26.5% | +0.6% (0.3) | 74% |
| GOLDEN | 5.3% / -30.4% | +4.6% (1.4) | 16.0% / -27.6% | +1.9% (0.7) | 74% |
| FABER10 | 2.6% / -34.8% | +1.9% (0.5) | 14.6% / -27.6% | +2.6% (1.0) | 72% |
| RSI14 | -3.4% / -75.0% | -1.4% (-0.3) | 11.0% / -30.7% | +2.0% (0.7) | 35% |
| RSI2 | 4.4% / -16.0% | +2.6% (1.7) | 5.1% / -11.9% | +2.2% (1.5) | 12% |
| BB_MR | 3.1% / -42.9% | +2.9% (0.8) | 6.2% / -21.1% | +1.5% (0.6) | 21% |
| BB_MR_TREND | 4.3% / -16.0% | +2.4% (1.9) | 4.6% / -21.1% | +0.6% (0.3) | 12% |
| MACD | -4.7% / -67.6% | -4.0% (-1.1) | 7.9% / -25.4% | -2.2% (-1.0) | 51% |
| HALLOWEEN | 0.1% / -60.4% | +0.7% (0.2) | 7.3% / -27.6% | -2.7% (-1.1) | 48% |

**XLK** — buy-and-hold: -4.4%/yr, max DD -81.0% (2000–12); 20.1%/yr, max DD -35.5% (2013–26)

| Rule | 2000–12: CAGR / max DD | α (t) | 2013–26: CAGR / max DD | α (t) | time invested |
|---|---|---|---|---|---|
| SMA150 | -1.2% / -58.2% | -1.4% (-0.3) | 13.7% / -36.3% | -0.9% (-0.4) | 72% |
| SMA200 | -0.8% / -46.0% | -1.0% (-0.3) | 17.6% / -25.4% | +2.5% (1.3) | 73% |
| GOLDEN | 3.0% / -28.0% | +2.6% (0.9) | 15.7% / -30.7% | +0.1% (0.1) | 72% |
| FABER10 | 0.4% / -40.8% | +0.3% (0.1) | 14.8% / -30.7% | +0.5% (0.2) | 73% |
| RSI14 | -5.4% / -75.2% | -2.7% (-0.8) | 11.0% / -29.9% | +2.4% (0.8) | 35% |
| RSI2 | 2.5% / -14.6% | +0.7% (0.5) | 4.9% / -12.9% | +2.0% (1.3) | 10% |
| BB_MR | 1.9% / -42.0% | +1.8% (0.6) | 8.2% / -24.4% | +3.5% (1.2) | 21% |
| BB_MR_TREND | 2.6% / -11.4% | +0.7% (0.7) | 7.3% / -24.4% | +3.6% (1.3) | 11% |
| MACD | -6.0% / -68.7% | -5.1% (-1.8) | 9.2% / -33.5% | -1.9% (-0.8) | 51% |
| HALLOWEEN | 0.1% / -50.1% | +1.4% (0.5) | 7.6% / -30.7% | -2.5% (-1.0) | 48% |

**XLE** — buy-and-hold: 9.1%/yr, max DD -56.9% (2000–12); 7.5%/yr, max DD -70.8% (2013–26)

| Rule | 2000–12: CAGR / max DD | α (t) | 2013–26: CAGR / max DD | α (t) | time invested |
|---|---|---|---|---|---|
| SMA150 | 8.1% / -26.8% | +2.1% (0.6) | 4.3% / -50.0% | -0.1% (-0.0) | 66% |
| SMA200 | 7.9% / -28.9% | +2.0% (0.6) | 7.6% / -42.5% | +2.8% (0.8) | 67% |
| GOLDEN | 9.4% / -32.0% | +2.7% (0.9) | 8.7% / -29.6% | +3.9% (1.2) | 68% |
| FABER10 | 8.7% / -29.9% | +2.4% (0.7) | 9.3% / -34.9% | +4.4% (1.4) | 67% |
| RSI14 | 7.6% / -46.0% | +2.5% (0.8) | 5.0% / -64.8% | +0.2% (0.0) | 38% |
| RSI2 | 7.4% / -18.3% | +4.7% (2.6) | 1.5% / -24.0% | -0.7% (-0.4) | 11% |
| BB_MR | 4.8% / -34.2% | +1.5% (0.6) | 4.5% / -60.3% | +0.8% (0.2) | 23% |
| BB_MR_TREND | 4.7% / -15.7% | +2.2% (1.2) | 3.9% / -17.1% | +1.7% (1.1) | 11% |
| MACD | 1.3% / -47.7% | -3.7% (-1.4) | 9.0% / -29.2% | +3.7% (1.0) | 51% |
| HALLOWEEN | 11.6% / -26.7% | +6.4% (2.3) | 6.7% / -60.2% | +1.1% (0.3) | 48% |

## Does focusing on a sector or on large caps beat the index? (buy-and-hold)

| Market | 2000–12 | 2013–26 | 2000–26 | max DD |
|---|---|---|---|---|
| SPY | 1.6% | 14.0% | 7.9% | -55.4% |
| QQQ | -3.2% | 18.6% | 7.6% | -81.5% |
| XLK | -4.4% | 20.1% | 7.7% | -81.0% |
| XLE | 9.1% | 7.5% | 8.3% | -70.8% |
| XLF | -0.9% | 12.6% | 5.9% | -83.1% |
| XLV | 3.6% | 13.0% | 8.4% | -40.5% |
| XLI | 4.0% | 12.7% | 8.5% | -62.0% |
| XLY | 5.1% | 13.2% | 9.2% | -59.0% |
| XLP | 6.2% | 9.8% | 8.1% | -35.7% |
| XLU | 6.1% | 10.1% | 8.2% | -52.3% |
| XLB | 6.6% | 9.2% | 7.9% | -59.7% |
| stocks ALL (survivors, equal weight) | 11.6% | 13.6% | 12.6% | -56.0% |
| stocks TOP100 (survivors, equal weight) | 5.0% | 14.2% | 9.7% | -55.1% |
| stocks TECH (survivors, equal weight) | 1.0% | 19.9% | 10.5% | -75.3% |
| stocks ENERGY (survivors, equal weight) | 24.7% | 7.8% | 15.6% | -80.0% |

- **Sector leadership reversed between the halves.** Energy was the best of the eleven in 2000–12
  and nearly the worst after. Tech was the worst in 2000–12 (−4.4% a year, −81% drawdown) and the
  best after (+20.1%).
- **Over 26 years tech, energy and the S&P 500 ended within 0.6% a year of each other.** Choosing a
  sector is a bet on which decade comes next, not an edge.
- **The stock rows show the survivorship bias in numbers.** Equal-weighting *today's* energy
  companies back to 2000 gives 24.7% a year for 2000–12; the investable energy sector (XLE) returned
  9.1%. All S&P stocks with data: 11.6% vs SPY's 1.6%. The bias fades in 2013–26, where coverage is
  85%: there, stock baskets and ETFs agree within about 1% a year.

## Stock baskets (survivor-biased)

| Basket | buy-and-hold 2000–12 | 2013–26 | rules with t ≥ 2, 2000–12 | 2013–26 | best rule 2000–12 (t) → t in 2013–26 |
|---|---|---|---|---|---|
| ALL | 11.6% | 13.6% | 3 | 0 | BB_MR_TREND (3.9) → 0.1 |
| TOP100 | 5.0% | 14.2% | 3 | 0 | BB_MR_TREND (3.9) → 1.4 |
| TECH | 1.0% | 19.9% | 1 | 0 | RSI2 (2.1) → 0.2 |
| ENERGY | 24.7% | 7.8% | 3 | 0 | RSI2 (2.8) → -0.3 |

**Ten rules were significant in 2000–12 and none in 2013–26.** The pattern is what the protocol
predicted for survivor-only data:
- dip-buying looks strong where most failed companies are missing (2000–12 coverage 57%);
- it disappears where the data is nearly complete (85%).

## Cross-sectional technical signals (monthly rank IC with next month's return)

- None of the 24 feature × basket combinations is significant in both halves.
- **Short-term reversal:** low RSI(14) or Bollinger %b → higher next-month return. It was significant
  in 2000–12 in the full basket (t = −2.7 and −3.2) and in TOP100 (%b t = −2.1). It weakened to
  t ≈ −1 in 2013–26.
- Distance from the 150/200-day averages and proximity to the 52-week high show nothing in any basket.

## What this means

- **The simple rules recommended online do not beat buy-and-hold** in the US market, in tech, in energy
  or among large caps. This holds once costs, a second period and the number of rules tried are
  accounted for.
- **Each looked good somewhere in 2000–2012,** a period with two crashes that flatter anything that
  goes to cash. Most of that faded in 2013–2026.
- **One family is consistently positive: short-term dip-buying in sector ETFs.** It is small, mostly
  in cash, and did not survive the multiple-testing deflation. The right next step is to test it,
  and the energy trend-drawdown lead, **on markets they have never seen**. That test is the
  [replication study](../technical_replication/RESULTS.md).
- **No default of the system changes.** No real money.

## Caveats

- **Stock arm:**
  - survivor-only data;
  - current Yahoo sectors;
  - approximate market caps before 2015.
- **ETF arm:** about 26 years and 11 correlated markets, so a family result is less independent than
  "10 of 11" suggests.
- **Costs** are 3 bps per side for ETFs and 10 for stocks, with taxes ignored. Dip-buying makes about
  10–15 round trips a year per market.
