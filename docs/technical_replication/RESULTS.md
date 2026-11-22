# Replication on 16 country ETFs — results

Protocol: [PROTOCOL.md](PROTOCOL.md), committed and pushed in `2e3fe57` before the country data was
downloaded. It was run once on 22 November 2026. Outputs:
- [`results.json`](results.json);
- [`trials.csv`](trials.csv): 80 market × rule trials;
- [`registry.jsonl`](registry.jsonl) and [`returns.csv`](returns.csv).

The registry now holds 379 real-data trials.

## Outcome: **neither lead replicates**

| Hypothesis | Verdict |
|---|---|
| H1 — dip-buying alpha (BB_MR_TREND, RSI2) | **does not replicate** |
| H2 — trend rules as drawdown control (SMA200, GOLDEN, FABER10) | **does not replicate** (1, 0, 2 of 16 markets; 9 needed) |
| H3 — dip-signal rotation as a usable portfolio | **not usable** (both versions) |

The two leads from the [US study](../technical_rules/RESULTS.md) were selected from 240 trials, and
on markets that played no part in finding them, both shrink or vanish.

## H1 — dip-buying (pooled alpha vs each market's buy-and-hold, 10 bps per side)

| Rule | 2000–26 | 2000–12 | 2013–26 | markets positive | at 20 bps | 8 most liquid only |
|---|---|---|---|---|---|---|
| BB_MR_TREND | +0.8% (t 1.31) | +0.9% (1.03) | +0.8% (0.85) | 13/16 | +0.4% (0.56) | +0.4% (0.62) |
| RSI2 | +0.5% (t 1.04) | +1.7% (2.01) | -0.5% (-0.84) | 11/16 | -0.8% (-1.56) | +0.0% (0.08) |

- **Bollinger-in-an-uptrend keeps its sign** in both halves and in 13 of 16 countries, but at
  **+0.8% a year** with t = 1.3. In the US sectors where it was found, the average was +2% a year.
  At 20 bps it is +0.4%. It is too small to trade.
- **Connors RSI(2)** worked in 2000–12 (t = 2.0) and turned negative in 2013–26. This is the same
  decay seen in the US stock baskets.

## H2 — trend rules: real drawdown reduction, but not a free lunch

The 200-day rule on each country (full period; drawdown p from 500 random-timing shifts):

| Market | max drawdown: buy-and-hold → SMA200 | CAGR | Sharpe 2000–12 | Sharpe 2013–26 | timing p (drawdown) |
|---|---|---|---|---|---|
| EWA | -65.6% → -32.9% | 8.1% → 5.6% | 0.51 → 0.52 | 0.25 → 0.09 | 0.030 |
| EWC | -61.1% → -38.3% | 7.2% → 4.9% | 0.34 → 0.31 | 0.38 → 0.22 | 0.014 |
| EWD | -72.5% → -37.7% | 5.3% → 3.8% | 0.19 → 0.25 | 0.35 → 0.13 | 0.016 |
| EWG | -64.5% → -32.7% | 4.1% → 6.0% | 0.15 → 0.46 | 0.30 → 0.26 | 0.012 |
| EWH | -62.4% → -29.2% | 5.0% → 4.5% | 0.28 → 0.37 | 0.22 → 0.09 | 0.014 |
| EWI | -70.3% → -43.5% | 4.3% → 3.2% | 0.05 → -0.02 | 0.41 → 0.31 | 0.020 |
| EWJ | -58.3% → -43.3% | 2.7% → 1.3% | -0.14 → -0.12 | 0.45 → 0.16 | 0.208 |
| EWK | -74.1% → -30.3% | 5.8% → 5.5% | 0.22 → 0.42 | 0.40 → 0.23 | 0.012 |
| EWL | -51.9% → -25.9% | 7.2% → 4.5% | 0.29 → 0.28 | 0.52 → 0.27 | 0.012 |
| EWN | -65.4% → -37.7% | 6.0% → 5.4% | 0.11 → 0.16 | 0.51 → 0.44 | 0.050 |
| EWO | -76.3% → -36.9% | 8.9% → 10.8% | 0.38 → 0.67 | 0.41 → 0.58 | 0.022 |
| EWP | -63.3% → -43.3% | 6.3% → 5.6% | 0.22 → 0.21 | 0.39 → 0.43 | 0.032 |
| EWQ | -61.8% → -34.8% | 4.5% → 4.5% | 0.12 → 0.34 | 0.36 → 0.17 | 0.014 |
| EWS | -64.2% → -27.1% | 6.5% → 7.6% | 0.39 → 0.64 | 0.22 → 0.24 | 0.010 |
| EWU | -64.1% → -36.3% | 4.2% → 3.9% | 0.13 → 0.41 | 0.31 → 0.06 | 0.050 |
| EWW | -64.9% → -62.9% | 7.6% → 3.6% | 0.53 → 0.51 | 0.15 → -0.22 | 0.489 |

**The timing part of the lead holds up.** In **14 of 16** countries the 200-day rule's drawdown is
milder than random timing with the same time in cash (p ≤ 0.05). The median drawdown is 54% of
buy-and-hold's: a −65% crash becomes about −35%.

**The free-lunch part does not.** The pre-registered criterion also asked for no loss of
risk-adjusted return in *both* halves:
- In 2013–26 the rule's Sharpe fell in 12 of 16 countries. Whipsaws in V-shaped recoveries cost
  more than the crashes it avoided.
- Over the full period it gave up a median **0.6% a year** of CAGR.

Only 1 country (SMA200), 0 (golden cross) and 2 (Faber) meet the full criterion.

**A trend filter is insurance with a premium, not an edge.** That is useful to know for a
mandate with a 20% drawdown limit, but it does not beat the market.

## H3 — rotating capital across markets with a dip signal

| Portfolio | CAGR vs equal-weight buy-and-hold | Sharpe | max DD | alpha (t) | alpha 2000–12 / 2013–26 | invested |
|---|---|---|---|---|---|---|
| BB_MR_TREND rotation, 16 countries (new data) | 3.8% vs 6.9% | 0.26 vs 0.35 | -32.5% vs -62.7% | +0.7% (0.45) | +1.2% / -0.1% | 28% |
| RSI2 rotation, 16 countries (new data) | 1.1% vs 6.9% | -0.08 vs 0.35 | -36.7% vs -62.7% | -1.8% (-1.43) | +1.4% / -4.7% | 26% |
| BB_MR_TREND rotation, 9 US sectors (in-sample, where the lead was found) | 6.6% vs 9.2% | 0.74 vs 0.54 | -26.6% vs -52.6% | +3.0% (2.56) | +2.2% / +3.0% | 23% |
| RSI2 rotation, 9 US sectors (in-sample, where the lead was found) | 6.1% vs 9.2% | 0.73 vs 0.54 | -13.2% vs -52.6% | +3.0% (3.02) | +3.8% / +2.1% | 22% |

On the US sectors, where the lead was found, both rotations look good: Sharpe 0.73–0.74 vs 0.54,
alpha t 2.6–3.0, about half the drawdown. On new markets they lose:
- **BB_MR_TREND:** Sharpe 0.26 vs 0.35, alpha t 0.45.
- **RSI2:** Sharpe −0.08, with −4.7% a year of alpha in 2013–26.

This is what a backtest selected from many looks like when it meets new data.

## What the two studies together say

1. **The textbook technical rules do not beat buy-and-hold.** That covers moving averages (150/200,
   golden cross, Faber), RSI (14 and 2), Bollinger Bands, MACD, Minervini, 52-week highs, turn of
   month, Halloween, VIX and breadth. It holds in the US market, in tech, in energy, among the
   largest companies, and in 16 other countries, net of realistic costs.
2. **The best-looking results were selection effects.** The leads were the strongest of 240 trials.
   On new markets the dip-buying alpha fell from about 2% to under 1% a year, and the rotation
   portfolio's Sharpe fell from 0.74 to 0.26.
3. **Something real, but not alpha: trend filters cut crash depth.** In 14 of 16 new markets the
   200-day rule's drawdowns were milder than random timing. The price was about 0.5–1% a year of
   return and no better Sharpe. If the 20% drawdown limit matters more than return, it is a known,
   honest tool with a known cost. It is not a way to beat the market.
4. **Sector focus is a bet, not an edge.** Tech and energy swapped places between 2000–12 and
   2013–26, and over 26 years they ended within 0.6% a year of the S&P 500.

## Caveats

- The 16 countries are correlated with each other and with the US: global crashes are shared.
  Pooled tests account for this; the "n of 16" counts do not.
- Country ETFs trade in US hours while their stocks trade abroad, and early spreads were wide.
  Neither helped the rules here.
- Taxes are not modeled. They would make every high-turnover rule look worse.
