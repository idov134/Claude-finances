# Simple technical rules by market and segment — pre-registration

Registered **22 November 2026** and committed before the run. Code:
- Runner: `stocksel/experiments/technical_rules.py`
- Rules and indicators: `stocksel/features/technical.py`
- Engine: `stocksel/backtest/rules.py`
- ETF data and sectors: `stocksel/data/build_etf_csv.py`

The tests in `tests/test_technical.py` cover the indicator formulas, execution at the next open,
costs, and **no look-ahead**: for each of the 16 rules, changing future prices leaves every past
position unchanged.

## Question

The ML model has no skill on real data, whether it uses price signals or fundamentals. Do **simple,
widely recommended technical rules** work instead? These are moving averages (including the
150-day), RSI, Bollinger Bands, MACD, breakouts, calendar effects and regime filters. We ask this:
- in the broad market,
- in **technology** and **energy**,
- among the **largest companies**.

"Works" means a better risk-adjusted return than holding the same market, net of costs, in data
the rule was not chosen on.

## Design

**ETF arm (primary; no survivorship bias).** An ETF's price history contains every company it ever
held.
- Focus markets: SPY (S&P 500), QQQ (Nasdaq-100, large-cap growth/tech), XLK (S&P 500 technology), XLE (S&P 500 energy).
- Controls: the other seven sector SPDRs (XLF, XLV, XLI, XLY, XLP, XLU, XLB). They show whether
  tech and energy are special, or whether a result is just one lucky market out of eleven.

**Stock arm (secondary; survivor-only data).** Equal-weight baskets of S&P 500 members:
- Point-in-time membership; at least 300 days of price history required.
- Each stock is held only while the rule is long on it, with 1/N of the basket's capital.
- Baskets:

| Basket | Definition | Average size 2000–2012 / 2013–2026 |
|---|---|---|
| ALL | every member with data | 283 / 425 |
| TOP100 | the 100 largest by market cap, re-ranked each month end | 100 / 100 |
| TECH | Yahoo sector "Technology" | 34 / 58 |
| ENERGY | Yahoo sector "Energy" | 17 / 21 |

Price coverage of index members is 57% in 2000–2012 and 85% in 2013–2026. Most failed and
acquired companies are missing, especially in the early half.

**Every rule is compared with buy-and-hold of the same market or basket.** Both use the same engine:
- The signal is computed at the close; trades execute at the **next open**.
- Returns are open-to-open and include dividends.
- Cash earns the 13-week T-bill rate.
- Costs per side: **ETFs 3 bps, stocks 10 bps**, with a 2× sensitivity check.

**Windows:**
- **Discovery:** 3 Apr 2000 – 31 Dec 2012. It contains two bear markets. April 2000 is the first
  month in which every indicator is defined for every ETF.
- **Confirmation:** 1 Jan 2013 – end of data (ETFs 19 Nov 2026, stocks 5 Nov 2026).
- **Full:** both together.

**16 rules × 15 markets = 240 trials. Every parameter is the textbook one; nothing is searched.**

| Rule | Long when | Source of the thresholds |
|---|---|---|
| SMA150 | close > 150-day SMA | Weinstein (1988), 30-week MA |
| SMA200 | close > 200-day SMA | classic long-term trend line |
| GOLDEN | 50-day SMA > 200-day SMA | "golden cross" |
| FABER10 | month-end close > 10-month SMA, held for the next month | Faber (2007) |
| MACD | MACD(12, 26) > signal(9) | Appel |
| TEMPLATE | Minervini's trend template (price above rising 150/200 SMA, 50 > 150 > 200, ≥ 30% above the 52-week low, within 25% of the 52-week high; no RS rank) | Minervini (2013) |
| RSI14 | from RSI(14) < 30 until RSI(14) > 70 | Wilder (1978) |
| RSI2 | from (close > SMA200 and RSI(2) < 10) until close > 5-day SMA | Connors & Alvarez (2008) |
| BB_MR | from close < lower Bollinger band (20, 2σ) until close > middle band | Bollinger |
| BB_MR_TREND | BB_MR with entries only while close > SMA200 | Bollinger + trend filter |
| BB_BREAK | from close > upper band until close < middle band | Bollinger breakout |
| HIGH52 | from a new 52-week closing high until close < 50-day SMA | 52-week-high breakout |
| TOM | last trading day of the month through the 3rd trading day of the next | Ariel (1987); Lakonishok & Smidt (1988) |
| HALLOWEEN | November–April; cash May–October | Bouman & Jacobsen (2002) |
| VIX30 | VIX closes below 30 | common "high fear" threshold |
| BREADTH | at least 5 of the 9 sector SPDRs close above their 150-day SMA | sector breadth (Weinstein-style) |

## Metrics

- **Alpha:** Jensen's alpha against the market's own buy-and-hold, from monthly returns, with
  Newey-West t-statistics (3 lags). Net of costs; gross is also reported. A positive alpha means
  better return per unit of market exposure: the rule could be scaled to buy-and-hold's risk and
  beat it.
- **Also reported:** CAGR, Sharpe, daily maximum drawdown, exposure, turnover, trades per year,
  cost drag, and the comparison with SPY.
- **Alpha DSR** over the full window, charged with **every real-data trial**: 57 earlier + 240
  here = **297**.
- **Random-timing test (circular shift):** the rule's own position series is shifted in time by a
  random 21+ day offset. This keeps its exposure and trading pattern but breaks its alignment with
  prices. Two p-values come from this: one for alpha, and one for drawdown. The drawdown test
  matters because a rule that holds less stock always draws down less; only a drawdown milder
  than most shifts shows that the timing helped. ETFs get 500 shifts each; stock baskets get 200,
  run only where a criterion needs them.

## Decision rules (fixed now)

1. **Edge confirmed** for a (rule, market) requires all of:
   - alpha t ≥ 2.0 in **both** discovery and confirmation (net of costs);
   - full-window alpha DSR ≥ 0.95;
   - alpha > 0 at double costs;
   - random-timing alpha p ≤ 0.05;
   - removing the best 0.1% of days keeps the beta-adjusted active return positive (brief's sanity check).
2. **Risk control.** This does not mean beating the market; it means smaller drawdowns at no loss of
   risk-adjusted return. It requires all of:
   - maximum drawdown no worse than 2/3 of buy-and-hold's in **both** halves;
   - Sharpe ≥ buy-and-hold's in both halves;
   - random-timing drawdown p ≤ 0.01.
3. **The stock arm is survivor-biased,** and the bias depends on the rule type:
   - **Mean reversion** (RSI, Bollinger dip-buying) is flattered: the data keeps only stocks that
     recovered.
   - **Trend-following** is penalized: its exits look needless when every stock in the data survived.

   An edge confirmed only in the stock arm is reported as **not established** until it is checked on
   survivorship-free data.
4. **Cross-sectional technical signals by basket.** Six features: RSI(14), Bollinger %b, distance
   from the 150- and 200-day SMAs, MACD histogram, proximity to the 52-week high. Each gets a monthly
   rank IC with the next month's return.
   - A feature is **promising** if |t| ≥ 2 in both halves with the same sign.
   - That would justify a separate pre-registered model study, nothing more.
5. **No default of the system changes from this study.**
   - A confirmed edge becomes a candidate for paper trading.
   - A risk-control rule becomes a candidate overlay for the 20% drawdown mandate, to be tested in a
     later pre-registered study.

## Null calibration (run before the real data, on random walks)

The whole pipeline was run on five synthetic worlds with no signal. Each has 11 fat-tailed
random-walk "ETFs", 120 stocks and a random VIX, over the same dates. These results set the false-positive
rates of the criteria above:

| Over 5 null worlds (1,200 trials: 880 ETF, 320 stock) | ETF arm | Stock arm |
|---|---|---|
| alpha t ≥ 2 in discovery / in confirmation | 13 / 11 (1.5% / 1.3%) | 1 / 1 |
| alpha t ≥ 2 in **both** halves | 0 | 0 |
| alpha DSR ≥ 0.95 (highest in any world: < 0.001) | 0 | 0 |
| **edge confirmed** | **0** | **0** |
| drawdown ≤ 2/3 and Sharpe ≥ buy-and-hold in both halves | 17 (1.9%) | 0 |
| **risk control** (adds random-timing drawdown p ≤ 0.01) | **1** (0.1%) | **0** |
| promising cross-sectional feature (of 24 per world) | 0 | 0 |
| persistence: Spearman of gross alphas, mean (range) | 0.06 (−0.13 to 0.17) | 0.16 (−0.04 to 0.33) |
| the same on net alphas, for reference | 0.06 | **0.62**: costs alone |

On real data, therefore:
- one confirmed edge is very unlikely to be chance;
- one risk-control flag in 176 ETF trials is what noise produces;
- a stock-arm persistence below about 0.35 is within the random-walk range.

Two design changes came out of this calibration, both made before the real run:
- **Drawdown test:** drawdown reduction is now judged against random timing, not just against
  buy-and-hold. Any rule with less exposure looks safer.
- **Persistence:** it is measured on **gross** alphas. Costs make net alphas persist mechanically,
  because frequent traders lose in both halves.

A rule that never differs from buy-and-hold gets alpha 0 and no DSR. Otherwise rounding noise of
about 1e-17 produced meaningless t-statistics.

## Also reported (descriptive, no decisions)

- **Best in discovery → confirmation** for each focus market: the rule with the best discovery alpha,
  and its rank among the 16 in confirmation. This shows what picking the best backtest does.
- **Persistence:** Spearman correlation of gross alphas between the halves, next to the
  random-walk reference.
- **Buy-and-hold of each market vs SPY** in each half. This asks whether focusing on tech, energy or
  large caps beats the index. It is a hindsight question, because sector leadership changes.
- Basket sizes and data coverage.

## Limitations stated before the run

- **Stock arm:**
  - survivor-only data;
  - Yahoo's *current* sector labels, not GICS and not point-in-time;
  - market caps approximate before 2015.
- **XLK and XLE** are cap-weighted and concentrated (XLK: Apple, Microsoft, Nvidia; XLE: Exxon,
  Chevron). They measure large-cap tech and energy as investable products.
- **Indicators** use dividend-adjusted prices. Traders watching raw prices see slightly different
  levels.
- **Taxes** are not modeled. Short-term trading is taxed more heavily, so a real edge would also
  have to cover that.
- **2013–2026 was used in earlier studies,** with other strategies. These rules have never been run
  on it. No later data exists yet; paper trading is the next truly new data.
