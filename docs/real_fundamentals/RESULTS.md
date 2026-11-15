# Real data + free point-in-time fundamentals — results

Protocol: [PROTOCOL.md](PROTOCOL.md), committed and pushed in `7dd69d9` before the single run on
15 November 2026. Outputs: [`results.json`](results.json); every configuration is in
[`registry.jsonl`](registry.jsonl), with its monthly excess returns in [`returns.csv`](returns.csv).
Raw prices and SEC data are not committed. Rebuild them with `python -m stocksel.data.build_yahoo_csv`,
then `python -m stocksel.data.build_sec_fundamentals`.

## Outcome: **no skill**

| Pre-registered criterion (primary window Jan 2014 – Sep 2024) | Required | Result | |
|---|---|---|---|
| F1 rank IC t-statistic, and AUC | t ≥ 2.0 and AUC > 0.505 | t = **−0.02**, AUC = **0.485** | fail |
| Paired net difference F1 − F0 (default mandate) | > 0 with P ≥ 0.90 | **−1.9%/yr** (CI −5.1, +1.3), P = 0.13 | fail |
| F1 alpha DSR, charged with all 57 real-data trials | ≥ 0.95 | **0.03** | fail |

Adding free point-in-time fundamentals and earnings surprises does **not** give the model measurable
stock-selection skill on real S&P 500 data. The model ranks stocks no better than chance with or
without them.

## Results, primary window (Jan 2014 – Sep 2024, 129 months, net of costs)

| | F0: price only | F1: + fundamentals | SPY |
|---|---:|---:|---:|
| **Default (absolute mandate)** CAGR | 7.1% | 5.2% | 13.4% |
| Excess vs SPY (95% CI) | −6.4% (−12.4, −0.9) | −8.4% (−15.2, −1.7) | |
| Sharpe / Sortino | 0.56 / 0.76 | 0.41 / 0.55 | 0.79 |
| Max drawdown (month-end) | −23% | −21% | −23% |
| β / CAPM α | 0.51 / −0.4%/yr | 0.46 / −1.7%/yr | |
| **Relative mandate** (fully invested) CAGR | 11.1% | 12.1% | 13.4% |
| Excess vs SPY (95% CI) | −2.0% (−5.0, +1.0) | −1.0% (−5.2, +3.0) | |
| Max drawdown | −28% | −27% | −23% |
| Paired F1 − F0, relative mandate | | +0.9%/yr (−3.8, +5.4), P = 0.68 | |
| **Selection**: AUC / mean rank IC (t) | 0.487 / −0.010 (−0.76) | 0.485 / −0.000 (−0.02) | |
| Alpha DSR (absolute / relative) | 0.12 / 0.05 | 0.03 / 0.08 | |

The default mandate trails SPY mainly because it holds about half the market's beta. That is the
volatility target implied by a 20% drawdown tolerance, the same finding as in the
[first look](../real_firstlook/RESULTS.md). Without skill in selection, a lower-beta book trails a
rising market. The relative mandate removes that effect, and is statistically indistinguishable from
the index in both versions.

Other windows tell the same story:
- **Full out-of-sample 2005–2024.** F0 reproduces the first look exactly (7.3% vs SPY 10.4%).
  F1: 6.1% (absolute), 9.4% (relative). IC t = +0.32.
- **Nov 2024 – Sep 2026.** This period was already viewed in the first look, so it is *not* a clean
  holdout. F1 returned 2.8% absolute and 13.5% relative, vs SPY 14.1%. IC −0.022.

## Single signals on real data (research window, 2010–2024, monthly rank IC with next-month excess return)

| Signal | Mean IC | t | | Signal | Mean IC | t |
|---|---:|---:|---|---|---:|---:|
| quality_roe | +0.014 | 1.56 | | rev_1m | +0.014 | 1.09 |
| quality_gpa | +0.012 | 1.62 | | size (small > large) | +0.008 | 1.01 |
| sue | +0.005 | 0.95 | | illiq | +0.005 | 0.64 |
| pead | +0.005 | 0.92 | | mom_12_1 | +0.005 | 0.32 |
| value_ep | +0.004 | 0.35 | | mom_6_1 | −0.004 | −0.28 |
| value_bm | −0.011 | −0.92 | | vol_252 (low vol) | −0.009 | −0.46 |

None of the 13 signals reaches t = 2. The pattern matches the published literature for US large caps
after 2010:
- **Quality** (gross profitability, ROE) is the best signal, but t ≈ 1.6. Among 13 noise signals, the
  best would typically show about that.
- **Value** (book-to-market) was negative. This is the well-documented value drawdown of the 2010s.
- **Earnings surprise / PEAD** is weak (t ≈ 0.9), as expected from free data. The 10-Q/10-K
  filing date comes days to weeks after the earnings press release, so most of the drift has
  already happened. The seasonal-random-walk surprise is also noisier than one measured against
  analyst consensus.
- **Momentum and low volatility**, the main price signals in the brief, show nothing in this window.

The model combines all of these, so its score inherits the zero-to-negative signals along with the
mildly positive ones. **No better combination was searched for.** Choosing weights after seeing this
table would fit noise in data that has already been viewed.

## What this means

1. **With free data (prices + SEC filings), the system has no demonstrable skill** on the S&P 500.
   This is the brief's default assumption, and it is now measured, not assumed. The data is
   survivor-only, which biases results *upward*, and there is still no skill.
2. **It does not rule out skill with better data.** Three signal classes from the brief are still
   untested on real data: earnings surprises timed by announcement and measured against consensus,
   analyst revisions, and news sentiment. All need paid sources. The literature also suggests their
   large-cap effect has weakened since publication; the brief's 50% haircut reflects this.
3. **The protocol's trigger for buying data was a positive result here; it did not occur.** Paying for
   data would now rest on the literature alone. If you do it, pre-register a new window, and use
   survivorship-free data.
4. **No real money.** The readiness gate correctly stays closed. The cheapest honest next step is
   paper trading, which builds a clean out-of-sample record going forward.

## Caveats

- Survivor-only prices (716 of 1,133 historical members): results are an upper bound.
- Gross profit is unavailable for banks, insurers and utilities (coverage about 56%).
- Some members have no fundamentals for part of their history, and score as neutral then. They
  include Navient, DuPont (pre-2017), TechnipFMC (pre-2017), DXC, and Alphabet (pre-2015, whose
  predecessor failed the automatic verification). In the primary window this is 0.5% of member-months
  (286 of 54,467).
- The study answers one question with one model family (the gated linear ensemble that the gate
  selected everywhere so far). It is not a search over models; that would need a new holdout.

## Data-quality issues found and fixed while building the data (before the run)

| Issue | Effect if unfixed | Fix (tested) |
|---|---|---|
| Real filings have gaps | trailing-year sums over 5 quarters | TTM requires 4 consecutive quarters |
| Same | SUE compares unlike quarters | Seasonal SUE uses only the same quarter a year earlier |
| Early XBRL filings restate old years as comparatives | a stale quarter overwrites the newest one in the as-of join | Only the newest fiscal period known at the time enters the join |
| Companies re-register with the SEC (Exxon 2026, BlackRock 2024, Disney 2019, Cigna 2019, …) | their history before the change disappears | Predecessor merged only when the successor's comparative figures match it (12 accepted, 6 rejected) |

None of these is look-ahead: each would have made the data noisier, not leakier. None changes the
synthetic results; this was checked.
