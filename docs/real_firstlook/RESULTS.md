# Real-data first look — results

Protocol: [PROTOCOL.md](PROTOCOL.md), with two addenda written before the respective runs.
**v1.1** (this page) re-runs the frozen strategy on corrected data. v1, built on mis-scaled market
caps, is kept in [`v1_buggy_market_caps/`](v1_buggy_market_caps/). Raw prices are not committed;
rebuild them with `python -m stocksel.data.build_yahoo_csv`. Reports:
[research](research_report.html), [holdout](holdout_report.html) (marked CONTAMINATED: second
opening, after the data fix).

## Data

| | |
|---|---|
| Former + current S&P 500 members since 1999 | 1,133 tickers |
| With Yahoo prices | 716. Every one still trades today; delisted/renamed names are missing |
| Coverage of index members | 45% (1999) → 66% (2010) → 87% (2020) → 99% (2026) |
| Signals available | price/volume only: momentum 12-1/6-1, 1-month reversal, Amihud, volatility, idio vol, size |
| Research window (OOS) | Jan 2005 – Sep 2024, 237 months |
| Holdout | Nov 2024 – Sep 2026, 23 months |

## Results (net of costs)

| | Research: strategy | Research: SPY | Holdout: strategy | Holdout: SPY |
|---|---:|---:|---:|---:|
| **Default** (absolute mandate) CAGR | 7.26% | 10.37% | 3.04% | 14.14% |
| Excess vs SPY (95% CI) | −3.66% (−8.8, +1.1) | | −10.91% (−23.6, +2.0) | |
| Sharpe | 0.62 | 0.61 | −0.35 | 0.86 |
| Max drawdown (month-end) | −25.5% | −51.3% | −2.7% | −7.0% |
| β / CAPM α | 0.47 / +1.4%/yr | | 0.04 / −1.2%/yr | |
| **Relative** (fully invested) CAGR | 8.83%, excess −1.21% (−3.8, +1.4), DD −60.8% | | 12.46%, excess −1.16% (−9.7, +7.5) | |
| **Relative + regime filter** CAGR | 9.30%, excess −1.23% (−4.1, +1.5), DD −48.6% | | 12.46%, excess −1.16% | |

The bias-matched reference is every eligible stock in the same survivor-only universe:

| | Universe equal-weight | Universe cap-weight |
|---|---:|---:|
| Research CAGR | 10.58% | 8.84%* |
| Default − universe EW | −4.05%/yr (CI −9.2, +0.9) | |
| Relative − universe EW | −1.60%/yr (CI −4.1, +0.9) | |

\* Before 2015 the cap weights still rest on 2015 share counts. Heavy issuers such as Citi and Bank
of America are overweighted in 2005–08, so use the EW column. From 2015 on, where share history is
real, the survivor universe cap-weighted returned **14.3%/yr vs SPY 13.3%/yr**. That is the expected
survivorship tilt, and it confirms the corrected caps.

## Forecast quality: no skill on real data

| | Research | Holdout |
|---|---:|---:|
| AUC | **0.494** | **0.497** |
| Rank IC (t) | −0.006 (−0.6) | +0.022 (+0.7) |
| Accuracy | 49.8% | 54.2% (base rate 45.6%) |

- **Shuffled labels:** the permutation test reports "failed". The real model's IC is no better than
  models trained on shuffled labels, which is the honest verdict for a model without skill.
- **Signals:** single-signal IC over 2005–2024 is near zero for momentum (0.001), low volatility
  (−0.003) and residual momentum (0.007). Only reversal (0.020, t = 2.0), illiquidity (0.019) and
  size (0.023) are borderline. Size favouring small stocks is suspect in a survivor-only universe:
  the small survivors are the ones that did not fail.
- **The gate** admitted only the linear models, as on synthetic data.

## Multiple testing

52 configurations were charged: v1 and v1.1 are counted separately because the config changed
between them.

| Statistic | Value | Reading |
|---|---:|---|
| DSR (total Sharpe) | 0.959 | Passes 0.95, but only because a long-only book at β ≈ 0.47 earns a Sharpe from **holding equities** |
| **Alpha DSR**, all 52 trials | **0.41** | No significant stock-selection skill |
| Alpha DSR, effective trials (1.6) | 0.77 | Still not significant |
| PBO | 0.58 | The in-sample best trial tends to lose out of sample |

This run is why the system now reports an **alpha DSR** (deflated appraisal ratio). For a long-only
strategy, the brief's "DSR > 0.95" can be met with zero skill. The readiness gate now requires both.
For comparison, the synthetic world gives alpha DSR 0.62 (all 225 trials) / 1.00 (effective 2.4).

## Sub-periods (descriptive)

| | Default | Relative + regime | SPY | Universe EW |
|---|---:|---:|---:|---:|
| 2005–2014 CAGR | 8.0% | 8.4% | 7.6% | 9.4% |
| 2015–2024 CAGR | 6.5% | 10.3% | 13.3% | 11.8% |

The default's good first decade came mostly from low beta through 2008. In the second decade it
lagged badly: low beta plus the cash collapse.

## What went wrong, and what was fixed

1. **Data bug (fixed, v1 → v1.1).** Share counts were as-reported while prices were
   split-adjusted, so market caps were understated by later splits (Apple 2019: USD 228B vs USD
   911B). Also the CSV cache ignored file changes. Both are fixed with tests.
2. **Design flaw (documented; fix tested in [v4](../algo_research_v4/RESULTS.md), not adopted).**
   The absolute mandate goes to **cash** when the probability of beating the index falls below 0.5
   for every stock. That happened in 2021–22 (13 months) and for 21 of 23 holdout months. Probability
   centering removes the cash months; post hoc on the real window it did not help (−4.4% vs −3.7%
   vs SPY), because there is no selection skill to invest.
3. **Root cause: no skill from price-only signals in large caps.** This matches the literature:
   momentum and low-volatility premia in the largest US stocks have been weak since 2000. The
   synthetic world's planted fundamental, earnings and analyst signals, which drove its results,
   are absent here.

## Conclusion

On real data, **this system as configured does not beat buy-and-hold of the S&P 500, and shows no
stock-selection skill**. That holds even before accounting for survivorship bias, which flatters it.
The default assumption in the brief holds. Defaults are unchanged. **No real money.**

The value of this run:
- the pipeline works end to end on real data;
- it found two real bugs and one design flaw;
- it shows that the synthetic results came from the simulator's planted signals.

Next steps, in order of value:
1. **Point-in-time fundamentals, earnings surprises and analyst revisions** (for example Sharadar
   SF1, or Zacks/I/B/E/S). In the literature these carry the large-cap signal that prices alone do not.
2. **Survivorship-free prices** (Sharadar SEP) and a **new pre-registered holdout window**.
3. Re-evaluate the mandate (absolute / relative / centering) only on that data.
