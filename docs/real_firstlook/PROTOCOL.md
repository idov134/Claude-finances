# Real-data first look — pre-registration

Registered **9 November 2026**, committed before the run (runner: `stocksel/experiments/real_firstlook.py`).
This is the first time the system sees real market data.

## Question

Does the system as designed find out-of-sample signal in **real** S&P 500 stocks, and does it beat
SPY net of costs, in a backtest (2000–2024) and in an untouched 24-month holdout
(Nov 2024 – Nov 2026)?

## Data (built by `python -m stocksel.data.build_yahoo_csv`, kept local, never committed)

- Historical S&P 500 membership intervals since 1996: github.com/fja05680/sp500.
- Daily OHLCV, split and dividend adjustment, share counts, SPY, ^VIX and ^IRX: Yahoo Finance.
- Current GICS sectors for current members; "Unknown" for former members.

**Known biases, accepted in advance:**
1. **Survivorship.** Yahoo no longer serves most delisted or renamed former members, so the
   tradable universe in early years is a subset tilted toward survivors, while SPY includes the
   failures. Backtest excess returns are therefore an **upper bound**. `coverage.csv` measures the
   gap month by month, and the system raises a critical survivorship flag.
2. Market cap = unadjusted close × shares outstanding. Share history starts around 2015 and is
   back-filled earlier, so size and value weights are approximate.
3. No point-in-time fundamentals, earnings, analyst estimates or news. Only price and volume
   signals can be used.

## Frozen configuration

`config.yaml` defaults, plus these overrides only:
- `data.provider: csv`, `data.csv_dir: data_csv`, `data.start: 1999-01-01`; separate artifacts
  (`artifacts/real_firstlook`), so the synthetic trial registry and holdout lock are untouched.
- Feature set **`price_only`** = mom_12_1, mom_6_1, rev_1m, illiq, vol_252, idio_vol, size. The
  fundamental and earnings signals of `core_comp` don't exist in this data. Trial grid feature sets:
  `[price_only]`.
- Everything else is the default: walk-forward (10-year rolling window, yearly refits), candidates
  {logit, ridge, enet, lgbm, mlp} behind the online gate, absolute mandate (Kelly caps, 10% vol
  target), top 50 with buffer, costs ×1, stops, kill switch.
- Pre-registered **alternative mandates**, evaluated on the same predictions and logged as trials:
  `relative` (fully invested) and `relative_regime` (fully invested + trend/VIX filter). They come
  from the synthetic v2/v3 findings. They are reported, not selected.

## Holdout

The last 24 months are physically removed before research. The holdout is opened **once**, in the
same run, immediately after research, for the frozen spec and the two pre-registered mandates.
There is no tuning between the two steps: the code path is a single command. No parameter may be
changed in response to this run's results without a new protocol, and this holdout is then used.

## Reported

All success metrics (net return, DSR charged for this run's trials, drawdown/Calmar/Sortino,
excess and IR vs SPY with bootstrap CIs, then hit rate), the forecast diagnostics (AUC, IC,
calibration), coverage statistics, and every red flag. Red flags (accuracy > 65%, Sharpe > 2)
mean investigation, not celebration. No adoption decision is made from this run.

## Addendum (9 Nov 2026, before the run)

The dataset build showed the scale of the survivorship gap: 716 of 1,133 tickers that were ever
members, with **45% coverage of index members in 1999 rising to 99% in 2026**. Every covered
ticker still trades today. SPY therefore contains failures that the strategy's universe cannot
hold, and "excess vs SPY" mixes skill with survivorship.

A **bias-matched reference** is added to the report: the monthly return of **all eligible stocks**
in the same survivor-only universe, cap-weighted and equal-weighted, frictionless, next open to
next rebalance open. It is the same universe the model chooses from. Strategy minus this reference
estimates **selection skill net of costs**, with the survivorship bias largely shared on both
sides. It is a diagnostic benchmark only: nothing is selected or tuned with it.

## Addendum 2 (9 Nov 2026, after v1, before v1.1): data-construction bug, frozen strategy

v1 ran as registered. Its results exposed a **bug in my dataset builder**, not in the strategy.
Yahoo's `Close` is split-adjusted, but its share-count history is as reported at the time. Market
caps before a split were therefore understated by the whole later split ratio, for example Apple in
2019 at USD 228B instead of about USD 900B. This hits exactly the biggest winners (Apple, Nvidia,
Amazon, Google, Tesla) and distorts the size signal, the size filter, the capped value weights and
the bias-matched reference. A second bug: the CSV cache was keyed on the config only, so rebuilt
files could be served stale. Both are fixed, with tests.

**v1.1 re-runs the identical frozen strategy on the corrected data.** No model, feature, portfolio
or threshold changes. The holdout is re-opened with `--reopen-after-data-fix`; the lock records a
second opening and the holdout report is marked **CONTAMINATED**. Its evidence is weaker than a
pristine first opening: I have seen the v1 holdout results, although nothing about the strategy
changed. v1 is kept in `v1_buggy_market_caps/`.

The v1 results also revealed a **strategy design flaw**, separate from the bug. The absolute
mandate holds a stock only if its haircut probability of beating the index exceeds 0.5. When the
recent cross-sectional base rate is low (a narrow, mega-cap-led market), every probability falls
below 0.5 and the portfolio goes to **cash**. That happened in 2021–22 and for 21 of the 23 holdout
months. The alternative to a stock under this mandate is cash, not the index, so this is an
unintended market-timing rule. **It is not changed in v1.1** (the strategy is frozen). It is
studied separately under its own pre-registration on data not used to find it.
