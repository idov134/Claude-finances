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
