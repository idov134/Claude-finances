# Algorithm research v3 — results

Pre-registration: [PROTOCOL.md](PROTOCOL.md), committed in `62519f4` before the single run. Raw
output: `results.json`, `registry.jsonl`, `returns.csv`, `net_returns_all_worlds.csv`. **Synthetic
data only.**

## Mandate frontier (net, monthly; 239 OOS months per world)

| World | Config | CAGR | Bench CAGR | Excess vs bench (95% CI) | Sharpe | Sortino | Max DD | Calmar | β |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 7 (dev) | B0 absolute | 12.47% | 9.11% | +1.98% (−3.8, +6.5) | 1.094 | 1.99 | −22.2% | 0.56 | 0.47 |
| | E5 relative | 16.01% | | +6.22% (+4.6, +7.9) | 0.827 | 1.53 | −44.2% | 0.36 | 0.99 |
| | **M1 relative + regime filter** | 16.08% | | +6.04% (+3.7, +8.1) | 0.891 | 1.66 | −36.9% | 0.44 | 0.87 |
| 404 | B0 | 11.35% | 7.13% | +2.82% (−2.8, +8.6) | 0.938 | 1.58 | −24.1% | 0.47 | 0.53 |
| | E5 | 14.44% | | +6.69% (+4.6, +8.9) | 0.759 | 1.20 | −55.3% | 0.26 | 0.98 |
| | **M1** | 15.49% | | +7.42% (+4.8, +10.3) | 0.846 | 1.39 | −42.7% | 0.36 | 0.91 |
| 505 | B0 | 10.80% | 6.99% | +2.19% (−3.2, +7.7) | 0.925 | 1.63 | −31.0% | 0.35 | 0.45 |
| | E5 | 14.37% | | +6.79% (+4.7, +8.8) | 0.719 | 1.13 | −64.8% | 0.22 | 0.99 |
| | **M1** | 15.61% | | +7.56% (+4.9, +10.4) | 0.821 | 1.42 | −51.0% | 0.31 | 0.89 |
| 606 | B0 | 12.39% | 8.31% | +2.48% (−3.2, +8.4) | 1.026 | 1.73 | −21.4% | 0.58 | 0.50 |
| | E5 | 15.68% | | +6.83% (+5.2, +8.4) | 0.775 | 1.19 | −53.2% | 0.30 | 1.02 |
| | **M1** | 16.31% | | +6.83% (+3.3, +10.6) | 0.912 | 1.43 | −35.1% | 0.47 | 0.83 |

Pooled M1 excess over the benchmark on the fresh worlds: **+7.27%/yr (95% CI +5.6, +9.0), P = 1.00**.

**Decision: M1 not adopted as the default.** Its drawdown (−35% to −51%) exceeds the configured 20%
tolerance in every world, which fails the pre-registered rule 2. Compared with E5 (no filter),
the regime filter raised Sharpe in all four worlds (by 0.06 to 0.14) and cut drawdown by 7 to 18
percentage points, while excess return was similar or higher (−0.2 to +0.8 points). It is the better
of the two fully invested mandates.

The choice is a risk preference, not a statistical question:

| Mandate (config) | Typical outcome on the synthetic worlds |
|---|---|
| `mandate: absolute` (default) | Sharpe ≈ 0.9–1.1, drawdown ≈ −21% to −31%, excess over raw index **not established** (+2–3%/yr, CI includes 0); clearly ahead of a volatility-matched index |
| `mandate: relative` + `regime_filter.enabled: true` | Excess ≈ +6–7.5%/yr with CIs above 0 in every world, drawdown ≈ −35% to −51% |

The synthetic excess figures depend on how strong the simulator's planted signals are (IC ≈ 0.05).
Real large-cap long-only factor portfolios usually earn far less after costs and decay. Apply the
50% publication haircut and expect a smaller edge. Only real point-in-time data can settle this.

## Diagnostic D1b — is the model gate too strict?

World with one planted interaction (momentum works mainly in low-volatility names). Standalone
walk-forward, same features, same portfolio rules:

| Model | AUC | IC (t) | Log-loss | Net Sharpe | Net return/yr |
|---|---:|---|---:|---:|---:|
| Logistic regression | 0.5275 | 0.0536 (12.5) | 0.69151 | 1.10 | 12.2% |
| Tuned LightGBM (nested purged CV) | 0.5248 | 0.0469 (11.2) | 0.69165 | 0.96 | 11.2% |

The gate's bootstrap P(LightGBM beats logit) peaked at 0.65 (2013–2015) and ended at 0.09; the
admission threshold is 0.80. **The gate is not the problem.** Even when a real interaction exists,
monthly stock returns are too noisy for the tree model to learn it from about 30k training rows
better than the linear model's main effects. Complex models stay behind the gate. On real data
they can still win admission if they earn it out of sample, net.
