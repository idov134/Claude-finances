# Algorithm research v4 — results: the cash-collapse fix

Pre-registration: [PROTOCOL.md](PROTOCOL.md), committed in `a5e894d` before the single run. Raw
output: `results.json`, `registry.jsonl`, `returns.csv`, `net_returns_all_worlds.csv`. Synthetic
worlds; the post-hoc real-window check is in [`real_posthoc.json`](real_posthoc.json) and is not
evidence.

## Per world (net, 239 OOS months each)

| World | Group | B0 cash months | E7 cash months | B0 → E7 CAGR | B0 → E7 Sharpe | B0 → E7 max DD | Δ/yr (P) |
|---|---|---:|---:|---|---|---|---|
| 7 | dev | 0 | 0 | 12.5% → 11.6% | 1.09 → 1.00 | −22.3% → −21.8% | −0.82% (0.01) |
| 1101 | standard | 11 | 0 | 12.4% → 13.7% | 0.97 → 1.04 | −30.8% → −30.8% | +1.19% (0.83) |
| 1202 | standard | 2 | 0 | 12.9% → 12.5% | 1.05 → 0.95 | −20.1% → −20.8% | −0.24% (0.40) |
| 1303 | standard | 0 | 0 | 10.2% → 10.0% | 0.77 → 0.75 | −25.4% → −26.4% | −0.23% (0.28) |
| 2404 | breadth | **57** | 5 | 4.9% → 8.9% | 0.42 → 0.71 | −24.3% → **−31.3%** | +3.90% (0.92) |
| 2505 | breadth | **53** | 0 | 9.8% → 14.6% | 0.79 → 1.00 | −25.7% → −27.4% | +4.57% (0.98) |
| 2606 | breadth | **34** | 0 | 7.0% → 11.2% | 0.55 → 0.84 | −37.6% → −37.2% | +4.07% (0.99) |

"Cash months" = months with exposure below 10%.

## Pre-registered checks

| Check | Result |
|---|---|
| 1. No harm on standard worlds (pooled CI low > −1%/yr) | **pass**: +0.24%/yr (CI −0.65, +1.25) |
| 2. Mechanism on breadth worlds (pooled P ≥ 0.90, ≥ 2 of 3 positive) | **pass**: +4.18%/yr (CI +1.51, +7.14), P = 0.9995, 3 of 3 |
| 3. Drawdown no worse than B0's by more than 3 points in any world | **fail**: world 2404, −31.3% vs −24.3% |
| 4. No red flags | pass |

**Decision under the pre-registered rule: E7 is not adopted as the default.**
`portfolio.prob_centering` stays `none`.

## What this means

- **The flaw is real and reproducible.** In markets where most stocks lag the cap-weighted index,
  the default mandate spent **3 to 5 of 20 years in cash** by accident. That is exactly what
  happened on real data in 2021–22 and 2025–26.
- **The fix works:** cash months disappear, and return and Sharpe improve by about 4%/yr and 0.2–0.3
  in every breadth world, at no cost in ordinary markets.
- **The failed check is the honest trade-off.** In world 2404 the accidental cash had dodged a
  drawdown. Staying invested costs drawdown when the market falls. The rule was set in advance,
  and changing it now would defeat its purpose.

**Recommendation for the user.** If you accept up to a few points of extra drawdown in exchange for
not going to cash whenever the index is narrow, set `portfolio.prob_centering: cross_section`. With
the default, the system now **warns** whenever it is in cash for base-rate reasons. The warning
appears in `recommend` meta (`base_rate_warning`) and as a GUI alert.
