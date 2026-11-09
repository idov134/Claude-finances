# Research trial history

Every configuration ever evaluated on this repository's research data is kept here or in the
experiment folders listed below, so that the Deflated Sharpe Ratio and PBO can always be charged
for the *full* search, not just the latest experiment.

| Source | Folder | Distinct trial IDs | Return convention |
|---|---|---|---|
| Original research run (Claude, 2 Nov 2026; 48 configurations, run twice) | `original_research_v1/` | 48 | monthly net − rf, decision-date index |
| Codex re-entry experiment (3 Nov 2026) | `../reentry_review/` | 2 | monthly net, calendar month-end index (daily engine) |
| Codex research comparison (5 Nov 2026) | `../research_comparison/` | 33 (11 completed, 12 failed, rest planned) | monthly net − rf, decision-date index |
| Algorithm research v2 (Claude, Nov 2026) | `../algo_research_v2/` | see that folder | monthly net − rf, decision-date index |

The original 48-trial return matrix was produced by the legacy monthly engine (zero substitution
for missing returns, gross-equity drift). It is kept unchanged as an audit record of what was
searched; its returns are used for trial *counting* and cross-trial Sharpe variance only.

`stocksel.validation.history.load_history()` reads all of them. Failed and planned-but-unfinished
trials count toward N (conservative).
