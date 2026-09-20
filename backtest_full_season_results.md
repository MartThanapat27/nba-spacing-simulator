# Point-in-Time Dual-Mode Backtest Report

- Generated: 2026-09-20T01:12:09
- Source dataset: `C:\Users\tpmar\Desktop\Project\nba-spacing-simulator\historical_games.csv`
- Games: 1225 (scored: 1225), date range 2024-10-22 to 2025-04-13

## Mode A (Blended, Safety Brake ON) vs Mode B (Pure Raw, Safety Brake OFF)

| Metric | Mode B (Raw) | Mode A (Blended) | Delta |
|---|---|---|---|
| Win/Loss Accuracy | 59.4% | 62.0% | +2.5% |
| Brier Score | 0.2339 | 0.2261 | -0.0077 |
| Point-Diff MAE | 11.93 | 11.93 | +0.00 |

MAE is mathematically identical between modes by construction -- the Safety Brake only ever blends the win probability, never the predicted score/margin.

## Statistical Significance

- N (paired games): 1225
- McNemar's test (accuracy): chi2=6.383, p=0.0115 -> **significant** at alpha=0.05 (86 games only Mode A got right, 55 only Mode B got right)
- Paired Brier-score difference (Mode A - Mode B): mean -0.0077, 95% CI [-0.0103, -0.0051], p=0.0000 -> **significant** at alpha=0.05

*Caveat: both tests assume per-game independence, which is only approximate (teams/dates repeat within the sample) -- treat p-values as a rough confidence signal, not a rigorously i.i.d. hypothesis test.*

## Dynamic Safety-Brake Weight Summary

- Games with a real NET_RATING match this season: 1151/1225
- Raw-sim weight actually used: min 0.400, mean 0.671, max 0.850
- abs(delta NET_RATING) across these games: min 0.0, mean 7.4, max 43.2

Full per-game breakdown (1225 games) is in the companion JSON report.
