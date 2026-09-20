#!/usr/bin/env python3
"""Renders a backtest_model.py `--report-json` artifact into a compact,
human-readable Markdown summary -- the companion to the full JSON report
(Artifact Preservation: the JSON keeps the complete per-game breakdown for
programmatic re-analysis; this Markdown file is for a quick morning read).

Usage:
    python build_markdown_report.py backtest_full_season_results.json backtest_full_season_results.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def render(report: dict) -> str:
    b, a, c = report["mode_b_raw"], report["mode_a_blended"], report["comparison"]
    sig = report["statistical_significance"]
    w = report["dynamic_weight_summary"]
    date_range = report.get("date_range") or ["?", "?"]

    lines = [
        "# Point-in-Time Dual-Mode Backtest Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Source dataset: `{report['source_dataset']}`",
        f"- Games: {report['n_games']} (scored: {report['n_scored']}), "
        f"date range {date_range[0]} to {date_range[1]}",
        "",
        "## Mode A (Blended, Safety Brake ON) vs Mode B (Pure Raw, Safety Brake OFF)",
        "",
        "| Metric | Mode B (Raw) | Mode A (Blended) | Delta |",
        "|---|---|---|---|",
        f"| Win/Loss Accuracy | {b['accuracy']:.1f}% | {a['accuracy']:.1f}% | {c['accuracy_delta']:+.1f}% |",
        f"| Brier Score | {b['brier']:.4f} | {a['brier']:.4f} | {c['brier_delta']:+.4f} |",
        f"| Point-Diff MAE | {b['mae']:.2f} | {a['mae']:.2f} | {c['mae_delta']:+.2f} |",
        "",
        "MAE is mathematically identical between modes by construction -- the Safety Brake "
        "only ever blends the win probability, never the predicted score/margin.",
        "",
        "## Statistical Significance",
        "",
        f"- N (paired games): {sig['n']}",
    ]

    if sig["mcnemar_chi2"] is not None:
        verdict = "**significant**" if sig["mcnemar_significant_at_0_05"] else "not significant"
        lines.append(
            f"- McNemar's test (accuracy): chi2={sig['mcnemar_chi2']:.3f}, "
            f"p={sig['mcnemar_p_value']:.4f} -> {verdict} at alpha=0.05 "
            f"({sig['only_blended_correct']} games only Mode A got right, "
            f"{sig['only_raw_correct']} only Mode B got right)"
        )
    else:
        lines.append("- McNemar's test: no discordant games -- accuracy figures are identical.")

    if sig["brier_diff_mean"] is not None:
        if sig.get("brier_diff_ci_95") is not None:
            lo, hi = sig["brier_diff_ci_95"]
            verdict = "**significant**" if (lo > 0 or hi < 0) else "not significant"
            lines.append(
                f"- Paired Brier-score difference (Mode A - Mode B): mean {sig['brier_diff_mean']:+.4f}, "
                f"95% CI [{lo:+.4f}, {hi:+.4f}], p={sig['brier_diff_p_value']:.4f} -> {verdict} at alpha=0.05"
            )
        else:
            lines.append(f"- Paired Brier-score difference: mean {sig['brier_diff_mean']:+.4f} "
                          "(zero variance across games -- no CI)")

    lines += [
        "",
        "*Caveat: both tests assume per-game independence, which is only approximate "
        "(teams/dates repeat within the sample) -- treat p-values as a rough confidence "
        "signal, not a rigorously i.i.d. hypothesis test.*",
        "",
        "## Dynamic Safety-Brake Weight Summary",
        "",
        f"- Games with a real NET_RATING match this season: {w['n_with_real_net_rating_match']}/{report['n_scored']}",
    ]
    if w["weight_mean"] is not None:
        lines.append(f"- Raw-sim weight actually used: min {w['weight_min']:.3f}, "
                      f"mean {w['weight_mean']:.3f}, max {w['weight_max']:.3f}")
        lines.append(f"- abs(delta NET_RATING) across these games: min {w['delta_net_rating_min']:.1f}, "
                      f"mean {w['delta_net_rating_mean']:.1f}, max {w['delta_net_rating_max']:.1f}")

    lines += [
        "",
        f"Full per-game breakdown ({len(report.get('per_game', []))} games) is in the companion JSON report.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("Usage: python build_markdown_report.py <report.json> <report.md>", file=sys.stderr)
        return 1

    json_path, md_path = Path(argv[0]), Path(argv[1])
    with json_path.open(encoding="utf-8") as f:
        report = json.load(f)

    md_path.write_text(render(report), encoding="utf-8")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
