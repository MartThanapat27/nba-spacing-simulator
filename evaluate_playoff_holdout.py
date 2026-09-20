#!/usr/bin/env python3
"""Evaluates the trained Hybrid ML pipeline against a SEPARATE, NEVER-
TRAINED-ON holdout: the real 2024-25 NBA playoffs (fetched via
`fetch_real_nba_data.py --season-type Playoffs` into
historical_games_playoffs.csv -- 84 real games, verified via the NBA's own
game_id convention to be genuine playoff games, see
train_ml_model.py's classify_game_id()).

This is a genuinely different evaluation than backtest_model.py's standard
n=245 regular-season chronological holdout: every one of these 84 games is
a real playoff game the model has NEVER seen in any form (playoff games are
deliberately excluded from the training CSV -- see train_ml_model.py's
module docstring -- so they can't leak into training even indirectly via
sample weighting). This measures whether the Hybrid ML pipeline, trained
only on regular-season games with training-time sample weighting toward
real high-leverage regular-season games (top-4 conference contenders,
prior-season Finals/Conf-Finals rematches, In-Season Tournament knockout
games -- see train_ml_model.py's compute_high_leverage_flags()), actually
generalizes to genuine playoff-atmosphere games.

Reuses the ALREADY-TRAINED model + team_features from ml_model/ (no
retraining here -- run train_ml_model.py first) and backtest_model.py's
existing per-game GPU evaluation harness (run_backtest_pass/print_summary/
print_comparison) for full methodological consistency with the regular-
season backtest.

Every playoff game gets --hot-hand-boost applied in the hybrid pass (a
playoff game is inherently high-stakes by definition -- no separate
top-4/rematch/tournament check is needed the way it is for a regular-season
game).

Usage:
    python evaluate_playoff_holdout.py
    python evaluate_playoff_holdout.py --csv historical_games_playoffs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib

import backtest_model as bt
import train_ml_model as ml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PLAYOFF_CSV = SCRIPT_DIR / "historical_games_playoffs.csv"
DEFAULT_ML_DIR = SCRIPT_DIR / "ml_model"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/players"
DEFAULT_SEASON = "2024-25"
DEFAULT_TIMEOUT = 90.0


class _Args:
    """Minimal stand-in for argparse.Namespace -- run_backtest_pass only
    reads .timeout and .verbose off whatever `args` object it's given."""
    def __init__(self, timeout: float, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose


def evaluate_playoff_holdout(csv_path: Path, ml_dir: Path, api_url: str, season: str,
                              exe: Path, timeout: float) -> dict:
    model_path = ml_dir / "margin_model.joblib"
    features_path = ml_dir / "team_features.json"
    if not model_path.is_file() or not features_path.is_file():
        raise ml.MlPipelineError(f"No trained model found in {ml_dir}/ -- run train_ml_model.py first.")

    model = joblib.load(model_path)
    with features_path.open(encoding="utf-8") as f:
        raw_features = json.load(f)
    team_features = {abbr: ml.TeamFeatures(**vals) for abbr, vals in raw_features.items()}

    print(f"Fetching live roster/defensive-rating snapshot (same inputs the last training run used)...")
    players = ml.fetch_players(api_url)
    players_df = ml._prep_players_df(players)
    def_ratings = ml.fetch_team_defensive_ratings(season)
    _, star_names = ml.compute_team_features(players_df, def_ratings)

    star_cache_path = ml_dir / "star_availability_cache.json"
    star_cache = {}
    if star_cache_path.is_file():
        with star_cache_path.open(encoding="utf-8") as f:
            star_cache = json.load(f)

    print(f"Building feature table for {csv_path.name} (per-game star-availability checks against "
          f"real box scores -- these are NEW games not in the cache, so this makes live nba_api "
          f"requests)...")
    table = ml.load_training_table(csv_path, players_df, team_features, star_names, def_ratings,
                                    star_cache, season=season)

    with star_cache_path.open("w", encoding="utf-8") as f:
        json.dump(star_cache, f, indent=2)

    # MODEL_FEATURE_NAMES (not FEATURE_NAMES) -- matches the actual columns
    # the loaded model was trained/predicts on; see MODEL_FEATURE_NAMES's
    # docstring in train_ml_model.py for why big_match_indicator is excluded.
    x_cols = [f"feat_{name}" for name in ml.MODEL_FEATURE_NAMES]
    predicted_margins = model.predict(table[x_cols].to_numpy()).tolist()

    games = [
        bt.HistoricalGame(
            team_a=row.team_a, team_b=row.team_b, actual_winner=row.actual_winner,
            actual_score_a=row.actual_score_a, actual_score_b=row.actual_score_b,
            game_date=str(row.game_date),
            team_a_rest_days=row.team_a_rest_days, team_b_rest_days=row.team_b_rest_days,
        )
        for _, row in table.iterrows()
    ]
    # Every game here is a REAL playoff game (verified via game_id, see
    # classify_game_id()) -- inherently high-stakes, so the hybrid pass
    # applies --hot-hand-boost to every one of them, no per-game check needed.
    is_marquee_flags = [True] * len(games)

    print(f"\n{len(games)} real {season} playoff games loaded -- NEVER used in training "
          f"(playoff games are excluded from historical_games.csv by design).")

    args = _Args(timeout=timeout)
    _, baseline_summary = bt.run_backtest_pass(
        f"Monte Carlo Baseline on PLAYOFF holdout (pure calibrated engine, no ML) -- {season}",
        games, exe, args, ml_margins=None, is_marquee_flags=None)

    _, hybrid_summary = bt.run_backtest_pass(
        f"Hybrid (engine + ML margin + Hot Hand) on PLAYOFF holdout -- {season}",
        games, exe, args, ml_margins=predicted_margins, is_marquee_flags=is_marquee_flags)

    if baseline_summary is None or hybrid_summary is None:
        raise ml.MlPipelineError("No playoff games were successfully scored -- see errors above.")

    bt.print_comparison(baseline_summary, hybrid_summary)

    return {
        "n_games": hybrid_summary["n"],
        "baseline_accuracy": baseline_summary["accuracy"] / 100.0,
        "baseline_correct": round(baseline_summary["accuracy"] / 100.0 * baseline_summary["n"]),
        "hybrid_accuracy": hybrid_summary["accuracy"] / 100.0,
        "hybrid_correct": round(hybrid_summary["accuracy"] / 100.0 * hybrid_summary["n"]),
        "baseline_brier": baseline_summary["brier"],
        "hybrid_brier": hybrid_summary["brier"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_PLAYOFF_CSV)
    parser.add_argument("--ml-dir", type=Path, default=DEFAULT_ML_DIR)
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--season", type=str, default=DEFAULT_SEASON)
    parser.add_argument("--exe", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--out-json", type=Path, default=None,
                         help="Where to write the result summary (default: <ml-dir>/playoff_holdout_results.json)")
    args = parser.parse_args(argv)

    try:
        exe = bt.find_executable(args.exe)
        result = evaluate_playoff_holdout(args.csv, args.ml_dir, args.api_url, args.season, exe, args.timeout)
    except (ml.MlPipelineError, bt.BacktestError) as e:
        print(f"[evaluate_playoff_holdout] Fatal: {e}", file=sys.stderr)
        return 1

    out_path = args.out_json or (args.ml_dir / "playoff_holdout_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
