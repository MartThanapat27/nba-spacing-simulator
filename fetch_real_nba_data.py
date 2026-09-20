#!/usr/bin/env python3
"""Fetches real, completed NBA game results via `nba_api` and writes them to
`historical_games.csv` in the schema the rest of the pipeline expects
(`train_ml_model.py`, `backtest_model.py`): team_a, team_b, actual_winner,
actual_score_a, actual_score_b, game_date, game_id, team_a_rest_days,
team_b_rest_days.

This replaces the small synthetic dataset used while building out the
backtesting pipeline's mechanics with real box scores, so accuracy numbers
downstream reflect actual games rather than fabricated placeholder data.

Data source: `nba_api`'s `leaguegamefinder.LeagueGameFinder` endpoint, which
returns one row per team per game (2 rows per game). This script pairs those
rows up by GAME_ID into one row per game: the home team (MATCHUP contains
"vs.") as team_a, the away team ("@") as team_b. `game_id` is kept so
`train_ml_model.py` can look up that specific game's box score (for the
star-availability feature); rest days are computed from each team's full
season game log (see compute_rest_days()) *before* any `--max-games`
truncation, so even the earliest game kept still has an accurate prior-game
date to compute rest from.

Usage:
    python fetch_real_nba_data.py                          # full 2024-25 regular season
    python fetch_real_nba_data.py --max-games 150           # most recent 150 games only
    python fetch_real_nba_data.py --season 2023-24
    python fetch_real_nba_data.py --season-type Playoffs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "historical_games.csv"
DEFAULT_SEASON = "2024-25"
DEFAULT_SEASON_TYPE = "Regular Season"
REQUEST_TIMEOUT_SECONDS = 30


class FetchError(Exception):
    pass


def fetch_team_game_log(season: str, season_type: str) -> pd.DataFrame:
    """One row per team per game (2 rows per game) for the given season."""
    resp = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",  # NBA
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return resp.get_data_frames()[0]


def compute_rest_days(team_games: pd.DataFrame) -> pd.DataFrame:
    """Adds a REST_DAYS column: full days off between a team's previous game
    and this one (0 = back-to-back, playing on consecutive calendar days).

    Must run on the *full* season's team-game log, not a --max-games-truncated
    slice, so a team's first tracked game still has its true previous game to
    diff against. A team's actual first game of the season (no previous game
    at all) has no ground truth to compute from, so it defaults to 2.0 (a
    normal-rest assumption -- reasonable for a season opener).
    """
    df = team_games.copy()
    df["GAME_DATE_PARSED"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE_PARSED"], kind="stable")
    prev_game_date = df.groupby("TEAM_ABBREVIATION")["GAME_DATE_PARSED"].shift(1)
    rest_days = (df["GAME_DATE_PARSED"] - prev_game_date).dt.days - 1
    df["REST_DAYS"] = rest_days.fillna(2.0).clip(lower=0)
    return df


def build_game_table(team_games: pd.DataFrame) -> pd.DataFrame:
    """Reduces the 2-rows-per-game team log into 1 row per game.

    MATCHUP is e.g. "MEM vs. DAL" (home) or "CHA @ BOS" (away); the home
    team's row becomes team_a, the away team's row becomes team_b. Games
    missing one side (e.g. an in-progress/malformed record) are dropped.
    Expects `team_games` to already carry a REST_DAYS column (see
    compute_rest_days()), computed from the *full* season log.
    """
    df = team_games.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ", regex=False)

    home = df[df["IS_HOME"]].set_index("GAME_ID")
    away = df[~df["IS_HOME"]].set_index("GAME_ID")

    common_ids = home.index.intersection(away.index)
    home = home.loc[common_ids]
    away = away.loc[common_ids]

    out = pd.DataFrame({
        "game_id": common_ids,
        "game_date": home["GAME_DATE"].values,
        "team_a": home["TEAM_ABBREVIATION"].values,
        "team_b": away["TEAM_ABBREVIATION"].values,
        "actual_score_a": home["PTS"].astype(int).values,
        "actual_score_b": away["PTS"].astype(int).values,
        "team_a_rest_days": home["REST_DAYS"].values,
        "team_b_rest_days": away["REST_DAYS"].values,
    }, index=common_ids)

    out["actual_winner"] = out["team_a"].where(
        out["actual_score_a"] > out["actual_score_b"], out["team_b"])

    out = out.sort_values("game_date", kind="stable").reset_index(drop=True)
    return out[["game_date", "game_id", "team_a", "team_b", "actual_winner",
                "actual_score_a", "actual_score_b", "team_a_rest_days", "team_b_rest_days"]]


def fetch_real_games(season: str, season_type: str, max_games: Optional[int]) -> pd.DataFrame:
    try:
        team_games = fetch_team_game_log(season, season_type)
    except Exception as e:
        raise FetchError(f"Could not reach the NBA stats API ({e})") from e

    if team_games.empty:
        raise FetchError(f"No games returned for {season} {season_type}.")

    team_games = compute_rest_days(team_games)
    games = build_game_table(team_games)
    if games.empty:
        raise FetchError(f"Fetched {len(team_games)} team-game rows but could not pair any into games.")

    if max_games is not None and len(games) > max_games:
        # Keep the most recent `max_games` games: newer games are also
        # closer to whatever roster snapshot /api/players currently serves,
        # which is what train_ml_model.py's features are computed from.
        games = games.tail(max_games).reset_index(drop=True)

    return games


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch real completed NBA games from nba_api into historical_games.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--season", default=DEFAULT_SEASON,
                         help=f"e.g. 2024-25 (default: {DEFAULT_SEASON})")
    parser.add_argument("--season-type", default=DEFAULT_SEASON_TYPE,
                         choices=["Regular Season", "Playoffs"],
                         help=f"default: {DEFAULT_SEASON_TYPE}")
    parser.add_argument("--max-games", type=int, default=None,
                         help="Keep only the most recent N games (default: all games in the season)")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV,
                         help=f"Output CSV path (default: {DEFAULT_CSV.name})")
    args = parser.parse_args(argv)

    print(f"Fetching {args.season} {args.season_type} game log from nba_api...")
    try:
        games = fetch_real_games(args.season, args.season_type, args.max_games)
    except FetchError as e:
        print(f"[fetch_real_nba_data] Fatal: {e}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.out, index=False)

    print(f"Wrote {len(games)} real games ({games['game_date'].min()} to {games['game_date'].max()}) "
          f"to {args.out}")
    print(f"Teams involved: {sorted(set(games['team_a']) | set(games['team_b']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
