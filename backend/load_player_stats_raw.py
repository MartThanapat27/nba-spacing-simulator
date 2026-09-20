"""
load_player_stats_raw.py -- One-time loader for the eda_dashboard.py raw stats table.

The `players` table in nba_db only carries the columns the spacing simulator
needs (three_point_pct, usage_rate, ...). The EDA dashboard wants the raw
per-player box-score columns from nba_api's LeagueDashPlayerStats "Base"
endpoint (FG3_PCT, PLUS_MINUS, FG3A, PTS, MIN, GP, ...), which aren't in the
DB yet. This script fetches that data (or reuses the existing preview CSV)
and loads it into a new `player_stats_raw` table so the dashboard can just
query Postgres.

Usage:
    python load_player_stats_raw.py --season 2024-25
    python load_player_stats_raw.py --csv nba_base_preview.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "NBA_DB_URL", "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/nba_db"
)
DEFAULT_SEASON = "2024-25"
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "nba_base_preview.csv")

COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
    "GP", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV",
    "STL", "BLK", "PF", "PTS", "PLUS_MINUS",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS player_stats_raw (
    player_id          INTEGER PRIMARY KEY,
    player_name        VARCHAR(100) NOT NULL,
    team_id            BIGINT,
    team_abbreviation  CHAR(3),
    age                NUMERIC(4, 1),
    gp                 INT,
    min                NUMERIC(7, 2),
    fgm                NUMERIC(6, 1),
    fga                NUMERIC(6, 1),
    fg_pct             NUMERIC(5, 3),
    fg3m               NUMERIC(6, 1),
    fg3a               NUMERIC(6, 1),
    fg3_pct            NUMERIC(5, 3),
    ftm                NUMERIC(6, 1),
    fta                NUMERIC(6, 1),
    ft_pct             NUMERIC(5, 3),
    oreb               NUMERIC(6, 1),
    dreb               NUMERIC(6, 1),
    reb                NUMERIC(6, 1),
    ast                NUMERIC(6, 1),
    tov                NUMERIC(6, 1),
    stl                NUMERIC(6, 1),
    blk                NUMERIC(6, 1),
    pf                 NUMERIC(6, 1),
    pts                NUMERIC(7, 1),
    plus_minus         NUMERIC(7, 1)
);
"""


def fetch_from_api(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerstats

    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, measure_type_detailed_defense="Base"
    )
    return resp.get_data_frames()[0]


def load_source(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        print(f"Reading {args.csv} ...")
        return pd.read_csv(args.csv)
    print(f"Fetching {args.season} Base stats from nba_api ...")
    return fetch_from_api(args.season)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].copy()
    df.columns = [c.lower() for c in df.columns]
    df["player_id"] = df["player_id"].astype(int)
    df["gp"] = df["gp"].astype(int)
    return df.drop_duplicates(subset="player_id").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=DEFAULT_SEASON, help="e.g. 2024-25")
    parser.add_argument(
        "--csv",
        nargs="?",
        const=DEFAULT_CSV,
        default=None,
        help="Load from a CSV (defaults to nba_base_preview.csv) instead of calling nba_api",
    )
    parser.add_argument("--db-url", default=DB_URL)
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 -- surface any connection failure plainly
        print(f"ERROR: could not connect to the database: {exc}")
        print("Is the Postgres container up?  ->  docker compose up -d")
        return 1
    print(f"Connected to {args.db_url.rsplit('@', 1)[-1]}")

    df = clean(load_source(args))
    print(f"  loaded {len(df)} players")

    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text("TRUNCATE TABLE player_stats_raw"))
    df.to_sql("player_stats_raw", engine, if_exists="append", index=False)
    print(f"SUCCESS: wrote {len(df)} rows into 'player_stats_raw'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
