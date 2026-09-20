"""
fetch_nba_data.py -- Data pipeline for the NBA Matchup & Spacing Simulator.

Pulls current NBA teams and current-season players (with shooting / usage stats)
from nba_api, reshapes them with pandas to match our PostgreSQL schema, and
upserts the result into the `teams` and `players` tables.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import leaguedashplayerstats, playerindex, commonteamroster

DB_URL = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/nba_db"
DEFAULT_SEASON = "2024-25"

REQUEST_PAUSE_SECONDS = 0.7

PCT_FLOOR, PCT_CEIL = 0.300, 0.450
VOL_CEIL = 9.0

POSITION_DEFAULTS = {
    "G":   (76, 195),
    "G-F": (78, 210),
    "F-G": (78, 210),
    "F":   (80, 225),
    "F-C": (82, 245),
    "C-F": (82, 245),
    "C":   (83, 250),
}
DEFAULT_PHYSICAL = (79, 215)


def _height_to_inches(value) -> "float | None":
    if not isinstance(value, str) or "-" not in value:
        return None
    feet, _, inches = value.partition("-")
    try:
        return int(feet) * 12 + int(inches)
    except ValueError:
        return None


def fetch_teams() -> pd.DataFrame:
    df = pd.DataFrame(static_teams.get_teams())
    df = df.rename(columns={"id": "team_id", "full_name": "team_name"})
    df = df[["team_id", "team_name", "abbreviation"]].copy()
    df["team_id"] = df["team_id"].astype(int)
    df["team_name"] = df["team_name"].str.strip()
    df["abbreviation"] = df["abbreviation"].str.strip().str.upper()
    return df.drop_duplicates(subset="team_id").reset_index(drop=True)


def _league_dash(season: str, measure_type: str) -> pd.DataFrame:
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure_type,
        season_type_all_star="Regular Season",
    )
    time.sleep(REQUEST_PAUSE_SECONDS)
    return resp.get_data_frames()[0]


def get_all_true_positions(season: str) -> pd.DataFrame:
    team_ids = [
        1610612737, 1610612738, 1610612739, 1610612740, 1610612741,
        1610612742, 1610612743, 1610612744, 1610612745, 1610612746,
        1610612747, 1610612748, 1610612749, 1610612750, 1610612751,
        1610612752, 1610612753, 1610612754, 1610612755, 1610612756,
        1610612757, 1610612758, 1610612759, 1610612760, 1610612761,
        1610612762, 1610612763, 1610612764, 1610612765, 1610612766
    ]
    
    roster_list = []
    print("Fetching actual player positions from team rosters...")
    
    for team_id in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = roster.get_data_frames()[0]
            roster_list.append(roster_df[['PLAYER_ID', 'POSITION']])
            time.sleep(REQUEST_PAUSE_SECONDS)
        except Exception as e:
            print(f"Error fetching roster for team {team_id}: {e}")
            
    if not roster_list:
        return pd.DataFrame(columns=["player_id", "TRUE_POSITION"])
        
    master_roster = pd.concat(roster_list, ignore_index=True)
    master_roster.columns = ['player_id', 'TRUE_POSITION']
    return master_roster.drop_duplicates(subset="player_id").reset_index(drop=True)


def fetch_players(season: str) -> pd.DataFrame:
    base = _league_dash(season, "Base").rename(
        columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "name", "TEAM_ID": "team_id"}
    )
    base = base[["player_id", "name", "team_id", "GP", "FG3_PCT", "FG3A"]]

    adv = _league_dash(season, "Advanced").rename(columns={"PLAYER_ID": "player_id"})
    adv = adv[["player_id", "USG_PCT"]]

    index = playerindex.PlayerIndex(season=season, league_id="00").get_data_frames()[0]
    time.sleep(REQUEST_PAUSE_SECONDS)
    index = index.rename(columns={"PERSON_ID": "player_id", "TEAM_ID": "index_team_id"})
    index = index[["player_id", "HEIGHT", "WEIGHT", "index_team_id"]]

    df = base.merge(adv, on="player_id", how="left").merge(index, on="player_id", how="left")

    team_id = pd.to_numeric(df["team_id"], errors="coerce").fillna(0)
    index_team_id = pd.to_numeric(df["index_team_id"], errors="coerce").fillna(0)
    df["team_id"] = team_id.where(team_id != 0, index_team_id).astype("int64")
    df = df.drop(columns=["index_team_id"])

    # Merge true positions from CommonTeamRoster lookup
    true_positions_df = get_all_true_positions(season)
    df = df.merge(true_positions_df, on="player_id", how="left")

    return df


def map_players(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["player_id"] = df["player_id"].astype(int)
    out["name"] = df["name"].str.strip()
    out["team_id"] = df["team_id"].astype(int)

    # --- position -------------------------------------------------------- #
    pos = (
        df["TRUE_POSITION"].astype("string").str.strip().str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    )
    out["position"] = pos.fillna("F")

    # --- height / weight, with position-based fallbacks ----------------- #
    height = pd.to_numeric(df["HEIGHT"].map(_height_to_inches), errors="coerce")
    weight = pd.to_numeric(df["WEIGHT"], errors="coerce")

    fallback = out["position"].map(lambda p: POSITION_DEFAULTS.get(p, DEFAULT_PHYSICAL))
    out["height_inches"] = height.fillna(fallback.map(lambda t: t[0])).round().astype(int)
    out["weight_lbs"] = weight.fillna(fallback.map(lambda t: t[1])).round().astype(int)

    # --- shooting / usage ---------------------------------------------- #
    pct = pd.to_numeric(df["FG3_PCT"], errors="coerce")
    fg3a = pd.to_numeric(df["FG3A"], errors="coerce")
    out["three_point_pct"] = pct.round(3)
    out["usage_rate"] = (pd.to_numeric(df["USG_PCT"], errors="coerce") * 100).round(1)

    # --- spacing impact (0-100) -------------------------------------- #
    pct_norm = ((pct.fillna(0.0) - PCT_FLOOR) / (PCT_CEIL - PCT_FLOOR)).clip(0.0, 1.0)
    vol_norm = (fg3a.fillna(0.0) / VOL_CEIL).clip(0.0, 1.0)
    base_score = 100.0 * (0.60 * pct_norm + 0.40 * vol_norm)
    gravity_bonus = 12.0 * pct_norm * vol_norm
    out["spacing_impact_score"] = (base_score + gravity_bonus).clip(0.0, 100.0).round(2)

    out["perimeter_defense_score"] = None

    out = out[
        [
            "player_id", "name", "team_id", "position",
            "height_inches", "weight_lbs", "three_point_pct",
            "usage_rate", "spacing_impact_score", "perimeter_defense_score",
        ]
    ]
    return out.dropna(subset=["name", "team_id"]).drop_duplicates(subset="player_id").reset_index(drop=True)


def upsert(engine, table_name: str, df: pd.DataFrame, pk: str) -> int:
    if df.empty:
        print(f"  ! nothing to write for {table_name}")
        return 0

    table = Table(table_name, MetaData(), autoload_with=engine)
    records = [
        {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]

    stmt = pg_insert(table).values(records)
    update_cols = {c.name: stmt.excluded[c.name] for c in table.columns if c.name != pk}
    stmt = stmt.on_conflict_do_update(index_elements=[pk], set_=update_cols)

    with engine.begin() as conn:
        conn.execute(stmt)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=DEFAULT_SEASON, help="e.g. 2024-25")
    parser.add_argument("--db-url", default=DB_URL)
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        print(f"ERROR: could not connect to the database: {exc}")
        return 1

    print("Fetching teams from nba_api ...")
    teams_df = fetch_teams()
    print(f"  fetched {len(teams_df)} teams")

    print(f"Fetching {args.season} players and positions from nba_api ...")
    players_df = map_players(fetch_players(args.season))
    print(f"  fetched {len(players_df)} players")

    known_teams = set(teams_df["team_id"])
    before = len(players_df)
    players_df = players_df[players_df["team_id"].isin(known_teams)].reset_index(drop=True)
    if before != len(players_df):
        print(f"  dropped {before - len(players_df)} players with an unrecognized team_id")

    n_teams = upsert(engine, "teams", teams_df, "team_id")
    print(f"SUCCESS: upserted {n_teams} rows into 'teams'")

    n_players = upsert(engine, "players", players_df, "player_id")
    print(f"SUCCESS: upserted {n_players} rows into 'players'")

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())