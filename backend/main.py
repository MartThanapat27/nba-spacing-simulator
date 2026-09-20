from __future__ import annotations
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from . import api_simulation as sim
from .api_simulation import router as simulation_router

# Database connection setup
DB_URL = os.environ.get(
    "NBA_DB_URL", "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/nba_db"
)

engine = create_engine(DB_URL)
app = FastAPI(title="NBA Matchup & Spacing Simulator API")

# Permissive CORS so the static `index.html` dashboard (opened directly via
# file:// or served from any dev origin/port) can call this API's endpoints
# -- including POST /api/simulate -- from the browser without being blocked
# by the same-origin policy. This is a local-development/demo API with no
# cookie-based auth, so a wide-open policy here doesn't weaken anything that
# actually needs protecting; tighten `allow_origins` to a specific origin
# list before ever exposing this API beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router)


# Automatic Startup Ingestion -- proactively fetches and caches the latest
# real team OFF_RATING/DEF_RATING/NET_RATING/PACE (see
# api_simulation.prefetch_team_efficiency_stats()'s own docstring for the
# full fetch/cache/fallback mechanics) as soon as the server boots, so
# every /api/simulate request with enable_shrinkage=true immediately uses
# a warm, current baseline instead of the first caller paying nba_api's
# own latency (and risk of a cold-start failure). `on_event` is deprecated
# in favor of lifespan handlers as of recent FastAPI/Starlette versions,
# but still functions correctly on the version this project targets, and
# is a single, self-contained hook that doesn't require restructuring
# `app`'s construction around a lifespan context manager.
@app.on_event("startup")
def _startup_prefetch_team_efficiency_stats() -> None:
    sim.prefetch_team_efficiency_stats()

# Season used for the live team-defense lookup below -- should match whatever
# season historical_games.csv / train_ml_model.py / calibrate_engine.py are
# using, so the C++ engines' intrinsic defensive-resistance calibration (see
# cpp_engine/cuda_simulator.cu's kDefResistanceProbPerRating) lines up with
# the same real-world reference frame those were calibrated against.
NBA_SEASON = os.environ.get("NBA_SEASON", "2024-25")
TEAM_DEFENSE_CACHE_TTL_SECONDS = 3600.0
_team_defense_cache: dict = {"season": None, "data": None, "fetched_at": 0.0}

# Define Prior Weight for Bayesian Smoothing
PRIOR_WEIGHT = 50.0

# --- Recency weighting (exponential-decay-style blend) ---------------------
#
# /api/players' base query above is a flat FULL-SEASON average -- every game
# counts equally, whether it was game 1 or game 70. RECENCY_LAST_N_GAMES /
# RECENCY_WEIGHT blend in a REAL, live "last N games" snapshot from nba_api
# (which itself aggregates exactly those games, not a fabricated number) so
# recent form carries more weight than one flat season-long number, without
# requiring a new per-game log ingestion pipeline (this project only stores
# season-aggregate rows in `player_stats_raw`; true continuous per-game
# exponential decay would need that, and doesn't exist yet -- this two-tier
# "recent snapshot vs. season snapshot" blend is the honest, data-available
# approximation of it).
#
# RECENCY_WEIGHT is a deliberate design choice (not independently
# regression-fit the way kDefResistanceProbPerRating was): the last 20 games
# are roughly 1/4-1/3 of a season, so a naive equal-weighting would already
# give them ~25-30% influence -- 0.35 is a modest, intentional recency
# EMPHASIS on top of that, not an extreme override. A player with zero
# appearances in the last-N window (traded, season-ending injury, etc.)
# falls back to 100% season average -- this is not a bug or an oversight,
# it's the correct, honest behavior when there's simply no recent real data
# for that player to weight in.
RECENCY_LAST_N_GAMES = 20
RECENCY_WEIGHT = 0.35
RECENCY_CACHE_TTL_SECONDS = 3600.0
_recency_cache: dict = {"season": None, "data": None, "fetched_at": 0.0}

# SQL Query using JOIN to fetch real positions from the 'players' table
PLAYERS_QUERY = text(
    """
    WITH league_stat AS (
        SELECT SUM(fg3m) / NULLIF(SUM(fg3a), 0) AS avg_fg3_pct
        FROM player_stats_raw
    )
    SELECT 
        p.player_id, p.player_name, 
        pos.position, 
        pos.usage_rate,
        p.team_id, p.team_abbreviation, p.age, p.gp, p.min,
        p.fgm, p.fga, p.fg_pct, p.fg3m, p.fg3a,
        (p.fg3m + (l.avg_fg3_pct * :prior_weight)) / (p.fg3a + :prior_weight) AS fg3_pct,
        p.ftm, p.fta, p.ft_pct, p.oreb, p.dreb, p.reb, p.ast, p.tov, p.stl, p.blk, p.pf, p.pts, p.plus_minus
    FROM player_stats_raw p
    JOIN players pos ON p.player_id = pos.player_id
    CROSS JOIN league_stat l
    ORDER BY p.pts DESC
    """
)

class Player(BaseModel):
    player_id: int
    player_name: str
    position: str | None = None  
    usage_rate: float | None = None
    team_id: int | None = None
    team_abbreviation: str | None = None
    age: float | None = None
    gp: int
    min: float | None = None
    fgm: float | None = None
    fga: float | None = None
    fg_pct: float | None = None
    fg3m: float | None = None
    fg3a: float | None = None
    fg3_pct: float | None = None
    ftm: float | None = None
    fta: float | None = None
    ft_pct: float | None = None
    oreb: float | None = None
    dreb: float | None = None
    reb: float | None = None
    ast: float | None = None
    tov: float | None = None
    stl: float | None = None
    blk: float | None = None
    pf: float | None = None
    pts: float | None = None
    plus_minus: float | None = None

class TeamDefenseRating(BaseModel):
    team_abbreviation: str
    def_rating: float


def _fetch_team_defense_ratings(season: str) -> list[dict]:
    """Real, live team defensive ratings (points allowed per 100 possessions,
    season-to-date) via nba_api's leaguedashteamstats (Advanced) -- the same
    call train_ml_model.py / calibrate_engine.py use, duplicated here (rather
    than imported) so the backend has no dependency on the root-level ML
    scripts. TEAM_ID -> abbreviation via nba_api's static team list, since
    this endpoint doesn't return an abbreviation column directly.
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    resp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        timeout=30,
    )
    df = resp.get_data_frames()[0]
    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    df["team_abbreviation"] = df["TEAM_ID"].map(id_to_abbr)
    return [
        {"team_abbreviation": abbr, "def_rating": float(rating)}
        for abbr, rating in zip(df["team_abbreviation"], df["DEF_RATING"])
        if abbr is not None
    ]


def _fetch_recent_player_stats(season: str, last_n_games: int) -> dict[int, dict]:
    """Real, live per-player stats over just their last `last_n_games` real
    games (via nba_api's own last_n_games aggregation -- not a fabricated
    number), keyed by player_id. Two calls (Base for min/fg3a/fg3_pct,
    Advanced for usg_pct, which only the Advanced measure type reports) --
    the same leaguedashplayerstats endpoint train_ml_model.py/
    calibrate_engine.py already use elsewhere in this project, just scoped
    to a recent window here. usg_pct is returned by nba_api as a 0-1
    fraction; rescaled to the 0-100 convention `/api/players`' `usage_rate`
    column already uses so the blend below compares like units.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        last_n_games=last_n_games,
        timeout=30,
    ).get_data_frames()[0]
    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        last_n_games=last_n_games,
        timeout=30,
    ).get_data_frames()[0]

    usg_by_id = {int(pid): float(usg) * 100.0 for pid, usg in zip(adv["PLAYER_ID"], adv["USG_PCT"])}

    recent: dict[int, dict] = {}
    for row in base.itertuples():
        if row.GP <= 0:
            continue  # no real games in this window -- caller must fall back to season stats
        pid = int(row.PLAYER_ID)
        recent[pid] = {
            "min": float(row.MIN),
            "fg3a": float(row.FG3A),
            "fg3_pct": float(row.FG3_PCT) if row.FG3_PCT is not None else None,
            "usage_rate": usg_by_id.get(pid),
        }
    return recent


def _get_recent_player_stats_cached(season: str) -> dict[int, dict]:
    now = time.monotonic()
    stale = (
        _recency_cache["season"] != season
        or _recency_cache["data"] is None
        or now - _recency_cache["fetched_at"] > RECENCY_CACHE_TTL_SECONDS
    )
    if stale:
        try:
            data = _fetch_recent_player_stats(season, RECENCY_LAST_N_GAMES)
        except Exception:
            # Non-fatal, same convention as /api/team_defense: recency
            # weighting is an enhancement, not a hard dependency -- a failed
            # live fetch just means every player falls back to its plain
            # season average this request, not a broken /api/players.
            return _recency_cache["data"] or {}
        _recency_cache.update(season=season, data=data, fetched_at=now)
    return _recency_cache["data"]


def _apply_recency_weighting(row: dict, recent_by_id: dict[int, dict]) -> dict:
    """Blends `row` (a full-season /api/players record) with its real
    last-N-games snapshot, if one exists, at RECENCY_WEIGHT. `min`/`fg3a`
    are season-CUMULATIVE totals in `row` (per this project's existing,
    documented convention -- the C++ engine divides them by `gp` itself),
    so the blend happens at the PER-GAME level and is re-multiplied by the
    player's real season `gp` before returning, preserving that exact
    contract unchanged -- no changes needed anywhere downstream (the C++
    engine, index.html's roster editor) to consume an already-recency-aware
    `/api/players` response.
    """
    recent = recent_by_id.get(row.get("player_id"))
    if not recent:
        return row  # no real recent games for this player -- honest no-op

    # The DB driver returns numeric columns as decimal.Decimal, not float --
    # cast explicitly so arithmetic below never mixes the two (Decimal *
    # float raises TypeError).
    gp = float(row.get("gp") or 1)
    row_min = float(row.get("min") or 0.0)
    row_fg3a = float(row.get("fg3a") or 0.0)
    row_fg3_pct = row.get("fg3_pct")
    row_usage_rate = row.get("usage_rate")

    blended = dict(row)

    season_min_per_game = row_min / gp
    blended_min_per_game = RECENCY_WEIGHT * recent["min"] + (1.0 - RECENCY_WEIGHT) * season_min_per_game
    blended["min"] = blended_min_per_game * gp

    season_fg3a_per_game = row_fg3a / gp
    blended_fg3a_per_game = RECENCY_WEIGHT * recent["fg3a"] + (1.0 - RECENCY_WEIGHT) * season_fg3a_per_game
    blended["fg3a"] = blended_fg3a_per_game * gp

    if recent.get("fg3_pct") is not None and row_fg3_pct is not None:
        blended["fg3_pct"] = RECENCY_WEIGHT * recent["fg3_pct"] + (1.0 - RECENCY_WEIGHT) * float(row_fg3_pct)

    if recent.get("usage_rate") is not None and row_usage_rate is not None:
        blended["usage_rate"] = RECENCY_WEIGHT * recent["usage_rate"] + (1.0 - RECENCY_WEIGHT) * float(row_usage_rate)

    return blended


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "nba-spacing-simulator-api"}


@app.get("/api/team_defense", response_model=list[TeamDefenseRating])
def get_team_defense(season: str = NBA_SEASON) -> list[dict]:
    """Real team defensive ratings for `season`, used by cuda_simulator.exe/
    simulator.exe to intrinsically calibrate defensive resistance per
    possession (see cpp_engine/cuda_simulator.cu's kDefResistanceProbPerRating
    and main.cpp's PossessionEngine) -- no external --ml-margin override
    needed for this effect. Cached in-process for
    TEAM_DEFENSE_CACHE_TTL_SECONDS since nba_api is a slow external call and
    a team's season-to-date defensive rating doesn't meaningfully change
    within that window.
    """
    now = time.monotonic()
    stale = (
        _team_defense_cache["season"] != season
        or _team_defense_cache["data"] is None
        or now - _team_defense_cache["fetched_at"] > TEAM_DEFENSE_CACHE_TTL_SECONDS
    )
    if stale:
        try:
            data = _fetch_team_defense_ratings(season)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"nba_api error: {exc}") from exc
        _team_defense_cache.update(season=season, data=data, fetched_at=now)
    return _team_defense_cache["data"]

@app.get("/api/players", response_model=list[Player])
def get_players(recency_weighted: bool = True) -> list[dict]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                PLAYERS_QUERY, {"prior_weight": PRIOR_WEIGHT}
            ).mappings().all()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"DB Error: {exc}"
        ) from exc

    players = [dict(row) for row in rows]
    if not recency_weighted:
        return players

    # Blend each player's flat full-season average with a real, live
    # last-RECENCY_LAST_N_GAMES-games snapshot (see _apply_recency_weighting's
    # docstring) so recent real form carries more weight than one flat
    # season-long number. A player with no appearances in that window
    # (season-ending injury, very recent trade, etc.) is returned unchanged
    # -- there's no real recent data to blend in for them, so their honest
    # full-season average is exactly what should come back.
    recent_by_id = _get_recent_player_stats_cached(NBA_SEASON)
    return [_apply_recency_weighting(p, recent_by_id) for p in players]