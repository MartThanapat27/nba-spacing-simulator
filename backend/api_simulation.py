"""REST API wrapper around the compiled C++/CUDA simulation engines.

Bridges `POST /api/simulate` to two executables built by `cpp_engine/`:

  * **`cuda_simulator.exe`** (`mode="gpu"`) -- 100,000-game parallel GPU Monte
    Carlo batch. Returns aggregate win probabilities, average scores, and
    point-differential stats. Supports every advanced-analytics CLI flag
    this project has built up (`--ml-margin`, `--hot-hand-boost`,
    `--home-a`/`--home-b`, `--b2b-a`/`--b2b-b`) plus roster customization
    (`--custom-roster`, `--injured-a`/`--injured-b`).
  * **`simulator.exe`** (`mode="cpu"`) -- a single stochastic, narrated,
    possession-by-possession game. Returns the full play-by-play text and
    final score. Supports roster customization (`--custom-roster`,
    `--injured-a`/`--injured-b`) and the engine's *intrinsic*, data-calibrated
    effects (`--home-a`/`--home-b`, `--b2b-a`/`--b2b-b`, and always-on
    defensive resistance -- see cpp_engine/calibrated_constants.h), since
    those are now baked into simulator.exe's own possession loop too, not
    just the GPU kernel. It does *not* support `ml_margin`/`enable_big_match`
    (those remain GPU-kernel-only external overrides) -- toggles that don't
    apply in this mode are reported back in the response's `warnings` list
    rather than silently ignored or erroring.

Design notes
------------
* **No shell involved.** Every subprocess call passes a Python list to
  `subprocess.run` (never `shell=True`, never a concatenated string), so
  there is no shell-injection surface regardless of what a caller submits
  as a team name, player name, or roster payload.
* **No caller-controlled filesystem paths.** `roster_type="custom"` writes
  the submitted roster to a server-generated temporary file (via
  `tempfile`) and passes *that* path to `--custom-roster`; a caller can
  never supply an arbitrary path for the engine to read.
* **No live retraining on the request path.** `enable_injuries` here means
  "exclude these named players from the roster" (a direct, immediate
  `--injured-a`/`--injured-b` roster edit) -- not the offline
  `train_ml_model.py` pipeline's box-score-driven auto-detection, which
  takes ~90 seconds and is unsuitable for a synchronous HTTP request. If
  you want that, run `train_ml_model.py` / `backtest_model.py` separately
  and pass their result in via `ml_margin`.
* **Executable + team abbreviations are the only inputs read from disk/
  network beyond the request body** -- the executable path is always
  resolved server-side from a fixed candidate list, never from the request.
* **Custom rosters carry real def_rating too, when available.** `CustomRoster.def_rating`
  is optional team-level metadata (a real, live `/api/team_defense` value,
  supplied by the caller -- typically index.html's roster editor, which
  fetches it alongside a team's real players) written into the
  `--custom-roster` JSON as `team_a_def_rating`/`team_b_def_rating`. Without
  it, a custom roster's defensive-resistance effect silently goes neutral
  even when every player on it is real -- this keeps a "trade a real
  player" custom run just as accurate as a `roster_type="real"` run of the
  same team. Omit it for a genuinely fictional/fantasy team.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXECUTABLE_CANDIDATES = {
    "gpu": [
        PROJECT_ROOT / "cpp_engine" / "build" / "Release" / "cuda_simulator.exe",
        PROJECT_ROOT / "cpp_engine" / "build" / "Debug" / "cuda_simulator.exe",
        PROJECT_ROOT / "cpp_engine" / "build" / "Release" / "cuda_simulator",
        PROJECT_ROOT / "cpp_engine" / "build" / "Debug" / "cuda_simulator",
    ],
    "cpu": [
        PROJECT_ROOT / "cpp_engine" / "build" / "Release" / "simulator.exe",
        PROJECT_ROOT / "cpp_engine" / "build" / "Debug" / "simulator.exe",
        PROJECT_ROOT / "cpp_engine" / "build" / "Release" / "simulator",
        PROJECT_ROOT / "cpp_engine" / "build" / "Debug" / "simulator",
    ],
}
ENGINE_NAMES = {"gpu": "cuda_simulator", "cpu": "simulator"}

# Hybrid ML layer (see /api/simulate-ml below): artifacts produced offline by
# `train_ml_model.py` (repo root) -- this backend never trains anything
# on the request path, only loads what's already saved here.
ML_MODEL_DIR = PROJECT_ROOT / "ml_model"

# GPU batch mode ("2") always -- Mode A wants aggregate stats, which only the
# GPU MONTE CARLO BATCH RESULTS block (parsed below) provides; the CPU batch
# block cuda_simulator.exe also prints in this mode is not parsed/returned.
GPU_SIM_MODE_ARG = "2"

# --- stdout parsing -----------------------------------------------------
# The simulator prints a CPU batch summary and then a GPU batch summary with
# near-identical field labels. Anchoring on this marker and slicing the GPU
# section off the tail of stdout keeps the regexes below from ever picking
# up the CPU engine's numbers by mistake (same approach as backtest_model.py).
GPU_BLOCK_MARKER = "GPU MONTE CARLO BATCH RESULTS"
SIM_COUNT_RE = re.compile(r"Simulations run \(threads\)\s*:\s*(\d+)")
TIE_RE = re.compile(r"Ties\s*:\s*([\d.]+)%")
MARGIN_RE = re.compile(r"Avg point differential[^:]*:\s*([-\d.]+)")
STDDEV_RE = re.compile(r"Point differential std dev\s*:\s*([\d.]+)")

TEAM_NAME_RE = re.compile(r"^[A-Za-z0-9 .'\-]{1,40}$")
PLAYER_NAME_RE = re.compile(r"^[A-Za-z0-9 .'\-]{1,60}$")


class SimulationError(Exception):
    """Any problem resolving the executable or running the simulation."""


# --- Layer 2: Bayesian Shrinkage / Market Prior Safety Brake --------------
#
# The C++/CUDA engines (Layer 1: shared game-clock possession model, see
# cuda_simulator.cu's "Shared Game Clock" comment block) reduce the engine's
# excess point-differential variance, but don't fully close the gap to real
# NBA margin variance (see this project's own audit -- current point-
# differential std dev is still somewhat above the ~12-13 real-NBA
# benchmark). A too-wide margin distribution silently overstates win-
# probability confidence for a genuine mismatch (an underdog can read as
# ~10% when a real market/analytics prior would have them closer to
# 20-25%). This is a post-processing SAFETY BRAKE against that specific
# symptom -- independent of, and a backstop for, whatever the raw engine's
# variance turns out to be -- not a replacement for continuing to improve
# the underlying possession model.
#
# IMPORTANT -- this is a fixed-weight convex blend toward a market-style
# prior ("shrinkage" in the colloquial sports-analytics sense), NOT formal
# Bayesian conjugate updating over the 100,000 simulated games' win/loss
# counts. Treating those 100,000 games as 100,000 independent real-world
# Bernoulli observations and doing a literal posterior update would make the
# prior's influence vanish (the sample size is enormous), which would
# defeat the point -- they're correlated draws from ONE possibly-
# miscalibrated model, not independent real observations, so a literal
# count-based Bayesian update is the wrong tool here. A fixed blend weight,
# honestly labeled as such, is the correct and honest mechanism.
def resolve_current_nba_season(today: Optional[datetime.date] = None) -> str:
    """The real NBA season label flips over in October (regular-season
    tip-off) -- from July through September the "current"/"active" season
    is still the one that just finished its Finals in June, not the one
    about to start. Used as NBA_SEASON_DEFAULT's own default below so the
    live team-efficiency prefetch always targets the latest REAL completed
    (or in-progress) season without a hardcoded year string that would
    otherwise need a manual edit every year. `NBA_SEASON` env var still
    overrides this outright, for a specific historical season or a
    deliberate pin.
    """
    today = today or datetime.date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


NBA_SEASON_DEFAULT = os.environ.get("NBA_SEASON", resolve_current_nba_season())
TEAM_EFFICIENCY_CACHE_TTL_SECONDS = 3600.0
_team_efficiency_cache: dict = {"season": None, "data": None, "fetched_at": 0.0}

# Automatic Startup Ingestion -- local, on-disk fallback snapshot of the
# last successful live fetch (see prefetch_team_efficiency_stats() below).
# Lives in ml_model/ (this project's existing "generated cache data"
# directory, already used for star_availability_cache.json) rather than a
# new top-level location.
TEAM_EFFICIENCY_FALLBACK_PATH = PROJECT_ROOT / "ml_model" / "team_efficiency_cache.json"

# If nba_api was unreachable at startup and the on-disk fallback had to be
# used, the in-memory cache is still marked "fetched" (so every request
# doesn't independently retry a live fetch that's likely to fail again
# immediately) but with an artificially shortened effective TTL, so the
# server self-heals and picks up live data again a few minutes later
# rather than being stuck on a stale snapshot for the full TTL window.
FALLBACK_RETRY_SECONDS = 300.0

# Real NBA margin std dev benchmark (see cpp_engine/cuda_simulator.cu's
# variance-audit comments) -- used to convert a possession-model-implied
# point margin into a win probability via the normal CDF, the same way real
# sports-betting/analytics models translate a point spread into a win%.
# Anchored to the commonly-cited real range, not independently regression-
# fit against this project's own data yet.
kRealMarginStdDev = 12.0

# League-average pace fallback (possessions per 48 minutes) -- only used if
# a team's real PACE is somehow missing from the fetched stats. Matches
# cuda_simulator.cu's kPossessionsPerTeam target.
kLeagueAvgPace = 100.0

# Real, commonly-cited NBA home-court point advantage -- folded into the
# structural prior's expected margin (not the raw sim, which applies its own
# intrinsic home-court effect -- see cpp_engine's kHomeCourtPointsOverride in
# main.cpp/cuda_simulator.cu) so enabling home_team doesn't silently go
# unrepresented in the prior half of the blend. Retuned from 3.0 to 2.25 --
# a modern regular-season home-court edge runs closer to 2-2.5 points than
# the older ~3-point "textbook" figure -- and kept numerically identical to
# the C++ engine's own kHomeCourtPointsOverride so both halves of the blend
# agree on this one real-world number instead of quietly disagreeing.
kHomeCourtPointsPrior = 2.25

# Real, calibrated points-of-margin effect per unit of "rest_advantage"
# (team_a_rest_days - team_b_rest_days), hand-mirrored from
# calibrate_engine.py's OLS fit against historical_games.csv -- see
# cpp_engine/calibrated_constants.h's header comment for the full
# coefficient table this is copied from (+1.1839, t=2.78, statistically
# significant at n=1225 games; same convention as kHomeCourtPointsPrior
# above). This is the SAME real-world schedule-fatigue effect the C++
# engine itself applies intrinsically via --b2b-a/--b2b-b
# (kFatigueProbPerRestDay is that exact coefficient converted to the
# kernel's probability space) -- reused here in points-of-margin space so
# the Hybrid ML Layer's own DISPLAYED prediction (/api/simulate-ml) reacts
# to the Fatigue toggle too, the same way compute_market_prior() folds
# home-court into its own parallel signal without feeding it back into
# --ml-margin -- see apply_contextual_ml_adjustment()'s docstring.
kRestAdvantageMarginPerDay = 1.1839

# Dynamic Blending Weight (Context-Aware Safety Brake) -- how much of the
# blend comes from the raw sim vs. the structural market prior.
#
# A flat weight applies the same pull regardless of how lopsided the
# MATCHUP ITSELF actually is -- exactly backwards from what a guardrail
# should do, and it can't tell a genuinely close game from a blowout. This
# scales the weight ON THE RAW SIM based on the real, absolute NET_RATING
# GAP between the two teams (delta_net_rating -- an EXOGENOUS, real-data
# signal read directly off each team's live season NET_RATING, not the
# sim's own output, and not the pace/home-court-adjusted projected margin
# compute_market_prior() separately computes for the prior PROBABILITY
# itself -- this is deliberately the simpler, more directly interpretable
# "how far apart are these two teams on paper" number):
#
#   * CLOSE matchups (delta_net_rating small, e.g. two playoff-caliber
#     teams): the raw engine's per-possession decision tree treats each of
#     a game's ~100 team possessions as an effectively independent
#     Bernoulli draw, so even a genuinely modest real efficiency edge
#     compounds, via the Central Limit Theorem, into a far more separated
#     aggregate win probability than real single-game NBA variance ever
#     shows. The structural prior's simpler, single-number (Pace +
#     OffRtg/DefRtg) projection does NOT have this compounding problem, so
#     it gets the LARGER share of the blend here (MIN_RAW_WEIGHT_CLOSE_MATCHUP
#     on the raw sim -- i.e. the prior dominates) -- pulling a falsely
#     lopsided raw split back into a realistic, competitive band.
#   * BLOWOUT mismatches (delta_net_rating large): a real, massive
#     efficiency gap is exactly the case where the raw engine's detailed,
#     real player-level structural modeling (archetypes, defense, rim
#     protection, foul trouble, etc.) has the most genuine signal to work
#     with, and the prior's crude two-number projection is more likely to
#     UNDERSTATE just how one-sided a true blowout is. So the raw sim's
#     share of the blend TAPERS UP toward MAX_RAW_WEIGHT_BLOWOUT as the gap
#     grows, letting the raw engine's own structural dominance govern the
#     result instead of a guardrail flattening a genuine mismatch.
#
# This is the opposite curve SHAPE from an earlier version of this
# mechanism (which shrunk MORE toward the prior as the gap grew, on the
# theory that a guardrail should distrust the raw sim most exactly where
# it could go most wrong) -- both are defensible adaptive-shrinkage
# philosophies; this one is deliberately chosen so a real blowout's
# structural talent gap isn't itself flattened by the safety brake, while
# still catching the specific failure mode this brake exists for: a small
# real edge compounding into a falsely extreme raw split for a genuinely
# close matchup. Neither curve is independently regression-fit against
# this project's own data -- the bounds/full_dominance_gap below are
# deliberately chosen, tunable constants. `shrinkage_weight` on the
# request sets the FLOOR (the raw sim's weight for a ~0-point
# delta_net_rating matchup); the ceiling and the gap scale are fixed here,
# not request-configurable, to keep the API surface simple.
MIN_RAW_WEIGHT_CLOSE_MATCHUP = 0.40   # weight on raw sim for a ~0 delta_net_rating matchup (prior dominates, ~60%)
MAX_RAW_WEIGHT_BLOWOUT = 0.85         # weight on raw sim at/beyond a genuinely massive net-rating gap (raw dominates, ~85%)
FULL_RAW_DOMINANCE_NET_RATING_GAP = 10.0  # abs(NET_RATING_a - NET_RATING_b) at which the raw sim's weight tops out
DEFAULT_SHRINKAGE_WEIGHT = MIN_RAW_WEIGHT_CLOSE_MATCHUP


def compute_dynamic_raw_weight(delta_net_rating_abs: float, min_weight: float = MIN_RAW_WEIGHT_CLOSE_MATCHUP,
                                max_weight: float = MAX_RAW_WEIGHT_BLOWOUT,
                                full_dominance_gap: float = FULL_RAW_DOMINANCE_NET_RATING_GAP) -> float:
    """How much weight the blend gives the raw simulation, scaled UP (more
    trust in the raw engine's own structural modeling, less in the
    structural prior) as the real, absolute NET_RATING gap between the two
    teams grows -- i.e. how genuinely mismatched the matchup is on paper,
    an exogenous signal independent of what the raw sim happened to
    output. 0 gap -> min_weight (a close matchup -- the prior's simpler,
    better-calibrated single-number model is trusted MORE, since the raw
    engine's per-possession compounding is most likely to overstate a
    genuinely close real talent gap into a falsely lopsided split);
    >= full_dominance_gap -> max_weight (a genuine blowout -- the raw
    engine's detailed structural modeling is trusted to capture the real
    talent gap better than the prior's simple two-number projection).
    """
    min_weight = min(min_weight, max_weight)  # defensive: never invert the range
    t = abs(delta_net_rating_abs) / full_dominance_gap if full_dominance_gap > 0 else 1.0
    t = max(0.0, min(1.0, t))
    return min_weight + t * (max_weight - min_weight)


def _fetch_team_efficiency_stats(season: str, date_from: Optional[str] = None,
                                  date_to: Optional[str] = None) -> dict[str, dict]:
    """Real OFF_RATING/DEF_RATING/PACE per team (points scored/allowed per
    100 possessions, and possessions per 48 minutes) via nba_api's
    leaguedashteamstats (Advanced) -- the same underlying call
    backend/main.py's _fetch_team_defense_ratings() uses, duplicated here
    (not imported) because api_simulation.py has no dependency on main.py
    (avoids a circular import -- main.py is the one that imports THIS
    module's router), matching the project's existing precedent for this
    exact tradeoff (see _fetch_team_defense_ratings' own docstring in
    main.py). This is what powers the structural (Pace + OffRtg/DefRtg)
    market prior -- see compute_market_prior()'s docstring for how these
    three real numbers become a projected score/margin/win probability.

    `date_from`/`date_to` (both "MM/DD/YYYY", nba_api's own expected format
    for these two params -- NOT ISO) restrict the aggregate to games played
    within that real date range, INCLUSIVE of both ends, instead of the
    full season: this is what makes a genuine POINT-IN-TIME snapshot
    possible (e.g. "every team's real rolling ORTG/DRTG as of the eve of
    a specific historical game," not today's live end-of-season number) --
    see backtest_model.py's fetch_point_in_time_efficiency_stats(), the
    only caller that passes these. Both default to None, which asks
    nba_api for the plain full-season aggregate -- the live
    enable_shrinkage request path's existing behavior, unchanged.
    A team with ZERO real games in the requested range is simply ABSENT
    from nba_api's response (not returned with a NaN/0 row) -- callers
    must handle a missing key themselves; `GP` is included in the
    returned dict specifically so a caller can also flag a THIN (but
    nonzero) real sample if it wants to.
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    resp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        date_from_nullable=date_from or "",
        date_to_nullable=date_to or "",
        timeout=30,
    )
    df = resp.get_data_frames()[0]
    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    df["team_abbreviation"] = df["TEAM_ID"].map(id_to_abbr)
    return {
        abbr: {
            "off_rating": float(off_rtg),
            "def_rating": float(def_rtg),
            "net_rating": float(net_rtg),
            "pace": float(pace),
            "gp": int(gp),
        }
        for abbr, off_rtg, def_rtg, net_rtg, pace, gp in zip(
            df["team_abbreviation"], df["OFF_RATING"], df["DEF_RATING"], df["NET_RATING"], df["PACE"], df["GP"]
        )
        if abbr is not None
    }


def _load_team_efficiency_fallback_file() -> Optional[dict]:
    """Reads the on-disk snapshot written by the last successful
    prefetch_team_efficiency_stats() call, if any -- the LOCAL FALLBACK
    Automatic Startup Ingestion's error-handling requirement asks for: even
    if nba_api is completely unreachable at boot, a server that has ever
    booted successfully before still has real (if possibly stale) data to
    serve immediately, instead of a cold, empty cache that fails every
    enable_shrinkage request until nba_api recovers. Returns None (not an
    empty dict) for "no usable fallback exists yet" vs. "fallback exists
    but covers zero teams", so callers can tell the two apart.
    """
    if not TEAM_EFFICIENCY_FALLBACK_PATH.is_file():
        return None
    try:
        with TEAM_EFFICIENCY_FALLBACK_PATH.open(encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("data")
        return data if isinstance(data, dict) and data else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_team_efficiency_fallback_file(season: str, data: dict[str, dict]) -> None:
    """Persists a successful live fetch to disk so it survives a server
    restart -- best-effort only (a write failure here -- read-only
    filesystem, disk full, etc. -- must never take down a request that
    otherwise succeeded), see prefetch_team_efficiency_stats().
    """
    payload = {
        "season": season,
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "data": data,
    }
    try:
        TEAM_EFFICIENCY_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TEAM_EFFICIENCY_FALLBACK_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        print(f"[api_simulation] Warning: could not write team efficiency fallback file "
              f"({TEAM_EFFICIENCY_FALLBACK_PATH}): {e}")


def prefetch_team_efficiency_stats(season: str = NBA_SEASON_DEFAULT) -> dict[str, dict]:
    """Automatic Startup Ingestion -- called once from backend/main.py's
    `@app.on_event("startup")` hook (see that module) so the live
    OFF_RATING/DEF_RATING/NET_RATING/PACE cache the Market-Prior Safety
    Brake reads (`_get_team_efficiency_stats_cached`) is already warm by
    the time the FIRST real `enable_shrinkage=true` request arrives,
    instead of that first caller paying nba_api's own latency (and risk of
    a cold-start failure) themselves.

    Error Handling & Fallback: tries the live nba_api fetch first. On
    success, populates the in-memory cache AND refreshes the on-disk
    fallback snapshot (so the fallback itself stays current across
    restarts, not just present once). On ANY failure (network error,
    timeout, nba_api schema/rate-limit change, etc.) -- deliberately a
    bare `except Exception`, since a slow server BOOT is far worse than a
    slightly-stale prior for the first few minutes -- falls back to the
    last on-disk snapshot if one exists, so the server still boots fully
    operational. Never raises: this is startup-path best-effort
    ingestion, not a hard dependency the server can't run without.
    """
    now = time.time()
    try:
        data = _fetch_team_efficiency_stats(season)
        _team_efficiency_cache.update(season=season, data=data, fetched_at=now)
        _write_team_efficiency_fallback_file(season, data)
        print(f"[api_simulation] Startup: fetched live team efficiency stats (OFF_RATING/DEF_RATING/"
              f"NET_RATING/PACE) for {len(data)} teams, season {season}, from nba_api.")
        return data
    except Exception as e:
        fallback_data = _load_team_efficiency_fallback_file()
        if fallback_data:
            # Artificially back-dated fetched_at -- see FALLBACK_RETRY_SECONDS's
            # comment above: the next enable_shrinkage request more than
            # FALLBACK_RETRY_SECONDS from now will retry the live fetch
            # itself, rather than trusting this stale snapshot for the
            # full TEAM_EFFICIENCY_CACHE_TTL_SECONDS window.
            _team_efficiency_cache.update(
                season=season, data=fallback_data,
                fetched_at=now - (TEAM_EFFICIENCY_CACHE_TTL_SECONDS - FALLBACK_RETRY_SECONDS),
            )
            print(f"[api_simulation] Startup: live nba_api fetch failed ({e}); loaded "
                  f"{len(fallback_data)} teams from local fallback cache "
                  f"({TEAM_EFFICIENCY_FALLBACK_PATH}). Will retry the live fetch on the first "
                  f"enable_shrinkage request after ~{FALLBACK_RETRY_SECONDS / 60:.0f} minutes.")
            return fallback_data
        print(f"[api_simulation] Startup: live nba_api fetch failed ({e}); no local fallback cache "
              "exists yet -- enable_shrinkage requests will retry the live fetch on demand until "
              "one succeeds. Server startup continues normally.")
        return {}


def _get_team_efficiency_stats_cached(season: str) -> dict[str, dict]:
    """Per-request lookup, normally already warm from
    prefetch_team_efficiency_stats() at startup. If the TTL has expired and
    a live re-fetch fails (nba_api hiccup mid-uptime, not just at boot),
    falls back to the on-disk snapshot -- same resilience story as startup,
    just triggered on a cache-refresh miss instead of process launch --
    before finally raising (which the caller in run_simulation() already
    catches and degrades to "shrinkage skipped" for that one request).
    """
    now = time.time()
    cached = _team_efficiency_cache
    if (
        cached["data"] is not None
        and cached["season"] == season
        and (now - cached["fetched_at"]) < TEAM_EFFICIENCY_CACHE_TTL_SECONDS
    ):
        return cached["data"]
    try:
        data = _fetch_team_efficiency_stats(season)
    except Exception:
        fallback_data = _load_team_efficiency_fallback_file()
        if fallback_data:
            _team_efficiency_cache.update(
                season=season, data=fallback_data,
                fetched_at=now - (TEAM_EFFICIENCY_CACHE_TTL_SECONDS - FALLBACK_RETRY_SECONDS),
            )
            return fallback_data
        raise
    _team_efficiency_cache.update(season=season, data=data, fetched_at=now)
    _write_team_efficiency_fallback_file(season, data)
    return data


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the stdlib erf -- no scipy/numpy dependency
    needed for this one lookup."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_market_prior(stats_a: dict, stats_b: dict, home_court_margin_a: float = 0.0) -> dict:
    """Structural (possession-based) market prior: Team A's win probability,
    each team's projected score, and the projected margin, built from real
    Pace + OffRtg/DefRtg -- NOT a simulation, and NOT a crude NET_RATING
    differential (the previous version of this function).

    Method (a standard, widely-used simplified matchup-projection technique
    -- e.g. the "average the two relevant ratings" approach behind public
    SRS-style predictions -- not independently regression-fit against this
    project's own data):
      game_pace       = average of both teams' real PACE
      proj_ortg_a     = average of A's real OFF_RATING and B's real DEF_RATING
                        (A's own scoring efficiency, tempered by how much B
                        typically allows)
      proj_ortg_b     = average of B's real OFF_RATING and A's real DEF_RATING
      expected_score_* = proj_ortg_* * (game_pace / 100)   -- ratings are
                        already per-100-possessions, so this rescales to one
                        game's actual expected possession count
      expected_margin_a = expected_score_a - expected_score_b, plus
                        home_court_margin_a (positive if Team A has the real,
                        commonly-cited home-court edge this game, negative if
                        Team B does, 0.0 for a neutral court)
      prior_prob_a    = normal_cdf(expected_margin_a / kRealMarginStdDev)
    """
    pace_a = stats_a.get("pace") or kLeagueAvgPace
    pace_b = stats_b.get("pace") or kLeagueAvgPace
    game_pace = (pace_a + pace_b) / 2.0

    proj_ortg_a = (stats_a["off_rating"] + stats_b["def_rating"]) / 2.0
    proj_ortg_b = (stats_b["off_rating"] + stats_a["def_rating"]) / 2.0

    expected_score_a = proj_ortg_a * (game_pace / 100.0)
    expected_score_b = proj_ortg_b * (game_pace / 100.0)
    expected_margin_a = (expected_score_a - expected_score_b) + home_court_margin_a

    return {
        "prior_prob_a": _normal_cdf(expected_margin_a / kRealMarginStdDev),
        "expected_score_a": expected_score_a,
        "expected_score_b": expected_score_b,
        "expected_margin_a": expected_margin_a,
        "game_pace": game_pace,
    }


def apply_bayesian_shrinkage(sim_prob_a: float, prior_prob_a: float, weight: float) -> float:
    """Blends the raw simulation's Team A win probability toward the market
    prior. `weight` is how much of the ORIGINAL win/loss split (excluding
    ties, which the market prior has no opinion on and this leaves alone)
    comes from the simulation; `1 - weight` comes from the prior. weight=1.0
    is a pure no-op (returns sim_prob_a unchanged) -- this is what the UI's
    "Prevent Overconfident Simulation" toggle gates on/off.
    """
    weight = max(0.0, min(1.0, weight))
    return weight * sim_prob_a + (1.0 - weight) * prior_prob_a


# --- request / response models ------------------------------------------

class CustomPlayer(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=60)
    position: str = Field("SG", max_length=8)
    fg3_pct: float = Field(0.350, ge=0.0, le=1.0, description="3PT field goal percentage")
    fg3a: float = Field(3.0, ge=0.0, le=20.0, description="3PT attempts per game")
    usage_rate: float = Field(20.0, ge=0.0, le=100.0, description="Usage rate (%)")
    min: float = Field(20.0, ge=0.0, le=48.0, description="Target minutes per game")

    # Optional real-data-grounded inputs to the possession state machine's
    # defensive/playmaking mechanics (see cpp_engine/cuda_simulator.cu).
    # All are ALREADY-PER-GAME figures (like fg3a/min above), not season
    # totals -- if you're importing a real player, divide their season
    # blk/stl/ast by games played yourself before submitting (the same
    # conversion index.html's roster editor already does for fg3a/min).
    # Any field left unset falls back to the engine's documented neutral
    # default, exactly like a real roster with no override at all.
    rim_protection_gravity: Optional[float] = Field(
        None, ge=0.0, le=6.0,
        description="Real per-game blocks (e.g. 3.8 for an elite rim protector). Drives the engine's "
                     "helpside rim-deterrence effect (team-wide max across the top-5 rotation). "
                     "Omit for the engine's neutral default (0.5).")
    help_defense_iq: Optional[float] = Field(
        None, ge=0.0, le=5.0,
        description="Real per-game steals. Drives the engine's forced-turnover effect (team-wide mean "
                     "across the top-5 rotation). Omit for the engine's neutral default (1.0).")
    playmaking_gravity: Optional[float] = Field(
        None, ge=0.0, le=15.0,
        description="Real per-game assists. Drives the engine's ball-movement/kick-out-3 boost "
                     "(team-wide mean across the top-5 rotation). Omit for the engine's neutral default (4.5).")
    on_ball_defense_rating: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="0-100 scale, higher = tougher individual defender. No real per-player tracking-stat "
                     "source is wired into this project yet, so most callers should omit this and let it "
                     "fall back to the engine's neutral value (50.0).")

    @model_validator(mode="after")
    def _validate_name(self):
        if not PLAYER_NAME_RE.match(self.player_name):
            raise ValueError(
                f"player_name '{self.player_name}' contains unsupported characters "
                "(letters, digits, spaces, '.', '-', and \"'\" only)"
            )
        return self


class CustomRoster(BaseModel):
    team_name: str = Field(..., min_length=1, max_length=40)
    players: list[CustomPlayer] = Field(..., min_length=1, max_length=30)
    def_rating: Optional[float] = Field(
        None, ge=90.0, le=130.0,
        description="This team's real, live defensive rating (points allowed per 100 possessions, "
                     "from /api/team_defense), if this custom roster represents a real team whose "
                     "players were edited/traded. Carries the engine's intrinsic, data-calibrated "
                     "defensive-resistance effect through to a custom-roster run instead of it "
                     "silently going neutral. Omit for a genuinely fictional/fantasy team -- the "
                     "engine then falls back to the neutral league-average default, same as always.")

    @model_validator(mode="after")
    def _validate_name(self):
        if not TEAM_NAME_RE.match(self.team_name):
            raise ValueError(
                f"team_name '{self.team_name}' contains unsupported characters "
                "(letters, digits, spaces, '.', '-', and \"'\" only)"
            )
        return self


class SimulateRequest(BaseModel):
    mode: Literal["gpu", "cpu"] = Field(
        "gpu", description="'gpu' = 100,000-game statistical Monte Carlo batch (cuda_simulator.exe); "
                            "'cpu' = single narrated play-by-play game (simulator.exe)")
    roster_type: Literal["real", "custom"] = Field(
        "real", description="'real' = fetched live from /api/players; 'custom' = use custom_roster_a/b")

    team_a: str = Field(..., min_length=1, max_length=40,
                         description="Team abbreviation (roster_type='real') or display name (roster_type='custom')")
    team_b: str = Field(..., min_length=1, max_length=40)
    custom_roster_a: Optional[CustomRoster] = None
    custom_roster_b: Optional[CustomRoster] = None

    enable_fatigue: bool = Field(False, description="Apply back-to-back schedule fatigue penalties (gpu and cpu modes)")
    team_a_rest_days: Optional[int] = Field(None, ge=0, le=14, description="0 = team_a is on a back-to-back")
    team_b_rest_days: Optional[int] = Field(None, ge=0, le=14, description="0 = team_b is on a back-to-back")

    enable_big_match: bool = Field(False, description="Apply the 'Big Match Hot Hands / Star Momentum' boost (gpu mode only)")
    hot_hand_boost: float = Field(1.05, ge=1.0, le=1.5, description="Star usage/shooting multiplier when enable_big_match")

    enable_injuries: bool = Field(False, description="Remove injured_players_a/b from their rosters before simulating")
    injured_players_a: list[str] = Field(default_factory=list, max_length=15)
    injured_players_b: list[str] = Field(default_factory=list, max_length=15)

    home_team: Optional[Literal["a", "b"]] = Field(
        None, description="Which side (if either) has home court this game (gpu and cpu modes). "
                           "Applies the engine's intrinsic, data-calibrated home-court effect "
                           "(engine_calibration::kHomeCourtProbShift) -- not a magnitude you set "
                           "yourself; None (default) is a neutral-court no-op.")

    ml_margin: Optional[float] = Field(
        None, ge=-60.0, le=60.0,
        description="Optional pre-computed ML-predicted Team A - Team B point margin (gpu mode only), "
                     "e.g. from train_ml_model.py --predict")

    enable_shrinkage: bool = Field(
        False, description="'Prevent Overconfident Simulation' -- gpu mode + roster_type='real' only. "
                            "Blends the raw simulation's win probability toward a STRUCTURAL market prior "
                            "built from each team's real, live Pace + OFF_RATING/DEF_RATING (a possession-"
                            "based projected score/margin, not a crude NET_RATING differential -- see "
                            "compute_market_prior()'s docstring). A Bayesian-shrinkage-style safety brake "
                            "against engine overconfidence on lopsided matchups; see "
                            "apply_bayesian_shrinkage()'s docstring for exactly what kind of blend this "
                            "is and why. False (default) returns the pure, unblended raw simulation "
                            "output -- exactly what the underlying shared-clock possession engine "
                            "produces, for auditing/verification.")
    shrinkage_weight: float = Field(
        DEFAULT_SHRINKAGE_WEIGHT, ge=0.1, le=1.0,
        description="FLOOR on the raw simulation's weight when enable_shrinkage=true -- the weight used "
                     "for a genuinely close matchup (~0 real NET_RATING gap between the two teams), where "
                     "the structural market prior dominates the blend. As the real, absolute NET_RATING "
                     "gap between the two teams grows (a genuine mismatch, NOT the raw sim's own "
                     "confidence), the raw sim's weight tapers UP toward a fixed ceiling, letting the raw "
                     "engine's own structural modeling govern a real blowout instead of the market prior "
                     "flattening it -- see compute_dynamic_raw_weight()'s docstring. 1.0 = no shrinkage "
                     "ever, even for a dead-even matchup (pure raw sim, same as enable_shrinkage=false).")

    timeout_seconds: float = Field(90.0, gt=0, le=300.0)

    @model_validator(mode="after")
    def _validate(self):
        if self.roster_type == "custom":
            if self.custom_roster_a is None or self.custom_roster_b is None:
                raise ValueError("roster_type='custom' requires both custom_roster_a and custom_roster_b")
        for name in (self.team_a, self.team_b):
            if not TEAM_NAME_RE.match(name):
                raise ValueError(
                    f"'{name}' contains unsupported characters (letters, digits, spaces, '.', '-', and \"'\" only)"
                )
        for name in [*self.injured_players_a, *self.injured_players_b]:
            if not PLAYER_NAME_RE.match(name):
                raise ValueError(
                    f"injured player name '{name}' contains unsupported characters "
                    "(letters, digits, spaces, '.', '-', and \"'\" only)"
                )
        return self


class SimulateResponse(BaseModel):
    status: Literal["ok"] = "ok"
    mode: str
    engine: str
    team_a: str
    team_b: str
    resolved_team_a: Optional[str] = None
    resolved_team_b: Optional[str] = None
    parameters_applied: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    players_removed: dict = Field(default_factory=dict)
    elapsed_ms: float
    command: list[str]

    # gpu mode
    simulations_run: Optional[int] = None
    win_probability_a: Optional[float] = None
    win_probability_b: Optional[float] = None
    tie_probability: Optional[float] = None
    average_score_a: Optional[float] = None
    average_score_b: Optional[float] = None
    average_point_differential_a_minus_b: Optional[float] = None
    point_differential_std_dev: Optional[float] = None

    # Layer 2: Bayesian shrinkage diagnostics (gpu mode only). raw_* is
    # always the pure, unblended simulation output -- present even when
    # enable_shrinkage=true -- so a caller can audit raw vs. blended side by
    # side. win_probability_a/b above IS the raw output when shrinkage
    # wasn't applied (enable_shrinkage=false, or it couldn't be: custom
    # roster, cpu mode, or missing real efficiency stats for one team).
    shrinkage_applied: bool = False
    raw_win_probability_a: Optional[float] = None
    raw_win_probability_b: Optional[float] = None
    market_prior_probability_a: Optional[float] = None
    market_prior_probability_b: Optional[float] = None
    market_prior_expected_score_a: Optional[float] = None
    market_prior_expected_score_b: Optional[float] = None

    # cpu mode
    final_score_a: Optional[int] = None
    final_score_b: Optional[int] = None
    winner: Optional[str] = None
    play_by_play: Optional[str] = None

    raw_stdout: str


# --- helpers --------------------------------------------------------------

def resolve_executable(mode: str) -> Path:
    for candidate in EXECUTABLE_CANDIDATES[mode]:
        if candidate.is_file():
            return candidate
    name = ENGINE_NAMES[mode]
    searched = "\n  ".join(str(c) for c in EXECUTABLE_CANDIDATES[mode])
    raise SimulationError(
        f"{name}.exe not found. Build it first "
        f"(cmake --build cpp_engine/build --config Release --target {name}). Looked in:\n  {searched}"
    )


def _custom_player_dict(p: CustomPlayer) -> dict:
    d = {
        "player_name": p.player_name,
        "position": p.position,
        "min": p.min,
        "usage_rate": p.usage_rate,
        "fg3a": p.fg3a,
        "fg3_pct": p.fg3_pct,
    }
    # Only included when the caller actually supplied them -- cpp_engine's
    # loader checks contains(), not just a default value, so omitting a key
    # here reproduces the exact same neutral engine default as a real
    # roster with no override, instead of previously being silently dropped
    # by Pydantic (these weren't declared fields at all before) and
    # defaulting to a fixed 0.0/50.0 regardless of what was meant.
    if p.rim_protection_gravity is not None:
        d["rim_protection_gravity"] = p.rim_protection_gravity
    if p.help_defense_iq is not None:
        d["help_defense_iq"] = p.help_defense_iq
    if p.playmaking_gravity is not None:
        d["playmaking_gravity"] = p.playmaking_gravity
    if p.on_ball_defense_rating is not None:
        d["on_ball_defense_rating"] = p.on_ball_defense_rating
    return d


def write_custom_roster_file(roster_a: CustomRoster, roster_b: CustomRoster) -> Path:
    """Writes the submitted rosters to a server-generated temp JSON file in
    the shape cpp_engine's --custom-roster loader expects. The caller never
    supplies (or sees) this path -- it's generated here and passed straight
    to the subprocess, then deleted once the run completes (see the
    `finally` block in run_simulation()).
    """
    payload = {
        "team_a_name": roster_a.team_name,
        "team_a_roster": [_custom_player_dict(p) for p in roster_a.players],
        "team_b_name": roster_b.team_name,
        "team_b_roster": [_custom_player_dict(p) for p in roster_b.players],
    }
    # Only included when the caller supplied it (a real team's real, live
    # def_rating) -- cpp_engine's loader checks contains(), not just a
    # default value, so omitting it here reproduces the exact same neutral
    # fallback as always for a genuinely fictional matchup.
    if roster_a.def_rating is not None:
        payload["team_a_def_rating"] = roster_a.def_rating
    if roster_b.def_rating is not None:
        payload["team_b_def_rating"] = roster_b.def_rating

    fd, path_str = tempfile.mkstemp(prefix="nba_custom_roster_", suffix=".json")
    path = Path(path_str)
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def build_command(exe: Path, req: SimulateRequest, custom_roster_path: Optional[Path]
                   ) -> tuple[list[str], str, str, dict, list[str]]:
    """Builds the argv list to execute, the exact team_a/team_b argv strings
    passed (needed by the parse_*_output functions below, since a custom
    team name can contain spaces and so can't be recovered by blindly
    splitting the engine's stdout on whitespace), the set of toggles
    actually applied, and any warnings about toggles that don't apply in
    this mode. Always a Python list passed straight to subprocess.run --
    never a shell string -- so there is no command-injection surface here
    regardless of what team/player names contain.
    """
    applied: dict = {}
    warnings: list[str] = []

    if req.roster_type == "custom":
        assert custom_roster_path is not None
        cmd = [str(exe), "--custom-roster", str(custom_roster_path)]
        team_a_arg = req.custom_roster_a.team_name
        team_b_arg = req.custom_roster_b.team_name
    else:
        cmd = [str(exe)]
        team_a_arg = req.team_a.upper()
        team_b_arg = req.team_b.upper()

    cmd += [team_a_arg, team_b_arg]
    if req.mode == "gpu":
        cmd.append(GPU_SIM_MODE_ARG)

    if req.enable_injuries:
        if req.injured_players_a:
            cmd += ["--injured-a", ",".join(req.injured_players_a)]
            applied["injured_players_a"] = req.injured_players_a
        if req.injured_players_b:
            cmd += ["--injured-b", ",".join(req.injured_players_b)]
            applied["injured_players_b"] = req.injured_players_b
        if not req.injured_players_a and not req.injured_players_b:
            warnings.append("enable_injuries=true but injured_players_a/injured_players_b are both empty -- no effect.")

    gpu_only = req.mode != "gpu"

    if req.ml_margin is not None:
        if gpu_only:
            warnings.append("ml_margin only applies in mode='gpu'; ignored for mode='cpu'.")
        else:
            cmd += ["--ml-margin", f"{req.ml_margin:.4f}"]
            applied["ml_margin"] = req.ml_margin

    if req.enable_big_match:
        if gpu_only:
            warnings.append("enable_big_match only applies in mode='gpu'; ignored for mode='cpu'.")
        else:
            cmd += ["--hot-hand-boost", f"{req.hot_hand_boost:.4f}"]
            applied["hot_hand_boost"] = req.hot_hand_boost

    if req.home_team is not None:
        # Intrinsic, data-calibrated effect -- both cuda_simulator.exe and
        # simulator.exe apply it now, so no gpu_only gate.
        flag = "--home-a" if req.home_team == "a" else "--home-b"
        cmd.append(flag)
        applied["home_team"] = req.home_team

    if req.enable_fatigue:
        # Intrinsic, data-calibrated effect -- both engines apply it now, so
        # no gpu_only gate.
        a_b2b = req.team_a_rest_days == 0
        b_b2b = req.team_b_rest_days == 0
        if a_b2b:
            cmd.append("--b2b-a")
            applied["team_a_b2b"] = True
        if b_b2b:
            cmd.append("--b2b-b")
            applied["team_b_b2b"] = True
        if not a_b2b and not b_b2b:
            warnings.append("enable_fatigue=true but neither team_a_rest_days nor team_b_rest_days is 0 -- no effect.")

    return cmd, team_a_arg, team_b_arg, applied, warnings


def parse_gpu_output(stdout: str, team_a: str, team_b: str) -> dict:
    """`team_a`/`team_b` are the exact strings passed as argv (real
    abbreviation or custom team_name) -- used, not re-derived from stdout,
    because a custom team_name can contain spaces (e.g. "Dream Team"), which
    a naive `\\S+`-based split on the engine's stdout would mis-parse.
    """
    idx = stdout.find(GPU_BLOCK_MARKER)
    if idx == -1:
        raise SimulationError("GPU Monte Carlo results block not found in simulator output.")
    section = stdout[idx:]

    a_re, b_re = re.escape(team_a), re.escape(team_b)
    header = re.search(rf"GPU MONTE CARLO BATCH RESULTS:\s*{a_re}\s+vs\s+{b_re}", section)
    if not header:
        raise SimulationError(
            f"No results for '{team_a}' vs '{team_b}' -- most likely one of these team "
            "abbreviations wasn't found in the fetched roster data, which silently falls "
            "back to a default matchup inside the engine."
        )

    win_prob_a = re.search(rf"^\s*{a_re}\s+win probability\s*:\s*([\d.]+)%", section, re.MULTILINE)
    win_prob_b = re.search(rf"^\s*{b_re}\s+win probability\s*:\s*([\d.]+)%", section, re.MULTILINE)
    avg_score_a = re.search(rf"^\s*{a_re}\s+average score\s*:\s*([\d.]+)", section, re.MULTILINE)
    avg_score_b = re.search(rf"^\s*{b_re}\s+average score\s*:\s*([\d.]+)", section, re.MULTILINE)

    if not win_prob_a or not win_prob_b:
        raise SimulationError(f"Win probability missing for {team_a}/{team_b} in GPU output.")
    if not avg_score_a or not avg_score_b:
        raise SimulationError(f"Average score missing for {team_a}/{team_b} in GPU output.")

    sim_count = SIM_COUNT_RE.search(section)
    tie_match = TIE_RE.search(section)
    margin_match = MARGIN_RE.search(section)
    stddev_match = STDDEV_RE.search(section)
    avg_a, avg_b = float(avg_score_a.group(1)), float(avg_score_b.group(1))

    return {
        "resolved_team_a": team_a,
        "resolved_team_b": team_b,
        "simulations_run": int(sim_count.group(1)) if sim_count else None,
        "win_probability_a": float(win_prob_a.group(1)) / 100.0,
        "win_probability_b": float(win_prob_b.group(1)) / 100.0,
        "tie_probability": float(tie_match.group(1)) / 100.0 if tie_match else 0.0,
        "average_score_a": avg_a,
        "average_score_b": avg_b,
        "average_point_differential_a_minus_b": float(margin_match.group(1)) if margin_match else (avg_a - avg_b),
        "point_differential_std_dev": float(stddev_match.group(1)) if stddev_match else None,
    }


def parse_cpu_output(stdout: str, team_a: str, team_b: str) -> dict:
    """See parse_gpu_output()'s docstring for why `team_a`/`team_b` are
    passed in rather than re-derived from stdout.
    """
    a_re, b_re = re.escape(team_a), re.escape(team_b)
    match = re.search(rf"FINAL SCORE:\s*{a_re}\s+(\d+)\s*-\s*(\d+)\s*{b_re}", stdout)
    if not match:
        raise SimulationError(
            f"No FINAL SCORE for '{team_a}' vs '{team_b}' -- most likely one of these team "
            "abbreviations wasn't found in the fetched roster data, which silently falls "
            "back to a default matchup inside the engine."
        )
    score_a, score_b = int(match.group(1)), int(match.group(2))
    winner = team_a if score_a > score_b else (team_b if score_b > score_a else None)
    return {
        "resolved_team_a": team_a,
        "resolved_team_b": team_b,
        "final_score_a": score_a,
        "final_score_b": score_b,
        "winner": winner,
        "play_by_play": stdout.strip(),
    }


def parse_players_removed(stdout: str, team_a: str, team_b: str) -> dict:
    """See parse_gpu_output()'s docstring for why `team_a`/`team_b` are
    passed in rather than re-derived from stdout.
    """
    result = {}
    for team in (team_a, team_b):
        m = re.search(rf"\[Injuries\]\s*{re.escape(team)}:\s*removing\s*(\d+)\s*flagged player", stdout)
        if m:
            result[team] = int(m.group(1))
    return result


def run_simulation(req: SimulateRequest) -> SimulateResponse:
    try:
        exe = resolve_executable(req.mode)
    except SimulationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    custom_roster_path: Optional[Path] = None
    if req.roster_type == "custom":
        custom_roster_path = write_custom_roster_file(req.custom_roster_a, req.custom_roster_b)

    try:
        cmd, team_a_arg, team_b_arg, applied, warnings = build_command(exe, req, custom_roster_path)

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,   # EOF on any interactive prompt -> default (skip customization, etc.)
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=req.timeout_seconds,
                cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired as e:
            raise HTTPException(
                status_code=504,
                detail=f"Simulation timed out after {req.timeout_seconds}s ({ENGINE_NAMES[req.mode]}.exe).",
            ) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Failed to launch {exe}: {e}") from e
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
            raise HTTPException(
                status_code=502,
                detail=f"{ENGINE_NAMES[req.mode]}.exe exited with code {proc.returncode}. Last output:\n{tail}",
            )

        try:
            if req.mode == "gpu":
                parsed = parse_gpu_output(proc.stdout, team_a_arg, team_b_arg)
            else:
                parsed = parse_cpu_output(proc.stdout, team_a_arg, team_b_arg)
        except SimulationError as e:
            raise HTTPException(status_code=502, detail=f"Could not parse simulator output: {e}") from e

        if req.mode == "gpu":
            # Always expose the pure raw simulation output, whether or not
            # shrinkage is requested/applied -- lets a caller audit raw vs.
            # blended side by side (the whole point of the diagnostic toggle).
            parsed["raw_win_probability_a"] = parsed["win_probability_a"]
            parsed["raw_win_probability_b"] = parsed["win_probability_b"]

        if req.enable_shrinkage:
            if req.mode != "gpu":
                warnings.append("enable_shrinkage only applies in mode='gpu'; ignored for mode='cpu'.")
            elif req.roster_type != "real":
                warnings.append("enable_shrinkage requires roster_type='real' (a custom/fantasy roster "
                                 "has no real Pace/OffRtg/DefRtg to build a structural prior from); ignored.")
            else:
                try:
                    efficiency_stats = _get_team_efficiency_stats_cached(NBA_SEASON_DEFAULT)
                    stats_a = efficiency_stats.get(team_a_arg)
                    stats_b = efficiency_stats.get(team_b_arg)
                except Exception as e:  # nba_api network hiccup -- don't fail the whole simulation over it
                    stats_a = stats_b = None
                    warnings.append(f"enable_shrinkage: could not fetch team efficiency stats ({e}); shrinkage skipped.")

                if stats_a is None or stats_b is None:
                    if stats_a is not None or stats_b is not None:
                        warnings.append(f"enable_shrinkage: efficiency stats not found for "
                                         f"'{team_a_arg if stats_a is None else team_b_arg}'; shrinkage skipped.")
                else:
                    win_a_raw = parsed["win_probability_a"]
                    win_b_raw = parsed["win_probability_b"]
                    non_tie = win_a_raw + win_b_raw
                    frac_a = (win_a_raw / non_tie) if non_tie > 0 else 0.5

                    # Fold in the SAME real home-court edge direction the raw
                    # sim's own intrinsic effect uses (req.home_team), so the
                    # structural prior isn't silently neutral-court when the
                    # raw sim isn't -- this is exactly the kind of directional
                    # mismatch that would otherwise look like a home/away bug.
                    if req.home_team == "a":
                        home_court_margin_a = kHomeCourtPointsPrior
                    elif req.home_team == "b":
                        home_court_margin_a = -kHomeCourtPointsPrior
                    else:
                        home_court_margin_a = 0.0

                    prior = compute_market_prior(stats_a, stats_b, home_court_margin_a)
                    delta_net_rating = abs(stats_a["net_rating"] - stats_b["net_rating"])
                    dynamic_weight = compute_dynamic_raw_weight(
                        delta_net_rating, min_weight=req.shrinkage_weight)
                    blended_frac_a = apply_bayesian_shrinkage(frac_a, prior["prior_prob_a"], dynamic_weight)

                    parsed["win_probability_a"] = blended_frac_a * non_tie
                    parsed["win_probability_b"] = (1.0 - blended_frac_a) * non_tie
                    parsed["market_prior_probability_a"] = prior["prior_prob_a"]
                    parsed["market_prior_probability_b"] = 1.0 - prior["prior_prob_a"]
                    parsed["market_prior_expected_score_a"] = prior["expected_score_a"]
                    parsed["market_prior_expected_score_b"] = prior["expected_score_b"]
                    parsed["shrinkage_applied"] = True
                    applied["shrinkage_weight"] = dynamic_weight
                    applied["shrinkage_weight_floor"] = req.shrinkage_weight
                    applied["delta_net_rating"] = delta_net_rating
                    applied["net_rating_a"] = stats_a["net_rating"]
                    applied["net_rating_b"] = stats_b["net_rating"]
                    applied["market_prior_expected_margin_a"] = prior["expected_margin_a"]
                    applied["market_prior_game_pace"] = prior["game_pace"]
                    applied["market_prior_off_rating_a"] = stats_a["off_rating"]
                    applied["market_prior_def_rating_a"] = stats_a["def_rating"]
                    applied["market_prior_off_rating_b"] = stats_b["off_rating"]
                    applied["market_prior_def_rating_b"] = stats_b["def_rating"]
                    if home_court_margin_a != 0.0:
                        applied["market_prior_home_court_points_a"] = home_court_margin_a

        return SimulateResponse(
            mode=req.mode,
            engine=f"{ENGINE_NAMES[req.mode]}.exe",
            team_a=team_a_arg,
            team_b=team_b_arg,
            parameters_applied=applied,
            warnings=warnings,
            players_removed=parse_players_removed(proc.stdout, team_a_arg, team_b_arg),
            elapsed_ms=elapsed_ms,
            command=cmd,
            raw_stdout=proc.stdout,
            **parsed,
        )
    finally:
        if custom_roster_path is not None:
            custom_roster_path.unlink(missing_ok=True)


# --- router -----------------------------------------------------------

router = APIRouter()


@router.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """Runs one matchup through either the GPU statistical batch engine or
    the CPU narrative single-game engine, with real or custom rosters and
    the project's advanced-analytics toggles, and returns a structured result.
    """
    return run_simulation(req)


# --- Hybrid ML layer (train_ml_model.py's saved XGBoost margin model) -----
# `train_ml_model.py --predict` already supports printing one ad hoc
# margin from the command line; this section is the same prediction
# (identical feature pipeline -- see _load_ml_artifacts()) exposed as a
# synchronous HTTP endpoint, then fed straight into a real GPU batch run via
# `--ml-margin` (see build_command()) so a caller gets the ML layer's own
# standalone read AND the resulting hybrid (ML + Monte Carlo) simulation in
# one response -- exactly what index.html's "Hybrid ML" button needs.

_ml_artifacts_cache: dict = {}


def _load_ml_artifacts() -> tuple[object, dict, Optional[dict], Optional[dict]]:
    """Lazily loads and in-process-caches the Hybrid ML layer's trained
    model (ml_model/margin_model.joblib), its team feature table
    (ml_model/team_features.json), its last training run's holdout accuracy
    report (ml_model/holdout_predictions.json's `eval_report`) if present,
    and its fitted win-probability calibrator (ml_model/
    margin_calibrator.joblib -- see train_ml_model.py's
    fit_win_probability_calibrator()) if present. All are produced OFFLINE
    by `train_ml_model.py`; this backend never trains or fetches nba_api
    data on the request path, only loads what's already saved to disk.

    The calibrator is the 4th tuple element and may be None (an older
    ml_model/ directory predating calibration support, or a training run
    with too few holdout games to fit one) -- callers fall back to the
    uncalibrated normal_cdf conversion in that case, not a crash.

    Raises SimulationError (never a raw exception) if no trained model
    exists yet, or if this environment is missing the ML dependencies
    (scikit-learn/xgboost/joblib -- see backend/requirements.txt).
    """
    if "model" in _ml_artifacts_cache:
        return (_ml_artifacts_cache["model"], _ml_artifacts_cache["team_features"],
                _ml_artifacts_cache["holdout_meta"], _ml_artifacts_cache["calibrator"])

    model_path = ML_MODEL_DIR / "margin_model.joblib"
    features_path = ML_MODEL_DIR / "team_features.json"
    if not model_path.is_file() or not features_path.is_file():
        raise SimulationError(
            f"No trained Hybrid ML model found in {ML_MODEL_DIR}/ -- run "
            f"`python train_ml_model.py` first (see its module docstring)."
        )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import joblib
        import train_ml_model as ml_pipeline
    except ImportError as e:
        raise SimulationError(
            f"Hybrid ML dependencies not installed in this environment ({e}); "
            "see backend/requirements.txt (scikit-learn, xgboost, joblib)."
        ) from e

    model = joblib.load(model_path)
    with features_path.open(encoding="utf-8") as f:
        raw_features = json.load(f)
    team_features = {abbr: ml_pipeline.TeamFeatures(**vals) for abbr, vals in raw_features.items()}

    holdout_meta = None
    holdout_path = ML_MODEL_DIR / "holdout_predictions.json"
    if holdout_path.is_file():
        with holdout_path.open(encoding="utf-8") as f:
            holdout_meta = json.load(f).get("eval_report")

    calibrator = None
    calibrator_path = ML_MODEL_DIR / "margin_calibrator.joblib"
    if calibrator_path.is_file():
        calibrator = joblib.load(calibrator_path)  # {"model": ..., "method": "platt"|"isotonic"}

    _ml_artifacts_cache.update(model=model, team_features=team_features, holdout_meta=holdout_meta,
                                calibrator=calibrator)
    return model, team_features, holdout_meta, calibrator


_hybrid_pipeline_benchmark_cache: dict = {}


def _load_hybrid_pipeline_benchmark() -> Optional[dict]:
    """Loads (and in-process-caches) ml_model/hybrid_pipeline_benchmark.json
    -- a static, versioned record of the FULL hybrid pipeline's (calibrated
    Monte Carlo engine + ML margin bias + Hot Hand boost) measured win/loss
    accuracy on a real chronological holdout, from `backtest_model.py
    --mode hybrid`. This is the project's best-performing measured
    configuration (~68.6% at last measurement) and is DIFFERENT from
    ml_model/holdout_predictions.json's directional_accuracy (~65.7%),
    which only checks the ML model's predicted margin SIGN in isolation,
    with no Monte Carlo engine involved at all -- see the JSON file's own
    `description` field for the full explanation.

    Returns None (not an error) if the file is missing -- this benchmark is
    a nice-to-have reference number, not something /api/simulate-ml's core
    prediction depends on, and re-deriving it live would mean running a
    ~245-game backtest (nba_api box-score lookups included) inside a single
    HTTP request, which is exactly what this project's other endpoints
    deliberately avoid (see this module's docstring's "No live retraining
    on the request path" note).
    """
    if "data" in _hybrid_pipeline_benchmark_cache:
        return _hybrid_pipeline_benchmark_cache["data"]

    benchmark_path = ML_MODEL_DIR / "hybrid_pipeline_benchmark.json"
    if not benchmark_path.is_file():
        _hybrid_pipeline_benchmark_cache["data"] = None
        return None

    with benchmark_path.open(encoding="utf-8") as f:
        data = json.load(f)
    _hybrid_pipeline_benchmark_cache["data"] = data
    return data


def apply_contextual_ml_adjustment(
    base_margin: float, model, team_features: dict, team_a: str, team_b: str,
    ml_pipeline, enable_big_match: bool, hot_hand_boost: float,
    enable_fatigue: bool, team_a_rest_days: Optional[int], team_b_rest_days: Optional[int],
    home_team: Optional[Literal["a", "b"]] = None,
) -> tuple[float, bool]:
    """Takes `base_margin` (train_ml_model.py's context-free prediction --
    the value actually passed to the GPU engine via --ml-margin) and
    returns a (possibly) DIFFERENT margin meant only for what
    /api/simulate-ml DISPLAYS, so toggling "Big Match / Hot Hand Boost",
    "Fatigue (B2B)", or the Home Court Advantage dropdown visibly moves the
    Hybrid ML Layer panel instead of it staying static -- plus whether any
    adjustment was actually applied.

    Deliberately NOT fed back into --ml-margin: the GPU engine already
    applies all three of these effects intrinsically on its own simulation
    (see cuda_simulator.cu's hot_hand_boost/--b2b-a/--b2b-b/--home-a/
    --home-b handling) -- doing so again here would double-count the exact
    same real-world effect the engine itself is already accounting for,
    precisely the failure mode train_ml_model.py's module docstring warns
    about for its base feature set. This function's whole point is to make
    the DISPLAY consistent with what's being simulated, not to change what's
    simulated -- the actual headline win_probability_a/b in the response
    (from the real GPU engine run) already moves correctly with home_team
    on its own; this only fixes the separate ML-only display panel looking
    static beside it.

    * Big Match / Hot Hand Boost: a genuine re-inference through the
      trained model (see predict_margin_with_context()'s docstring for
      exactly what counterfactual it re-infers on) -- not an invented
      constant. Falls back to `base_margin` silently on any ML pipeline
      error (team validity was already checked upstream by the caller's
      own predict_margin() call, so this should only fail on something
      genuinely unexpected).
    * Fatigue (B2B): a real, calibrated points-of-margin shift
      (kRestAdvantageMarginPerDay) applied only when exactly one side is on
      a back-to-back -- if both (or neither) are, the engine's own
      per-side, symmetric penalty cancels out in relative terms, so no net
      adjustment is applied here either (matches the engine's own logic).
    * Home Court Advantage: the same real, calibrated points-of-margin
      shift already used for the market-prior blend (kHomeCourtPointsPrior,
      see its own docstring), applied toward whichever side home_team names
      -- +kHomeCourtPointsPrior for team_a, -kHomeCourtPointsPrior for
      team_b, no shift when neutral (None). Mirrors the engine's own
      --home-a/--home-b sign convention exactly, so this display-only shift
      moves in the same direction as the real simulated result.
    """
    margin = base_margin
    adjusted = False

    if enable_big_match and hot_hand_boost != 1.0:
        try:
            margin = ml_pipeline.predict_margin_with_context(
                model, team_features, team_a, team_b, hot_hand_boost=hot_hand_boost)
            adjusted = True
        except ml_pipeline.MlPipelineError:
            margin = base_margin  # fall back rather than fail the whole request over a display-only extra

    if enable_fatigue:
        a_b2b = team_a_rest_days == 0
        b_b2b = team_b_rest_days == 0
        if a_b2b and not b_b2b:
            margin -= kRestAdvantageMarginPerDay
            adjusted = True
        elif b_b2b and not a_b2b:
            margin += kRestAdvantageMarginPerDay
            adjusted = True

    if home_team == "a":
        margin += kHomeCourtPointsPrior
        adjusted = True
    elif home_team == "b":
        margin -= kHomeCourtPointsPrior
        adjusted = True

    return margin, adjusted


class SimulateMlRequest(BaseModel):
    """Same shape as the relevant subset of SimulateRequest, minus what the
    Hybrid ML layer doesn't support: mode is always 'gpu' (--ml-margin is a
    GPU-kernel-only override) and roster_type is always 'real' (the trained
    model's team_features are keyed by real NBA abbreviations from the last
    `train_ml_model.py` run -- there is no equivalent feature vector for a
    custom/fantasy roster).
    """
    team_a: str = Field(..., min_length=1, max_length=40, description="Real team abbreviation")
    team_b: str = Field(..., min_length=1, max_length=40, description="Real team abbreviation")

    enable_fatigue: bool = Field(False, description="Apply back-to-back schedule fatigue penalties")
    team_a_rest_days: Optional[int] = Field(None, ge=0, le=14, description="0 = team_a is on a back-to-back")
    team_b_rest_days: Optional[int] = Field(None, ge=0, le=14, description="0 = team_b is on a back-to-back")

    enable_big_match: bool = Field(False, description="Apply the 'Big Match Hot Hands / Star Momentum' boost")
    hot_hand_boost: float = Field(1.05, ge=1.0, le=1.5, description="Star usage/shooting multiplier when enable_big_match")

    home_team: Optional[Literal["a", "b"]] = Field(
        None, description="Which side (if either) has home court -- applies the engine's intrinsic, "
                           "data-calibrated home-court effect, same as /api/simulate.")

    enable_shrinkage: bool = Field(
        False, description="Also apply the Market-Prior Safety Brake on top of this hybrid GPU result "
                            "(stacks with the ML margin bias -- see apply_bayesian_shrinkage()).")
    shrinkage_weight: float = Field(DEFAULT_SHRINKAGE_WEIGHT, ge=0.1, le=1.0)

    timeout_seconds: float = Field(90.0, gt=0, le=300.0)

    @model_validator(mode="after")
    def _validate(self):
        for name in (self.team_a, self.team_b):
            if not TEAM_NAME_RE.match(name):
                raise ValueError(
                    f"'{name}' contains unsupported characters (letters, digits, spaces, '.', '-', and \"'\" only)"
                )
        return self


class SimulateMlResponse(SimulateResponse):
    """Everything SimulateResponse already returns for a gpu-mode run (raw
    Monte Carlo win probabilities, and Safety-Brake-blended ones too if
    enable_shrinkage was set), PLUS the Hybrid ML layer's own standalone
    read on the matchup.
    """
    ml_predicted_margin_a_minus_b: float = Field(
        description="The Hybrid ML Layer's DISPLAYED Team A - Team B point margin -- this is "
                     "ml_base_predicted_margin_a_minus_b below, contextually adjusted for the "
                     "enable_big_match/hot_hand_boost and enable_fatigue toggles if either was set (see "
                     "apply_contextual_ml_adjustment()'s docstring). NOT the same value passed into the GPU "
                     "engine via --ml-margin (that's always the unadjusted base prediction, to avoid "
                     "double-counting effects the engine already applies intrinsically) -- see "
                     "ml_base_predicted_margin_a_minus_b for that exact value.")
    ml_base_predicted_margin_a_minus_b: float = Field(
        description="train_ml_model.py's XGBoost-predicted Team A - Team B point margin with NO contextual "
                     "adjustment applied -- this IS the exact value passed into the GPU engine via --ml-margin. "
                     "Equal to ml_predicted_margin_a_minus_b whenever ml_context_adjustment_applied is False.")
    ml_context_adjustment_applied: bool = Field(
        description="True if enable_big_match/hot_hand_boost and/or enable_fatigue actually shifted "
                     "ml_predicted_margin_a_minus_b away from the base prediction for this request.")
    ml_win_probability_a: float = Field(
        description="ml_predicted_margin_a_minus_b (the DISPLAYED, contextually-adjusted margin) converted to "
                     "a win probability via a calibrator GENUINELY FIT against real held-out win/loss outcomes "
                     "(Platt scaling or Isotonic Regression, whichever measured a lower Brier score -- see "
                     "train_ml_model.py's fit_win_probability_calibrator() and ml_calibration_method below) -- "
                     "NOT an assumed-Gaussian normal_cdf(margin / kRealMarginStdDev) shortcut. Falls back to "
                     "that shortcut only if this ml_model/ directory predates calibration support (see "
                     "ml_calibration_method).")
    ml_win_probability_b: float
    ml_calibration_method: Optional[str] = Field(
        None, description="Which calibrator produced ml_win_probability_a: 'platt' or 'isotonic' (see "
                           "train_ml_model.py's fit_win_probability_calibrator(), which fits both and deploys "
                           "whichever has the lower Brier score on a genuinely held-out slice), or null if no "
                           "calibrator was available and the uncalibrated normal_cdf fallback was used instead.")
    ml_confidence_score: float = Field(
        description="How far this specific prediction sits from a 50/50 toss-up: "
                     "abs(ml_win_probability_a - 0.5) * 2, so 0.0 = coin flip, 1.0 = maximal confidence. "
                     "A per-prediction signal, NOT a calibrated probability of being correct.")
    # --- Two DIFFERENT accuracy metrics, deliberately both exposed and
    # deliberately named to not be confusable with each other -- see each
    # field's own description for exactly what it measures and why they
    # differ. Neither is recomputed live by this endpoint; both are static
    # context loaded from ml_model/ (see _load_ml_artifacts() /
    # _load_hybrid_pipeline_benchmark()).
    ml_only_holdout_accuracy: Optional[float] = Field(
        None, description="NARROW metric: the ML margin model's OWN historical accuracy, ML ALONE, no "
                           "Monte Carlo engine involved at all -- just whether its predicted margin's SIGN "
                           "matched the actual game's margin sign, on its last training run's held-out games "
                           "(ml_model/holdout_predictions.json). Static context about the model in general, "
                           "not this specific prediction. Typically LOWER than "
                           "hybrid_pipeline_benchmark_accuracy below, since it gets no credit for whatever "
                           "the possession-simulation engine itself adds on top.")
    ml_only_holdout_n_games: Optional[int] = None

    hybrid_pipeline_benchmark_accuracy: Optional[float] = Field(
        None, description="HEADLINE metric: the FULL hybrid pipeline's (calibrated Monte Carlo engine + this "
                           "same ML margin bias + Hot Hand boost for real high-leverage matchups) win/loss "
                           "accuracy on the STANDARD 245-game REGULAR-SEASON chronological holdout, measured "
                           "by `backtest_model.py --mode hybrid` -- i.e. an actual simulated game's predicted "
                           "winner vs. the real winner, with both the engine AND the ML layer contributing. "
                           "See ml_model/hybrid_pipeline_benchmark.json for full provenance and "
                           "hybrid_pipeline_playoff_accuracy below for how this generalizes to real playoff "
                           "games specifically. NOT reproduced live by this request.")
    hybrid_pipeline_benchmark_n_games: Optional[int] = None
    hybrid_pipeline_benchmark_baseline_accuracy: Optional[float] = Field(
        None, description="For context alongside hybrid_pipeline_benchmark_accuracy: the SAME held-out "
                           "regular-season games' win/loss accuracy using the pure Monte Carlo engine with no "
                           "ML margin bias at all (the 'baseline' backtest_model.py compares the hybrid pass "
                           "against).")

    hybrid_pipeline_playoff_accuracy: Optional[float] = Field(
        None, description="A SEPARATE, never-trained-on evaluation: this SAME hybrid pipeline's win/loss "
                           "accuracy on real 2024-25 NBA PLAYOFF games (see evaluate_playoff_holdout.py) -- "
                           "genuinely out-of-domain, since playoff games are deliberately excluded from "
                           "training (see train_ml_model.py's module docstring). Typically LOWER than "
                           "hybrid_pipeline_benchmark_accuracy above: playoff basketball is systematically "
                           "different (tighter rotations, intensified defense, a field already filtered to "
                           "competent teams), and the model has no training exposure to actual playoff "
                           "dynamics beyond a training-time sample weight toward regular-season games "
                           "involving top-4 contenders/prior-Finals rematches/tournament games.")
    hybrid_pipeline_playoff_n_games: Optional[int] = None
    hybrid_pipeline_playoff_baseline_accuracy: Optional[float] = Field(
        None, description="For context alongside hybrid_pipeline_playoff_accuracy: the SAME real playoff "
                           "games' win/loss accuracy using the pure Monte Carlo engine with no ML margin bias.")


@router.post("/api/simulate-ml", response_model=SimulateMlResponse)
def simulate_ml(req: SimulateMlRequest) -> SimulateMlResponse:
    """Runs the Hybrid ML layer: predicts this matchup's expected point
    margin from the trained XGBoost model, feeds that margin into a real
    100,000-game GPU Monte Carlo batch via --ml-margin (biasing its
    per-possession shot probabilities toward the data-driven baseline --
    see cpp_engine/cuda_simulator.cu), and returns the resulting simulation
    together with the ML model's own standalone prediction. See
    SimulateMlResponse's field docs for exactly what each ml_* field means
    and how it relates to the plain /api/simulate fields it's returned
    alongside.
    """
    team_a = req.team_a.strip().upper()
    team_b = req.team_b.strip().upper()

    try:
        model, team_features, holdout_meta, calibrator = _load_ml_artifacts()
    except SimulationError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    import train_ml_model as ml_pipeline  # sys.path already fixed up by _load_ml_artifacts()

    try:
        base_ml_margin = ml_pipeline.predict_margin(model, team_features, team_a, team_b)
    except ml_pipeline.MlPipelineError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Contextual adjustment is DISPLAY-ONLY -- base_ml_margin (unadjusted)
    # is what actually goes into --ml-margin below, so the toggles' real
    # effects aren't double-counted between this display and the engine's
    # own intrinsic hot-hand/fatigue handling. See
    # apply_contextual_ml_adjustment()'s docstring.
    display_ml_margin, ml_context_adjusted = apply_contextual_ml_adjustment(
        base_ml_margin, model, team_features, team_a, team_b, ml_pipeline,
        req.enable_big_match, req.hot_hand_boost,
        req.enable_fatigue, req.team_a_rest_days, req.team_b_rest_days,
        home_team=req.home_team,
    )

    # Genuinely calibrated win probability -- see train_ml_model.py's
    # fit_win_probability_calibrator()'s docstring for the Platt-scaling-
    # vs-Isotonic comparison this was chosen from. Falls back to the old
    # assumed-Gaussian normal_cdf(margin / kRealMarginStdDev) shortcut ONLY
    # if no calibrator has been saved yet (an ml_model/ directory from
    # before calibration support existed) -- not a silent behavior change
    # for anyone already running a current training output.
    ml_calibration_method: Optional[str] = None
    if calibrator is not None:
        ml_win_probability_a = ml_pipeline.predict_win_probability(
            calibrator["model"], calibrator["method"], display_ml_margin)
        ml_calibration_method = calibrator["method"]
    else:
        ml_win_probability_a = _normal_cdf(display_ml_margin / kRealMarginStdDev)
    ml_win_probability_b = 1.0 - ml_win_probability_a
    ml_confidence_score = abs(ml_win_probability_a - 0.5) * 2.0

    sim_req = SimulateRequest(
        mode="gpu",
        roster_type="real",
        team_a=team_a,
        team_b=team_b,
        enable_fatigue=req.enable_fatigue,
        team_a_rest_days=req.team_a_rest_days,
        team_b_rest_days=req.team_b_rest_days,
        enable_big_match=req.enable_big_match,
        hot_hand_boost=req.hot_hand_boost,
        home_team=req.home_team,
        ml_margin=base_ml_margin,
        enable_shrinkage=req.enable_shrinkage,
        shrinkage_weight=req.shrinkage_weight,
        timeout_seconds=req.timeout_seconds,
    )
    base_response = run_simulation(sim_req)
    benchmark = _load_hybrid_pipeline_benchmark()

    # hybrid_pipeline_benchmark.json nests the regular-season and playoff
    # evaluations separately (see that file's own "description" fields) --
    # both read defensively since either sub-object may be absent on an
    # older/partial benchmark file.
    regular_season_bench = (benchmark or {}).get("regular_season_holdout", {})
    playoff_bench = (benchmark or {}).get("playoff_holdout", {})

    return SimulateMlResponse(
        **base_response.model_dump(),
        ml_predicted_margin_a_minus_b=display_ml_margin,
        ml_base_predicted_margin_a_minus_b=base_ml_margin,
        ml_context_adjustment_applied=ml_context_adjusted,
        ml_win_probability_a=ml_win_probability_a,
        ml_win_probability_b=ml_win_probability_b,
        ml_calibration_method=ml_calibration_method,
        ml_confidence_score=ml_confidence_score,
        ml_only_holdout_accuracy=(holdout_meta or {}).get("directional_accuracy"),
        ml_only_holdout_n_games=(holdout_meta or {}).get("n_test"),
        hybrid_pipeline_benchmark_accuracy=regular_season_bench.get("hybrid_accuracy"),
        hybrid_pipeline_benchmark_n_games=regular_season_bench.get("n_games"),
        hybrid_pipeline_benchmark_baseline_accuracy=regular_season_bench.get("baseline_accuracy"),
        hybrid_pipeline_playoff_accuracy=playoff_bench.get("hybrid_accuracy"),
        hybrid_pipeline_playoff_n_games=playoff_bench.get("n_games"),
        hybrid_pipeline_playoff_baseline_accuracy=playoff_bench.get("baseline_accuracy"),
    )
