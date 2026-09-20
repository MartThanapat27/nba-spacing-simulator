#!/usr/bin/env python3
"""Backtests the C++/GPU NBA Monte Carlo engine against historical results.

By default this runs the project's **hybrid (ML + Monte Carlo) pipeline**
against real NBA games (fetch them first with `fetch_real_nba_data.py`):

  1. Trains `train_ml_model.py`'s expected-point-margin model with a
     chronological train/test split and pulls back the **held-out test
     games** and their predicted margins (games the model never saw during
     training -- see train_ml_model.py's module docstring).
  2. For each held-out game, shells out to the compiled `cuda_simulator`
     executable in batch Monte Carlo mode for that exact matchup, passing:
       --home-a            always (fetch_real_nba_data.py's convention: team_a
                            is always the home team) -- applies the engine's
                            intrinsic, data-calibrated home-court effect
                            (engine_calibration::kHomeCourtProbShift, see
                            calibrated_constants.h); NOT hybrid-only, since
                            home-court is now baked into the engine itself
       --b2b-a / --b2b-b   for whichever team has 0 rest days for this game
                           (real, schedule-derived -- see fetch_real_nba_data.py)
                           -- applies engine_calibration::kFatigueProbPerRestDay;
                           also NOT hybrid-only, same reason
       --ml-margin        the ML-predicted point margin (hybrid pass only)
       --hot-hand-boost   for games flagged "marquee" (train_ml_model.py's
                           big_match_indicator proxy) -- "Big Match Hot Hands"
                           (hybrid pass only, an opt-in hypothesis test, not
                           a proven/calibrated engine effect)
     Note: intrinsic defensive resistance (engine_calibration::kDefResistanceProbPerRating)
     requires no flag at all -- it's computed automatically from each team's
     real def_rating (see cpp_engine/cuda_main.cpp's /api/team_defense fetch).
  3. Parses the GPU Monte Carlo section of its stdout (win probabilities and
     average scores for both teams).
  4. Scores the prediction against the recorded actual result.

Only the held-out games are ever scored in hybrid/comparison mode -- games
used to train the ML model are excluded, since testing on them would be
leakage (the ML component would already have memorized their outcome). It
prints a per-game table plus a summary report (win/loss accuracy, average
Brier score, point-differential MAE) for the hybrid run, and -- by default
-- also runs a **baseline pass**, i.e. the engine's pure, autonomous,
data-calibrated output (home-court + fatigue + defensive resistance all
still applied intrinsically, but no `--ml-margin`/`--hot-hand-boost`) over
that same held-out set, so the two are directly, fairly comparable and the
delta isolates exactly what the external ML layer adds on top of the
self-contained engine.

Usage:
    python backtest_model.py                        # baseline + hybrid, compared
    python backtest_model.py --mode hybrid           # hybrid pass only
    python backtest_model.py --mode baseline         # plain Monte Carlo, all games, no ML
    python backtest_model.py --test-fraction 0.25    # bigger holdout set
    python backtest_model.py --csv my_games.csv --exe path\\to\\cuda_simulator.exe
    python backtest_model.py --limit 5 --verbose

Requires the FastAPI backend (http://127.0.0.1:8000) to be running, since both
the simulator and the ML feature step fetch player data from it.

See train_ml_model.py's module docstring for important caveats (dataset size
vs. model capacity, and point-in-time roster mismatch) that apply to any
accuracy figure this script reports.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import re
import subprocess
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "historical_games.csv"
DEFAULT_ML_DIR = SCRIPT_DIR / "ml_model"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/players"

# Multiplier passed as --hot-hand-boost for games train_ml_model.py flags as
# "marquee" (big_match_indicator at/above the dataset median) in the hybrid
# pass -- see cpp_engine/cuda_simulator.cu for what this does on the GPU.
# "Slight," per the task: a 5% bump to the featured star's effective shot
# probability and usage weight, not a dramatic rewrite of the matchup.
HOT_HAND_BOOST_VALUE = 1.05

# Prefer a Release build (faster, no /RTC1 runtime checks) but fall back to
# Debug, and to a non-Windows binary name in case this ever runs elsewhere.
DEFAULT_EXE_CANDIDATES = [
    SCRIPT_DIR / "cpp_engine" / "build" / "Release" / "cuda_simulator.exe",
    SCRIPT_DIR / "cpp_engine" / "build" / "Debug" / "cuda_simulator.exe",
    SCRIPT_DIR / "cpp_engine" / "build" / "Release" / "cuda_simulator",
    SCRIPT_DIR / "cpp_engine" / "build" / "Debug" / "cuda_simulator",
]

# The simulator prints a CPU batch summary and then a GPU batch summary,
# using near-identical field labels ("<TEAM> win probability", "<TEAM>
# average score") in both blocks. Anchoring on this marker and slicing the
# GPU section off the tail of stdout keeps the regexes below from ever
# picking up the CPU engine's numbers by mistake.
GPU_BLOCK_MARKER = "GPU MONTE CARLO BATCH RESULTS"
GPU_HEADER_RE = re.compile(r"GPU MONTE CARLO BATCH RESULTS:\s*(\S+)\s+vs\s+(\S+)")
WIN_PROB_RE = re.compile(r"^\s*(\S+)\s+win probability\s*:\s*([\d.]+)%", re.MULTILINE)
AVG_SCORE_RE = re.compile(r"^\s*(\S+)\s+average score\s*:\s*([\d.]+)", re.MULTILINE)


class BacktestError(Exception):
    """Raised for any problem scoring a single historical game."""


@dataclass
class HistoricalGame:
    team_a: str
    team_b: str
    actual_winner: str
    actual_score_a: float
    actual_score_b: float
    game_date: str = ""
    team_a_rest_days: Optional[float] = None
    team_b_rest_days: Optional[float] = None


@dataclass
class GameResult:
    game: HistoricalGame
    win_prob: dict            # team abbreviation -> predicted win probability (0-1)
    avg_score: dict           # team abbreviation -> predicted average score
    predicted_favorite: str
    predicted_margin: float   # avg_score[team_a] - avg_score[team_b]
    correct: bool
    brier: float
    margin_error: float       # abs(predicted_margin - actual_margin)
    ml_margin: Optional[float] = None        # ML-predicted margin fed into --ml-margin, if hybrid mode
    hot_hand_boost: Optional[float] = None   # multiplier fed into --hot-hand-boost, if flagged marquee
    is_team_a_home: bool = False             # --home-a applied (intrinsic, always for team_a)
    is_team_a_b2b: bool = False              # --b2b-a applied
    is_team_b_b2b: bool = False              # --b2b-b applied
    shrinkage_weight: Optional[float] = None  # dynamic raw-sim weight actually used, if Safety Brake applied
    delta_net_rating: Optional[float] = None  # abs(NET_RATING_a - NET_RATING_b), if Safety Brake applied
    cutoff_date: Optional[str] = None  # real point-in-time snapshot cutoff (D-1) actually used, if Safety Brake applied


def load_games(csv_path: Path) -> list[HistoricalGame]:
    if not csv_path.is_file():
        raise BacktestError(f"Historical games CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"team_a", "team_b", "actual_winner", "actual_score_a", "actual_score_b"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise BacktestError(f"CSV is missing required column(s): {sorted(missing)}")

        games = []
        for row in reader:
            games.append(HistoricalGame(
                team_a=row["team_a"].strip().upper(),
                team_b=row["team_b"].strip().upper(),
                actual_winner=row["actual_winner"].strip().upper(),
                actual_score_a=float(row["actual_score_a"]),
                actual_score_b=float(row["actual_score_b"]),
                game_date=row.get("game_date", "").strip() if row.get("game_date") else "",
                team_a_rest_days=float(row["team_a_rest_days"]) if row.get("team_a_rest_days") else None,
                team_b_rest_days=float(row["team_b_rest_days"]) if row.get("team_b_rest_days") else None,
            ))
    return games


def find_executable(explicit: Optional[str]) -> Path:
    if explicit:
        exe = Path(explicit)
        if not exe.is_file():
            raise BacktestError(f"--exe path does not exist: {exe}")
        return exe

    for candidate in DEFAULT_EXE_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched = "\n  ".join(str(c) for c in DEFAULT_EXE_CANDIDATES)
    raise BacktestError(
        "Could not find the cuda_simulator executable. Build it first "
        "(cmake --build cpp_engine/build --config Release --target cuda_simulator), "
        f"or pass --exe <path>. Looked in:\n  {searched}"
    )


def run_simulation(exe: Path, team_a: str, team_b: str, timeout: float,
                    ml_margin: Optional[float] = None,
                    hot_hand_boost: Optional[float] = None,
                    is_team_a_home: bool = False,
                    is_team_a_b2b: bool = False,
                    is_team_b_b2b: bool = False,
                    custom_roster_path: Optional[Path] = None) -> str:
    """Runs one GPU+CPU batch Monte Carlo pass for team_a vs team_b.

    Passing "2" as the 3rd CLI arg selects batch Monte Carlo mode up front,
    so the executable never blocks on its interactive sim-mode prompt.
    Closing stdin (DEVNULL) makes the separate "Customize rosters before
    simulating? [y/N]" prompt read EOF and default to "no" immediately, so
    the whole run is non-interactive end to end. `ml_margin`/`hot_hand_boost`,
    when given, forward as `--ml-margin`/`--hot-hand-boost` (the only two
    still-optional *external* overrides); `is_team_a_home`/`is_team_a_b2b`/
    `is_team_b_b2b`, when true, forward as the flags `--home-a`/`--b2b-a`/
    `--b2b-b` (no value) -- these trigger the engine's *intrinsic*,
    data-calibrated home-court/fatigue effects (calibrated_constants.h), not
    hybrid-only additions. Defensive resistance needs no flag at all: it's
    computed automatically from each roster's real def_rating. All
    calibrate the GPU kernel's possession probabilities (see
    cpp_engine/cuda_simulator.cu); everything omitted/false, the run is the
    engine's fully neutral-court, non-fatigued Monte Carlo model.

    `custom_roster_path`, when given, prepends `--custom-roster <path>`
    (mirrors backend/api_simulation.py's build_command() exactly -- the
    team_a/team_b positional args still follow, matching that same
    proven pattern) -- used by evaluate_game_point_in_time() to feed the
    engine a real, point-in-time-only roster (see that function's
    docstring) instead of whatever the live /api/players database
    currently holds.
    """
    cmd = [str(exe)]
    if custom_roster_path is not None:
        cmd += ["--custom-roster", str(custom_roster_path)]
    cmd += [team_a, team_b, "2"]
    if ml_margin is not None:
        cmd += ["--ml-margin", f"{ml_margin:.4f}"]
    if hot_hand_boost is not None:
        cmd += ["--hot-hand-boost", f"{hot_hand_boost:.4f}"]
    if is_team_a_home:
        cmd.append("--home-a")
    if is_team_a_b2b:
        cmd.append("--b2b-a")
    if is_team_b_b2b:
        cmd.append("--b2b-b")

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise BacktestError(f"Failed to launch simulator: {e}") from e

    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        raise BacktestError(f"cuda_simulator exited with code {proc.returncode}. Last output:\n{tail}")

    return proc.stdout


def parse_gpu_block(stdout: str) -> tuple[str, str, dict, dict]:
    idx = stdout.find(GPU_BLOCK_MARKER)
    if idx == -1:
        raise BacktestError("GPU Monte Carlo results block not found in simulator output.")
    gpu_section = stdout[idx:]

    header = GPU_HEADER_RE.search(gpu_section)
    if not header:
        raise BacktestError("Could not parse the GPU results header (team names).")
    resolved_a, resolved_b = header.group(1), header.group(2)

    win_prob = {name: float(pct) / 100.0 for name, pct in WIN_PROB_RE.findall(gpu_section)}
    avg_score = {name: float(val) for name, val in AVG_SCORE_RE.findall(gpu_section)}

    if resolved_a not in win_prob or resolved_b not in win_prob:
        raise BacktestError(f"Win probability missing for {resolved_a}/{resolved_b} in GPU output.")
    if resolved_a not in avg_score or resolved_b not in avg_score:
        raise BacktestError(f"Average score missing for {resolved_a}/{resolved_b} in GPU output.")

    return resolved_a, resolved_b, win_prob, avg_score


def evaluate_game(game: HistoricalGame, exe: Path, timeout: float, verbose: bool,
                   ml_margin: Optional[float] = None,
                   hot_hand_boost: Optional[float] = None,
                   is_team_a_home: bool = False,
                   is_team_a_b2b: bool = False,
                   is_team_b_b2b: bool = False,
                   custom_roster_path: Optional[Path] = None) -> GameResult:
    stdout = run_simulation(exe, game.team_a, game.team_b, timeout, ml_margin, hot_hand_boost,
                             is_team_a_home, is_team_a_b2b, is_team_b_b2b,
                             custom_roster_path=custom_roster_path)
    if verbose:
        print(f"\n----- raw output: {game.team_a} vs {game.team_b} -----")
        print(stdout)
        print("----- end raw output -----\n")

    resolved_a, resolved_b, win_prob, avg_score = parse_gpu_block(stdout)

    if {resolved_a, resolved_b} != {game.team_a, game.team_b}:
        raise BacktestError(
            f"Simulator resolved a different matchup ({resolved_a} vs {resolved_b}) than "
            f"requested ({game.team_a} vs {game.team_b}) -- likely one of these abbreviations "
            f"isn't in the fetched roster, triggering the executable's NYK/SAS fallback."
        )
    if game.actual_winner not in (game.team_a, game.team_b):
        raise BacktestError(
            f"actual_winner '{game.actual_winner}' is not '{game.team_a}' or '{game.team_b}'."
        )

    # "Favorite" = whichever side the model gave the higher win probability,
    # not necessarily >50% (ties eat a couple of percentage points from both
    # sides, so a heavy favorite can still show just under 50%).
    predicted_favorite = game.team_a if win_prob[game.team_a] >= win_prob[game.team_b] else game.team_b
    predicted_margin = avg_score[game.team_a] - avg_score[game.team_b]
    actual_margin = game.actual_score_a - game.actual_score_b

    correct = predicted_favorite == game.actual_winner

    # Brier score against the team that actually won: how much probability
    # mass the model put on the correct outcome, squared-error style.
    # 0.0 = the model was 100% certain of the actual winner; 1.0 = the model
    # gave the actual winner a 0% chance.
    p_actual_winner = win_prob[game.actual_winner]
    brier = (p_actual_winner - 1.0) ** 2

    margin_error = abs(predicted_margin - actual_margin)

    return GameResult(
        game=game,
        win_prob=win_prob,
        avg_score=avg_score,
        predicted_favorite=predicted_favorite,
        predicted_margin=predicted_margin,
        correct=correct,
        brier=brier,
        margin_error=margin_error,
        ml_margin=ml_margin,
        hot_hand_boost=hot_hand_boost,
        is_team_a_home=is_team_a_home,
        is_team_a_b2b=is_team_a_b2b,
        is_team_b_b2b=is_team_b_b2b,
    )


_point_in_time_cache: dict[tuple[str, str], dict] = {}

# Remembers (season, cutoff_date) keys whose live nba_api fetch has failed
# this run, and how -- without this, every game sharing a "bad" date
# (rate-limited/timed-out) independently re-attempts and re-fails the SAME
# doomed request, each costing a full request timeout and, worse, adding
# more load onto an endpoint that's already struggling (observed in
# practice: a run that hit sustained nba_api throttling spent many minutes
# retrying the same handful of unlucky dates once per game on them instead
# of once per date). A cache miss still means "try it" -- this only
# short-circuits a date already tried too recently or too many times.
#
# Maps fail_key -> {"attempts": int, "last_failed_at": float (monotonic)}.
# A date is NEVER permanently blacklisted for the rest of the run -- that
# previous behavior meant one transient nba_api rate-limit blip early in a
# date's lookup permanently skipped the Safety Brake for every later game
# sharing that date, even if nba_api would have recovered moments later
# (this is what caused the 1181->1120 matched-game regression between two
# backtest runs -- see backtest_full_season_results.md.prev). Instead, a
# date that has already failed (after fetch_point_in_time_efficiency_stats()
# has itself already retried several times with a 10-30s backoff -- see that
# function) is only skipped for a COOLDOWN window that grows with repeated
# failures (so a genuinely-dead date stops getting hammered every single
# game) but is always eligible for another real attempt once the cooldown
# elapses -- there is no attempt count that gives up on a date forever.
#
# The base cooldown MUST exceed fetch_point_in_time_efficiency_stats()'s own
# worst-case single-call retry duration (6 attempts x up to 30s backoff each
# -- roughly 10+15+20+25+30 = 100s before it gives up and raises). A cooldown
# shorter than that (an earlier version of this constant used 60s) is not a
# real throttle at all: a date shared by several games in the same run (a
# normal NBA slate is 8-12 games on one date) would have each of those
# games independently pay the full ~100s retry cost back to back, since by
# the time game 2 asks "has the cooldown elapsed yet?" the ~100s the FIRST
# game's own retries already burned has usually exceeded a 60s cooldown on
# its own -- silently defeating the whole point of this cache and making a
# single bad date cost minutes instead of the one ~100s hit it should.
_failed_point_in_time_dates: dict[tuple[str, str], dict] = {}
_POINT_IN_TIME_BASE_COOLDOWN_SECONDS = 180.0
_POINT_IN_TIME_MAX_COOLDOWN_SECONDS = 1800.0


def _point_in_time_cooldown_seconds(attempts_so_far: int) -> float:
    """Cooldown before a date that has already failed `attempts_so_far`
    times is worth trying again: doubles each time, capped at
    _POINT_IN_TIME_MAX_COOLDOWN_SECONDS, so a persistently-failing date is
    retried less and less often over a long run instead of either being
    hammered constantly or given up on forever.
    """
    return min(_POINT_IN_TIME_BASE_COOLDOWN_SECONDS * (2 ** max(0, attempts_so_far - 1)),
               _POINT_IN_TIME_MAX_COOLDOWN_SECONDS)


def resolve_season_for_date(game_date_str: str) -> str:
    """Real NBA season label ("YYYY-YY") for a specific calendar date
    ("YYYY-MM-DD"), reusing backend/api_simulation.py's own season-
    boundary rule (the season label flips over in October) instead of
    re-deriving it -- see resolve_current_nba_season()'s docstring there.
    This is the "Season-Matched Prior Alignment" requirement: a game
    played in April 2025 resolves to "2024-25", NEVER whatever season is
    live today, regardless of when this backtest itself is actually run.
    """
    import datetime
    from backend import api_simulation as sim
    return sim.resolve_current_nba_season(datetime.date.fromisoformat(game_date_str))


def _season_start_date_str(season: str) -> str:
    """A safe, real lower bound ("MM/DD/YYYY", nba_api's expected format)
    for a season's OWN games -- nba_api simply returns zero games for any
    date before the season actually tipped off, so this doesn't need to be
    the exact tip-off date, just guaranteed to precede it.
    """
    start_year = int(season.split("-")[0])
    return f"10/01/{start_year}"


def fetch_point_in_time_efficiency_stats(season: str, as_of_date_exclusive: str) -> dict[str, dict]:
    """Point-in-Time Rolling Statistics (No Future Leakage) -- REAL
    OFF_RATING/DEF_RATING/NET_RATING/PACE for every team, computed ONLY
    from that team's real games played from the start of `season` up
    through the day BEFORE `as_of_date_exclusive` ("YYYY-MM-DD", a game's
    own game_date) -- i.e. exactly what was genuinely knowable on the eve
    of that specific historical matchup, never that game itself or any
    later one. Delegates the actual nba_api call to
    backend/api_simulation.py's `_fetch_team_efficiency_stats` (now
    date_from/date_to-aware -- see that function's docstring), so the
    live API and this backtest share one real data-fetching code path,
    just pointed at a different date window.

    Cached per (season, cutoff date) -- a compressed real schedule window
    (several held-out games on the same date, or consecutive dates) reuses
    one nba_api call rather than repeating it per game.
    """
    import datetime
    from backend import api_simulation as sim

    as_of = datetime.date.fromisoformat(as_of_date_exclusive)
    cutoff = as_of - datetime.timedelta(days=1)
    cache_key = (season, cutoff.isoformat())
    if cache_key in _point_in_time_cache:
        return _point_in_time_cache[cache_key]

    print(f"  [point-in-time] Fetching real {season} team stats as of {cutoff.isoformat()} "
          f"(cutoff for games on {as_of.isoformat()})...", flush=True)

    # Several retries with a real 10-30s backoff absorb a transient nba_api
    # read-timeout/rate-limit here (same convention as the heavier
    # player-gamelog fetch below) -- cheap insurance against writing off a
    # whole date (and every game that shares it) over one unlucky request,
    # before apply_safety_brake()'s _failed_point_in_time_dates bookkeeping
    # takes over to stop hammering a date that's still failing across
    # separate call-level attempts of its own. ReadTimeout/ConnectTimeout/
    # ConnectionError are the specific transient failure modes this targets
    # (nba_api's stats.nba.com endpoint is a real, occasionally slow/rate-
    # limited external service); any other exception (a genuine bug, a
    # malformed response, etc.) is also retried the same way rather than
    # crashing the whole backtest over one bad date, but is logged with its
    # full traceback (not just str(e)) so a real bug is still diagnosable.
    import time
    import traceback as _traceback
    from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError as RequestsConnectionError
    _RETRY_BACKOFF_SECONDS = [10.0, 15.0, 20.0, 25.0, 30.0]
    last_exc: Optional[Exception] = None
    for attempt in range(len(_RETRY_BACKOFF_SECONDS) + 1):
        try:
            data = sim._fetch_team_efficiency_stats(
                season, date_from=_season_start_date_str(season), date_to=cutoff.strftime("%m/%d/%Y"))
            _point_in_time_cache[cache_key] = data
            return data
        except (ReadTimeout, ConnectTimeout, RequestsConnectionError) as e:
            last_exc = e
            print(f"  [point-in-time] {type(e).__name__} on attempt {attempt + 1}/"
                  f"{len(_RETRY_BACKOFF_SECONDS) + 1} for {cutoff.isoformat()}: {e}", flush=True)
            if attempt < len(_RETRY_BACKOFF_SECONDS):
                wait_s = _RETRY_BACKOFF_SECONDS[attempt]
                print(f"  [point-in-time] retrying in {wait_s:.0f}s...", flush=True)
                time.sleep(wait_s)
        except Exception as e:
            last_exc = e
            print(f"  [point-in-time] Unexpected {type(e).__name__} on attempt {attempt + 1}/"
                  f"{len(_RETRY_BACKOFF_SECONDS) + 1} for {cutoff.isoformat()}:\n"
                  f"{_traceback.format_exc()}", flush=True)
            if attempt < len(_RETRY_BACKOFF_SECONDS):
                wait_s = _RETRY_BACKOFF_SECONDS[attempt]
                print(f"  [point-in-time] retrying in {wait_s:.0f}s...", flush=True)
                time.sleep(wait_s)
    raise last_exc


# ---------------------------------------------------------------------------
# Point-in-Time PLAYER-LEVEL rosters (No Future Leakage, player granularity)
#
# The rest of this file's point-in-time discipline (fetch_point_in_time_
# efficiency_stats above) only fixed the MARKET-PRIOR side: each team's
# real NET_RATING/OFF_RATING/DEF_RATING/PACE, as of that game's own D-1.
# The raw C++/CUDA engine's own inputs -- fed via /api/players and
# /api/team_defense in every other mode in this file -- still reflect
# whatever season the backing Postgres DB was last loaded with, not that
# historical game's real point-in-time roster (train_ml_model.py's own
# module docstring already documents this same gap for its ML features).
# The functions below close that gap for the RAW SIMULATION step itself:
# they build a real, point-in-time-only --custom-roster JSON per game and
# feed the engine THAT instead of live team abbreviations.
#
# The hard part is TEAM ATTRIBUTION for a traded player. nba_api's own
# aggregate endpoint (leaguedashteamstats's player cousin,
# leaguedashplayerstats) attributes a traded player's ENTIRE season --
# including games played for their OLD team -- to whichever team is
# "current" for them, regardless of the date_from/date_to window
# requested: verified empirically against a real 2024-25 in-season trade
# (De'Aaron Fox, SAC -> SAS) -- querying a date range ENTIRELY BEFORE the
# trade still returned him under SAS. Using that endpoint here would
# silently misattribute pre-trade production to the wrong team, exactly
# the kind of leakage this whole exercise exists to eliminate.
# `playergamelogs` (bulk, one row per player per REAL game actually
# played) does not have this problem -- each row carries the real team
# that specific game was played for -- so point-in-time rosters are built
# by fetching that raw per-game log and aggregating it HERE, ourselves,
# rather than trusting nba_api's own (buggy-for-trades) aggregation.
# Bounded, not a plain growing dict: this cache's VALUES are full
# league-wide per-game-log DataFrames (thousands of rows for a late-season
# cutoff), one per distinct date needed -- unlike the small team-stats
# cache above, letting this one grow for an entire long run risks real
# memory pressure (observed in practice: a full run was killed by the
# system for low memory). Games are processed in chronological order (see
# fetch_real_nba_data.py's own sort, which load_games() doesn't re-order),
# so the cutoff date is non-decreasing across a run -- keeping just the
# last few distinct entries (not all of them) still captures the real
# win (many games sharing one date), without the unbounded growth.
_PLAYER_GAMELOG_CACHE_MAX_ENTRIES = 3
_player_gamelog_cache: "OrderedDict[tuple[str, str], pd.DataFrame]" = OrderedDict()
_position_lookup_cache: Optional[dict[str, str]] = None


def _bounded_cache_put(cache, key, value, max_entries: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)

# Matches build_gpu_roster()'s rotation cap in cuda_main.cpp / train_ml_model.py's
# MAX_ROTATION_PLAYERS.
MAX_ROTATION_PLAYERS = 9

# Real per-game/per-player league-average fallbacks -- same neutral
# defaults the C++ engine itself falls back to (see cpp_engine/main.cpp's
# Player struct) -- used only when a specific stat is NaN for a player's
# point-in-time window (e.g. zero 3PT attempts in their games so far this
# season, so FG3_PCT has no real rate to report yet).
_PIT_FALLBACKS = {
    "min": 20.0, "usage_rate": 20.0, "fg3a": 0.0, "fg3_pct": 0.35, "fg_pct": 0.46,
    "ft_pct": 0.77, "blk": 0.5, "stl": 1.0, "ast": 4.5, "oreb": 2.0, "dreb": 6.5,
    "fta": 3.0, "pf": 2.0,
}


def get_position_lookup(api_url: str) -> dict[str, str]:
    """Real player -> position map, keyed by normalized name. Position is
    NOT treated as a point-in-time-sensitive field (a player's listed
    position essentially never changes within a season, and rarely across
    a career, in any way that would leak FUTURE performance information
    into a PAST prediction) -- pulled once from the live /api/players
    roster (already-established real data, just not date-scoped) and
    reused for every game, same "not every field needs the same rigor"
    judgment call this project already makes elsewhere. A player not
    found here (retired, G-League call-up with too few games to appear
    in the live roster snapshot) falls back to "SG", the engine's own
    neutral default.
    """
    global _position_lookup_cache
    if _position_lookup_cache is not None:
        return _position_lookup_cache
    import train_ml_model as ml
    players = ml.fetch_players(api_url)
    _position_lookup_cache = {
        ml._normalize_name(p.get("player_name", "")): p.get("position", "SG") or "SG"
        for p in players
    }
    return _position_lookup_cache


def fetch_point_in_time_player_gamelogs(season: str, as_of_date_exclusive: str) -> "pd.DataFrame":
    """Every real player-game row (Base + Advanced/USG_PCT merged) from the
    start of `season` through the day BEFORE `as_of_date_exclusive` -- see
    the module comment block above for why this is built from raw
    per-game rows (playergamelogs) rather than nba_api's own aggregate
    endpoint. Cached per (season, cutoff date), same convention as
    fetch_point_in_time_efficiency_stats() -- a compressed schedule
    window reuses one fetch rather than repeating it per game. This
    single fetch grows with how far into the season the cutoff is (a
    March cutoff pulls ~5 months of league-wide game logs), so it is
    naturally the most expensive point-in-time call in this file -- still
    just one real nba_api round trip either way, not one per player.
    """
    import pandas as pd
    from nba_api.stats.endpoints import playergamelogs

    as_of = datetime.date.fromisoformat(as_of_date_exclusive)
    cutoff = as_of - datetime.timedelta(days=1)
    cache_key = (season, cutoff.isoformat())
    if cache_key in _player_gamelog_cache:
        _player_gamelog_cache.move_to_end(cache_key)
        return _player_gamelog_cache[cache_key]

    date_from = _season_start_date_str(season)
    date_to = cutoff.strftime("%m/%d/%Y")
    print(f"  [point-in-time] Fetching real per-game player logs for {season} "
          f"through {cutoff.isoformat()} (this grows with season progress)...", flush=True)

    def _fetch_with_retry(**kwargs):
        # stats.nba.com is a real, occasionally-flaky external service,
        # more so for this endpoint than the lighter team-level ones (it
        # returns every player-game row in the window, growing large late
        # in a season) -- several retries with a real 10-30s backoff (same
        # convention as fetch_point_in_time_efficiency_stats() above) absorb
        # a transient read-timeout/rate-limit without needing to skip (and
        # permanently lose) the whole game this cutoff date was needed for.
        # Exhausting all attempts still raises (with the full traceback
        # logged, not just str(e)), which evaluate_game_point_in_time()
        # catches and converts into a graceful per-game skip.
        import time
        from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError as RequestsConnectionError
        backoff = [10.0, 15.0, 20.0, 25.0, 30.0]
        last_exc = None
        for attempt in range(len(backoff) + 1):
            try:
                return playergamelogs.PlayerGameLogs(timeout=60, **kwargs).get_data_frames()[0]
            except (ReadTimeout, ConnectTimeout, RequestsConnectionError) as e:
                last_exc = e
                print(f"  [point-in-time] {type(e).__name__} fetching player logs "
                      f"(attempt {attempt + 1}/{len(backoff) + 1}): {e}", flush=True)
            except Exception as e:
                last_exc = e
                print(f"  [point-in-time] Unexpected {type(e).__name__} fetching player logs "
                      f"(attempt {attempt + 1}/{len(backoff) + 1}):\n{traceback.format_exc()}", flush=True)
            if attempt < len(backoff):
                wait_s = backoff[attempt]
                print(f"  [point-in-time] retrying in {wait_s:.0f}s...", flush=True)
                time.sleep(wait_s)
        raise last_exc

    base = _fetch_with_retry(season_nullable=season, season_type_nullable="Regular Season",
                              date_from_nullable=date_from, date_to_nullable=date_to)
    if base.empty:
        _bounded_cache_put(_player_gamelog_cache, cache_key, base, _PLAYER_GAMELOG_CACHE_MAX_ENTRIES)
        return base
    adv = _fetch_with_retry(season_nullable=season, season_type_nullable="Regular Season",
                             measure_type_player_game_logs_nullable="Advanced",
                             date_from_nullable=date_from, date_to_nullable=date_to)

    usg = adv.set_index(["PLAYER_ID", "GAME_ID"])["USG_PCT"]
    base = base.set_index(["PLAYER_ID", "GAME_ID"])
    base["USG_PCT"] = usg
    base = base.reset_index()
    _bounded_cache_put(_player_gamelog_cache, cache_key, base, _PLAYER_GAMELOG_CACHE_MAX_ENTRIES)
    return base


def build_point_in_time_roster(team_abbr: str, gamelogs: "pd.DataFrame",
                                position_lookup: dict[str, str],
                                max_rotation: int = MAX_ROTATION_PLAYERS,
                                rolling_window_games: Optional[int] = None) -> list[dict]:
    """Real per-player point-in-time averages for `team_abbr`'s top-
    `max_rotation`-by-minutes rotation, aggregated ONLY from real games
    THIS team's roster actually played (see the module comment block
    above for why a traded player is correctly split by real game-level
    team, not misattributed). Empty if this team has no real games yet in
    the window (very start of a season).

    `rolling_window_games`, when given, restricts EACH PLAYER's own
    average to their most recent `rolling_window_games` real games
    actually played (not a fixed calendar-date cutoff, which would be
    thrown off by that specific player's own missed games/rest) --
    trading some sample size for recency: a season-to-date average (the
    default, `None`) is lower-variance but can lag a real form change
    (a trade, a return from injury, an in-season improvement); a rolling
    window is noisier per-player but reflects the team's CURRENT real
    rotation and form more directly. Still strictly leak-free either way
    -- both only ever look at real games strictly before the target date.
    """
    if gamelogs.empty:
        return []
    team_logs = gamelogs[gamelogs["TEAM_ABBREVIATION"] == team_abbr]
    if team_logs.empty:
        return []

    if rolling_window_games is not None:
        team_logs = (
            team_logs.sort_values("GAME_DATE", ascending=False)
            .groupby("PLAYER_NAME", group_keys=False)
            .head(rolling_window_games)
        )

    grouped = team_logs.groupby("PLAYER_NAME").agg(
        min=("MIN", "mean"), fg3a=("FG3A", "mean"), fg3_pct=("FG3_PCT", "mean"),
        fg_pct=("FG_PCT", "mean"), ft_pct=("FT_PCT", "mean"),
        oreb=("OREB", "mean"), dreb=("DREB", "mean"), ast=("AST", "mean"),
        stl=("STL", "mean"), blk=("BLK", "mean"), pf=("PF", "mean"), fta=("FTA", "mean"),
        usage_rate=("USG_PCT", "mean"),
    ).reset_index()
    # pandas' mean() already skips NaN (e.g. a game with 0 FG3A leaves
    # FG3_PCT undefined for that row) -- an "average of games with a real
    # attempt," not a zero-filled one, matching this project's existing
    # "empirical rate" convention for fg_pct/fg3_pct/ft_pct everywhere else.
    grouped = grouped.sort_values("min", ascending=False).head(max_rotation)

    roster = []
    for row in grouped.itertuples():
        def _val(x, key):
            return float(x) if x == x else _PIT_FALLBACKS[key]  # x==x is False only for NaN

        pos = position_lookup.get(_norm_name(row.PLAYER_NAME), "SG")
        usage_rate = float(row.usage_rate) * 100.0 if row.usage_rate == row.usage_rate else _PIT_FALLBACKS["usage_rate"]
        roster.append({
            "player_name": row.PLAYER_NAME,
            "position": pos,
            "min": _val(row.min, "min"),
            "usage_rate": usage_rate,
            "fg3a": _val(row.fg3a, "fg3a"),
            "fg3_pct": _val(row.fg3_pct, "fg3_pct"),
            "fg_pct": _val(row.fg_pct, "fg_pct"),
            "ft_pct": _val(row.ft_pct, "ft_pct"),
            "rim_protection_gravity": _val(row.blk, "blk"),
            "help_defense_iq": _val(row.stl, "stl"),
            "playmaking_gravity": _val(row.ast, "ast"),
            "oreb_gravity": _val(row.oreb, "oreb"),
            "dreb_gravity": _val(row.dreb, "dreb"),
            "drive_gravity_rating": _val(row.fta, "fta"),
            "personal_fouls_rate": _val(row.pf, "pf"),
        })
    return roster


def _norm_name(name: str) -> str:
    import train_ml_model as ml
    return ml._normalize_name(name)


def build_point_in_time_custom_roster_payload(team_a: str, team_b: str, game_date: str,
                                               api_url: str,
                                               rolling_window_games: Optional[int] = None) -> Optional[dict]:
    """Full --custom-roster JSON payload for one historical game, built
    entirely from real data available STRICTLY BEFORE that game's own
    date -- both team-level (real NET_RATING/OFF_RATING/DEF_RATING/PACE,
    reusing fetch_point_in_time_efficiency_stats -- the exact same values
    the Safety Brake itself uses, so this is season/date-consistent with
    that mechanism too) and now player-level (real per-player rotation
    averages, see build_point_in_time_roster -- `rolling_window_games`
    forwards straight through to it). Returns None when either team has
    no real point-in-time roster yet (the season's first few games) --
    the caller falls back to skipping/erroring that one game rather than
    simulating from a fabricated roster.
    """
    season = resolve_season_for_date(game_date)
    gamelogs = fetch_point_in_time_player_gamelogs(season, game_date)
    position_lookup = get_position_lookup(api_url)
    roster_a = build_point_in_time_roster(team_a, gamelogs, position_lookup,
                                           rolling_window_games=rolling_window_games)
    roster_b = build_point_in_time_roster(team_b, gamelogs, position_lookup,
                                           rolling_window_games=rolling_window_games)
    if not roster_a or not roster_b:
        return None

    payload = {
        "team_a_name": team_a, "team_a_roster": roster_a,
        "team_b_name": team_b, "team_b_roster": roster_b,
    }
    try:
        team_stats = fetch_point_in_time_efficiency_stats(season, game_date)
        if team_a in team_stats:
            payload["team_a_def_rating"] = team_stats[team_a]["def_rating"]
        if team_b in team_stats:
            payload["team_b_def_rating"] = team_stats[team_b]["def_rating"]
    except Exception:
        pass  # def_rating is optional on --custom-roster -- engine falls back to neutral
    return payload


def evaluate_game_point_in_time(game: HistoricalGame, exe: Path, timeout: float, verbose: bool,
                                 api_url: str,
                                 is_team_a_b2b: bool = False, is_team_b_b2b: bool = False,
                                 rolling_window_games: Optional[int] = None) -> GameResult:
    """Full point-in-time evaluation: builds a real --custom-roster JSON
    from ONLY data available before `game.game_date` (team AND player
    level -- see build_point_in_time_custom_roster_payload()) and runs
    the raw engine against THAT, instead of live team abbreviations
    resolved against whatever season /api/players currently serves.
    Raises BacktestError if no real point-in-time roster exists yet for
    either team (too early in the season). `rolling_window_games` forwards
    straight through -- see build_point_in_time_roster()'s docstring.
    """
    if not game.game_date:
        raise BacktestError(f"Game {game.team_a} vs {game.team_b} has no game_date.")

    try:
        payload = build_point_in_time_custom_roster_payload(
            game.team_a, game.team_b, game.game_date, api_url, rolling_window_games=rolling_window_games)
    except BacktestError:
        raise
    except Exception as e:
        # A real nba_api network hiccup (read timeout, rate limit, etc.)
        # fetching the per-game player log for this date -- deliberately
        # a bare `except Exception`, converted to BacktestError so the
        # per-game loop in run_backtest_pass_point_in_time() degrades
        # gracefully (skip just this one game) instead of crashing the
        # entire run, matching apply_safety_brake()'s own resilience
        # convention for the exact same class of failure.
        raise BacktestError(
            f"Could not fetch point-in-time player data for {game.team_a} vs {game.team_b} "
            f"on {game.game_date}: {e}"
        ) from e
    if payload is None:
        raise BacktestError(
            f"No real point-in-time roster data yet for {game.team_a} vs {game.team_b} "
            f"on {game.game_date} (too early in the season)."
        )

    import tempfile
    fd, path_str = tempfile.mkstemp(prefix="nba_pit_roster_", suffix=".json")
    roster_path = Path(path_str)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        stdout = run_simulation(exe, game.team_a, game.team_b, timeout,
                                 is_team_a_home=True, is_team_a_b2b=is_team_a_b2b,
                                 is_team_b_b2b=is_team_b_b2b, custom_roster_path=roster_path)
        if verbose:
            print(f"\n----- raw output (point-in-time roster): {game.team_a} vs {game.team_b} -----")
            print(stdout)
            print("----- end raw output -----\n")

        resolved_a, resolved_b, win_prob, avg_score = parse_gpu_block(stdout)
        if {resolved_a, resolved_b} != {game.team_a, game.team_b}:
            raise BacktestError(
                f"Simulator resolved a different matchup ({resolved_a} vs {resolved_b}) than "
                f"requested ({game.team_a} vs {game.team_b})."
            )
        if game.actual_winner not in (game.team_a, game.team_b):
            raise BacktestError(f"actual_winner '{game.actual_winner}' is not '{game.team_a}' or '{game.team_b}'.")

        predicted_favorite = game.team_a if win_prob[game.team_a] >= win_prob[game.team_b] else game.team_b
        predicted_margin = avg_score[game.team_a] - avg_score[game.team_b]
        actual_margin = game.actual_score_a - game.actual_score_b
        correct = predicted_favorite == game.actual_winner
        p_actual_winner = win_prob[game.actual_winner]
        brier = (p_actual_winner - 1.0) ** 2
        margin_error = abs(predicted_margin - actual_margin)

        return GameResult(
            game=game, win_prob=win_prob, avg_score=avg_score,
            predicted_favorite=predicted_favorite, predicted_margin=predicted_margin,
            correct=correct, brier=brier, margin_error=margin_error,
            is_team_a_home=True, is_team_a_b2b=is_team_a_b2b, is_team_b_b2b=is_team_b_b2b,
        )
    finally:
        roster_path.unlink(missing_ok=True)


def run_backtest_pass_point_in_time(title: str, games: list[HistoricalGame], exe: Path, args,
                                     show_rows: Optional[bool] = None,
                                     rolling_window_games: Optional[int] = None
                                     ) -> tuple[list[GameResult], Optional[dict]]:
    """Same shape/reporting as run_backtest_pass(), but scores each game
    with evaluate_game_point_in_time() -- a real, point-in-time-only
    --custom-roster (team AND player level) instead of live team
    abbreviations. Games with no real point-in-time roster yet (too early
    in a season) are skipped and reported, same graceful-degradation
    convention as a timeout/parse error elsewhere in this file.
    `rolling_window_games` forwards straight through -- see
    build_point_in_time_roster()'s docstring.
    """
    if show_rows is None:
        show_rows = len(games) <= MAX_ROWS_TO_PRINT

    print(f"\n--- {title} ---")
    if show_rows:
        header = f"{'#':>3}  {'Matchup':<10} {'Actual':<15} {'Fav':<5}{'P(win)':>7}  {'Pred Marg':>10}  {'Actual Marg':>12}  Correct"
        print(header)
        print("-" * len(header))
    else:
        print(f" ({len(games)} games -- per-game rows suppressed for readability; "
              f"pass --verbose for full detail. Running accuracy every {_PROGRESS_STEP} games:)")

    results: list[GameResult] = []
    correct_so_far = 0
    for i, game in enumerate(games, start=1):
        is_a_b2b = bool(game.team_a_rest_days == 0)
        is_b_b2b = bool(game.team_b_rest_days == 0)
        try:
            result = evaluate_game_point_in_time(game, exe, args.timeout, args.verbose, args.api_url,
                                                  is_a_b2b, is_b_b2b, rolling_window_games=rolling_window_games)
        except subprocess.TimeoutExpired:
            if show_rows:
                print_game_row(i, game, None, f"timed out after {args.timeout:.0f}s")
            continue
        except BacktestError as e:
            if show_rows:
                print_game_row(i, game, None, str(e))
            else:
                print(f"   ...skipped game {i} ({game.team_a} vs {game.team_b}, {game.game_date}): {e}")
            continue

        results.append(result)
        correct_so_far += int(result.correct)
        if show_rows:
            print_game_row(i, game, result, None)
        elif i % _PROGRESS_STEP == 0 or i == len(games):
            print(f"   ...{i:>4}/{len(games)} scored, running accuracy {100.0 * correct_so_far / max(1, len(results)):.1f}%")

    print(f" (--home-a applied to every game (team_a); B2B = a team on 0 rest days, --b2b-a/--b2b-b applied;"
          f" {len(games) - len(results)} game(s) skipped -- no real point-in-time roster yet)")

    metrics = print_summary(title, games, results)
    return results, metrics


def apply_safety_brake(results: list[GameResult]) -> list[GameResult]:
    """Mode A: Blended (Market-Prior Safety Brake ON) -- takes the SAME raw
    Monte Carlo predictions already produced for Mode B (no second, more
    expensive C++/GPU run needed, since the brake is a pure Python
    post-process over an existing raw win probability -- exactly how
    backend/api_simulation.py's live request path works), and blends each
    game's win probability toward a POINT-IN-TIME structural market prior
    -- that game's own real season, using ONLY real stats from games
    played strictly before it (see fetch_point_in_time_efficiency_stats())
    -- using the EXACT SAME functions the live backend uses
    (`compute_market_prior`/`compute_dynamic_raw_weight`/
    `apply_bayesian_shrinkage`), so this comparison can never silently
    drift from the live API's actual blending behavior, only from WHEN its
    input snapshot is taken.

    This eliminates both failure modes the previous version of this
    function had: DATA LEAKAGE (a team's own later, not-yet-played games
    could never influence its prior for an earlier game -- the date cutoff
    is strict and per-game) and SEASON MISMATCH (a 2024-25 game is always
    blended against a real 2024-25 rolling snapshot, never today's live
    season, regardless of when this backtest is actually run).

    Deliberately leaves `avg_score`/`predicted_margin`/`margin_error`
    UNTOUCHED: the Safety Brake only ever blends the WIN PROBABILITY
    (backend/api_simulation.py never touches `average_score_a`/`_b`), so
    Point-Differential MAE is mathematically IDENTICAL between Mode A and
    Mode B by construction -- not a bug, and reported as such rather than
    silently hidden.

    Games for either team missing a real point-in-time sample (too early
    in a season for nba_api to return that team at all, or a name
    mismatch) keep their RAW win probability unchanged (mirrors
    backend/api_simulation.py's own graceful "shrinkage skipped"
    degradation for the same situation) -- they're still scored, just
    without a brake applied to that one game.
    """
    import time
    from backend import api_simulation as sim

    blended: list[GameResult] = []
    for r in results:
        team_a, team_b = r.game.team_a, r.game.team_b
        if not r.game.game_date:
            raise BacktestError(
                f"Game {team_a} vs {team_b} has no game_date -- cannot compute a point-in-time "
                "prior for it (Season-Matched Prior Alignment requires a real date to anchor to)."
            )

        season = resolve_season_for_date(r.game.game_date)
        cutoff_date = (datetime.date.fromisoformat(r.game.game_date) - datetime.timedelta(days=1)).isoformat()
        fail_key = (season, cutoff_date)
        failure = _failed_point_in_time_dates.get(fail_key)
        if failure is not None:
            cooldown = _point_in_time_cooldown_seconds(failure["attempts"])
            cooling_down = (time.monotonic() - failure["last_failed_at"]) < cooldown
            if cooling_down:
                # Tried too recently to be worth another real attempt yet
                # (see _failed_point_in_time_dates' own comment) -- skip
                # straight to the graceful fallback instead of repeating a
                # request that's very likely still doomed. NOT a permanent
                # skip -- this same date becomes eligible again once its
                # cooldown elapses, however many games from now that is.
                blended.append(r)
                continue
        try:
            stats = fetch_point_in_time_efficiency_stats(season, r.game.game_date)
        except Exception as e:
            prev_attempts = failure["attempts"] if failure is not None else 0
            attempts_now = prev_attempts + 1
            next_cooldown = _point_in_time_cooldown_seconds(attempts_now)
            print(f"  [safety-brake] Warning: could not fetch point-in-time stats for "
                  f"{team_a} vs {team_b} ({r.game.game_date}, season {season}) after "
                  f"fetch_point_in_time_efficiency_stats()'s own retries were exhausted -- "
                  f"skipping the brake for this game (raw probability kept); this date will "
                  f"be retried again after a {next_cooldown:.0f}s cooldown (failure "
                  f"#{attempts_now} for this date this run). Full error:\n"
                  f"{traceback.format_exc()}", flush=True)
            _failed_point_in_time_dates[fail_key] = {"attempts": attempts_now, "last_failed_at": time.monotonic()}
            blended.append(r)
            continue

        stats_a = stats.get(team_a)
        stats_b = stats.get(team_b)
        if stats_a is None or stats_b is None:
            blended.append(r)
            continue

        # team_a is always the home team in this dataset's convention (see
        # fetch_real_nba_data.py) -- matches run_backtest_pass's own
        # is_team_a_home=True always, and mirrors api_simulation.py's
        # req.home_team=="a" branch exactly.
        home_court_margin_a = sim.kHomeCourtPointsPrior

        win_a_raw = r.win_prob[team_a]
        win_b_raw = r.win_prob[team_b]
        non_tie = win_a_raw + win_b_raw
        frac_a = (win_a_raw / non_tie) if non_tie > 0 else 0.5

        prior = sim.compute_market_prior(stats_a, stats_b, home_court_margin_a)
        delta_net_rating = abs(stats_a["net_rating"] - stats_b["net_rating"])
        dynamic_weight = sim.compute_dynamic_raw_weight(delta_net_rating)
        blended_frac_a = sim.apply_bayesian_shrinkage(frac_a, prior["prior_prob_a"], dynamic_weight)

        new_win_prob = {
            team_a: blended_frac_a * non_tie,
            team_b: (1.0 - blended_frac_a) * non_tie,
        }
        predicted_favorite = team_a if new_win_prob[team_a] >= new_win_prob[team_b] else team_b
        correct = predicted_favorite == r.game.actual_winner
        p_actual_winner = new_win_prob[r.game.actual_winner]
        brier = (p_actual_winner - 1.0) ** 2

        blended.append(dataclasses.replace(
            r,
            win_prob=new_win_prob,
            predicted_favorite=predicted_favorite,
            correct=correct,
            brier=brier,
            shrinkage_weight=dynamic_weight,
            delta_net_rating=delta_net_rating,
            cutoff_date=cutoff_date,
        ))
    return blended


def compute_ml_margins(csv_path: Path, api_url: str, ml_dir: Path,
                        test_fraction: float, split_mode: str
                        ) -> tuple[list[HistoricalGame], list[float], list[bool], dict]:
    """Trains train_ml_model.py fresh against this exact CSV/roster data and
    returns the **held-out** games (chronological test split, or every game
    under leave-one-out -- see train_ml_model.py) together with their
    predicted margins and marquee flags (for --hot-hand-boost), plus the
    run's split metadata for reporting.

    Games are built directly from train_ml_model's saved predictions rather
    than matched back against a separately-loaded game list, so there's no
    way for the two to drift out of alignment. This is what keeps the
    backtest honest: every margin here comes from a model that never saw
    that specific game's outcome during training, and (in the default
    chronological mode) never saw games chronologically *after* it either.
    """
    import train_ml_model as ml

    try:
        ml.train(csv_path, api_url, ml_dir, test_fraction=test_fraction, split_mode=split_mode)
    except ml.MlPipelineError as e:
        raise BacktestError(f"ML margin training failed: {e}") from e

    holdout_path = ml_dir / "holdout_predictions.json"
    with holdout_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    holdout_games: list[HistoricalGame] = []
    margins: list[float] = []
    is_marquee_flags: list[bool] = []
    for entry in payload["games"]:
        holdout_games.append(HistoricalGame(
            team_a=entry["team_a"],
            team_b=entry["team_b"],
            actual_winner=entry["actual_winner"],
            actual_score_a=float(entry["actual_score_a"]),
            actual_score_b=float(entry["actual_score_b"]),
            game_date=entry.get("game_date", ""),
            team_a_rest_days=entry.get("team_a_rest_days"),
            team_b_rest_days=entry.get("team_b_rest_days"),
        ))
        margins.append(float(entry["holdout_predicted_margin"]))
        is_marquee_flags.append(bool(entry.get("is_high_leverage", entry.get("is_marquee", False))))

    meta = {
        "split_mode": payload["split_mode"],
        "split_info": payload["split_info"],
        "eval_report": payload["eval_report"],
    }
    return holdout_games, margins, is_marquee_flags, meta


def print_game_row(i: int, game: HistoricalGame, result: Optional[GameResult], error: Optional[str]) -> None:
    matchup = f"{game.team_a}-{game.team_b}"
    if error is not None:
        print(f"{i:>3}  {matchup:<10} ERROR: {error}")
        return

    assert result is not None
    actual = f"{game.actual_winner} {game.actual_score_a:.0f}-{game.actual_score_b:.0f}"
    actual_margin = game.actual_score_a - game.actual_score_b
    p_fav = result.win_prob[result.predicted_favorite]
    flags = ""
    if result.hot_hand_boost is not None:
        flags += " M"
    if result.is_team_a_b2b or result.is_team_b_b2b:
        flags += " B2B"
    print(
        f"{i:>3}  {matchup:<10} {actual:<15} "
        f"{result.predicted_favorite:<5}{p_fav * 100:>6.1f}%  "
        f"pred {result.predicted_margin:>+6.2f}  actual {actual_margin:>+6.1f}  "
        f"{'YES' if result.correct else 'no'}{flags}"
    )


def print_summary(title: str, all_games: list[HistoricalGame], results: list[GameResult]) -> Optional[dict]:
    print("\n========================================================")
    print(f" BACKTEST SUMMARY: {title}")
    print("========================================================")
    print(f" Games in dataset            : {len(all_games)}")
    skipped = len(all_games) - len(results)
    scored_line = f" Games successfully scored   : {len(results)}"
    if skipped:
        scored_line += f"  ({skipped} skipped due to errors)"
    print(scored_line)

    if not results:
        print(" No games were successfully scored -- no metrics to report.")
        print("========================================================\n")
        return None

    correct_count = sum(1 for r in results if r.correct)
    accuracy = 100.0 * correct_count / len(results)
    avg_brier = mean(r.brier for r in results)
    mae = mean(r.margin_error for r in results)

    print("--------------------------------------------------------")
    print(f" Win/Loss accuracy           : {accuracy:.1f}%  ({correct_count}/{len(results)} correct)")
    print(f" Average Brier score         : {avg_brier:.4f}  (0=perfect, 0.25=coin-flip, 1=worst)")
    print(f" Point differential MAE      : {mae:.2f} points")
    print("========================================================\n")

    return {"n": len(results), "accuracy": accuracy, "brier": avg_brier, "mae": mae}


def print_comparison(baseline: dict, hybrid: dict, threshold: float = 60.0) -> None:
    print("========================================================")
    print(" HYBRID (ML + MONTE CARLO) vs BASELINE (MONTE CARLO ONLY)")
    print(" -- both scored on the same held-out (never-trained-on) games --")
    print("========================================================")
    print(f" {'Metric':<28}{'Baseline':>12}{'Hybrid':>12}{'Delta':>12}")
    print(f" {'Win/Loss accuracy':<28}{baseline['accuracy']:>11.1f}%{hybrid['accuracy']:>11.1f}%"
          f"{hybrid['accuracy'] - baseline['accuracy']:>+11.1f}%")
    print(f" {'Average Brier score':<28}{baseline['brier']:>12.4f}{hybrid['brier']:>12.4f}"
          f"{hybrid['brier'] - baseline['brier']:>+12.4f}")
    print(f" {'Point differential MAE':<28}{baseline['mae']:>11.2f} {hybrid['mae']:>11.2f} "
          f"{hybrid['mae'] - baseline['mae']:>+11.2f}")
    print("--------------------------------------------------------")
    print(" (accuracy higher is better; Brier score and MAE lower is better)")
    crossed = hybrid["accuracy"] > threshold
    print(f"\n Hybrid win/loss accuracy vs {threshold:.0f}% threshold: "
          f"{'CROSSED' if crossed else 'did NOT cross'} ({hybrid['accuracy']:.1f}%)")
    print("========================================================\n")


def compute_statistical_significance(raw_results: list[GameResult], blended_results: list[GameResult]) -> dict:
    """Confidence assessment for the accuracy/Brier lift: is it statistically
    robust, or plausibly a sample-size artifact? Both games lists are the
    SAME games in the SAME order (apply_safety_brake() only ever re-scores
    an existing raw_results list, never reorders/drops), so this is a
    PAIRED comparison -- the right statistical framing for "two classifiers
    scored on the identical items," not two independent samples. Pure
    computation, no printing -- see print_statistical_significance() for
    the human-readable report and build_full_report() for the JSON one,
    both built from this single source of truth so they can never drift
    apart.

    McNemar's test (accuracy): only the DISCORDANT games -- where the two
    modes disagreed on whether they got it right -- carry any information
    about which mode is genuinely better; games both modes got right (or
    both got wrong) are uninformative for this specific question. Uses the
    continuity-corrected chi-square statistic, converted to a p-value via
    the standard normal CDF (chi2 with 1 degree of freedom is exactly the
    square of a standard normal variable, so this needs no scipy
    dependency -- reuses backend/api_simulation.py's own `_normal_cdf`,
    the same no-extra-dependency convention that module already
    established for its win-probability conversion).

    Paired Brier-score difference: a standard paired z-test on the
    per-game (blended_brier - raw_brier) differences, reported as a mean
    difference with a 95% confidence interval.
    """
    from backend.api_simulation import _normal_cdf
    import math

    n = min(len(raw_results), len(blended_results))
    only_blended_correct = 0  # raw wrong, blended right
    only_raw_correct = 0      # raw right, blended wrong
    brier_diffs = []
    for r_raw, r_blend in zip(raw_results, blended_results):
        if r_raw.correct and not r_blend.correct:
            only_raw_correct += 1
        elif r_blend.correct and not r_raw.correct:
            only_blended_correct += 1
        brier_diffs.append(r_blend.brier - r_raw.brier)

    b, c = only_blended_correct, only_raw_correct
    result: dict = {
        "n": n,
        "only_blended_correct": b,
        "only_raw_correct": c,
        "mcnemar_chi2": None,
        "mcnemar_p_value": None,
        "mcnemar_significant_at_0_05": None,
        "brier_diff_mean": None,
        "brier_diff_ci_95": None,
        "brier_diff_p_value": None,
    }
    if b + c > 0:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected, df=1
        z = math.sqrt(chi2_stat)
        p_value = 2.0 * (1.0 - _normal_cdf(z))
        result["mcnemar_chi2"] = chi2_stat
        result["mcnemar_p_value"] = p_value
        result["mcnemar_significant_at_0_05"] = p_value < 0.05

    if n >= 2:
        mean_diff = mean(brier_diffs)
        var = sum((d - mean_diff) ** 2 for d in brier_diffs) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        result["brier_diff_mean"] = mean_diff
        if se > 0:
            result["brier_diff_ci_95"] = [mean_diff - 1.96 * se, mean_diff + 1.96 * se]
            z_brier = mean_diff / se
            result["brier_diff_p_value"] = 2.0 * (1.0 - _normal_cdf(abs(z_brier)))
    return result


def print_statistical_significance(sig: dict) -> None:
    """Human-readable report for compute_statistical_significance()'s output."""
    print("========================================================")
    print(" STATISTICAL SIGNIFICANCE (paired by game -- same N games, both modes)")
    print("========================================================")
    print(f" N (paired games)                         : {sig['n']}")
    print(f" Games ONLY Mode A (Blended) got right     : {sig['only_blended_correct']}")
    print(f" Games ONLY Mode B (Raw) got right         : {sig['only_raw_correct']}")
    if sig["mcnemar_chi2"] is None:
        print(" McNemar's test: no discordant games (the two modes never disagreed) -- ")
        print(" the accuracy figures are identical; no test to run.")
    else:
        print(f" McNemar's chi-square (continuity-corrected, df=1) : {sig['mcnemar_chi2']:.3f}")
        print(f" p-value                                    : {sig['mcnemar_p_value']:.4f}")
        verdict = "IS statistically significant" if sig["mcnemar_significant_at_0_05"] else "is NOT statistically significant"
        print(f" -> The accuracy difference {verdict} at alpha=0.05.")

    if sig["brier_diff_mean"] is not None:
        print("--------------------------------------------------------")
        print(" Paired Brier-score difference (Mode A - Mode B, negative = Blended better)")
        if sig["brier_diff_ci_95"] is not None:
            ci_lo, ci_hi = sig["brier_diff_ci_95"]
            print(f"   Mean difference : {sig['brier_diff_mean']:+.4f}  (95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}])")
            print(f"   p-value (paired z-test) : {sig['brier_diff_p_value']:.4f}")
        else:
            print(f"   Mean difference : {sig['brier_diff_mean']:+.4f}  (zero variance across games -- no CI to report)")
    print("========================================================")
    print(" CAVEAT: both tests assume the per-game outcomes are independent, which is only")
    print(" approximately true (teams and dates repeat within the sample). Treat the p-values")
    print(" as a rough confidence signal, not a rigorously i.i.d. hypothesis test.")
    print("========================================================\n")


def build_per_game_breakdown(raw_results: list[GameResult], blended_results: list[GameResult]) -> list[dict]:
    """Per-game breakdown for the JSON report -- Artifact Preservation:
    every game's real inputs/outputs in both modes, not just the aggregate
    metrics, so a reviewer can audit (or re-derive) any individual
    prediction later without re-running the backtest.
    """
    rows = []
    for r_raw, r_blend in zip(raw_results, blended_results):
        g = r_raw.game
        rows.append({
            "team_a": g.team_a, "team_b": g.team_b, "game_date": g.game_date,
            "actual_winner": g.actual_winner,
            "actual_score_a": g.actual_score_a, "actual_score_b": g.actual_score_b,
            "cutoff_date": r_blend.cutoff_date,
            "raw": {
                "favorite": r_raw.predicted_favorite, "win_prob": r_raw.win_prob,
                "predicted_margin": r_raw.predicted_margin, "correct": r_raw.correct, "brier": r_raw.brier,
            },
            "blended": {
                "favorite": r_blend.predicted_favorite, "win_prob": r_blend.win_prob,
                "predicted_margin": r_blend.predicted_margin, "correct": r_blend.correct, "brier": r_blend.brier,
                "shrinkage_weight": r_blend.shrinkage_weight, "delta_net_rating": r_blend.delta_net_rating,
            },
        })
    return rows


def build_full_report(csv_path: Path, games: list[HistoricalGame], raw_results: list[GameResult],
                       blended_results: list[GameResult], raw_metrics: dict, blended_metrics: dict,
                       sig: dict) -> dict:
    """Single structured artifact combining everything the terminal report
    prints -- dataset info, both modes' aggregate metrics, the comparison,
    statistical significance, dynamic-weight summary, and the full
    per-game breakdown -- built from the SAME already-computed values the
    printed report uses (no re-derivation, so the two can never disagree).
    """
    weights = [r.shrinkage_weight for r in blended_results if r.shrinkage_weight is not None]
    gaps = [r.delta_net_rating for r in blended_results if r.delta_net_rating is not None]
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(csv_path),
        "n_games": len(games),
        "n_scored": len(blended_results),
        "date_range": [games[0].game_date, games[-1].game_date] if games else None,
        "mode_b_raw": raw_metrics,
        "mode_a_blended": blended_metrics,
        "comparison": {
            "accuracy_delta": blended_metrics["accuracy"] - raw_metrics["accuracy"],
            "brier_delta": blended_metrics["brier"] - raw_metrics["brier"],
            "mae_delta": blended_metrics["mae"] - raw_metrics["mae"],
        },
        "statistical_significance": sig,
        "dynamic_weight_summary": {
            "n_with_real_net_rating_match": len(weights),
            "weight_min": min(weights) if weights else None,
            "weight_mean": mean(weights) if weights else None,
            "weight_max": max(weights) if weights else None,
            "delta_net_rating_min": min(gaps) if gaps else None,
            "delta_net_rating_mean": mean(gaps) if gaps else None,
            "delta_net_rating_max": max(gaps) if gaps else None,
        },
        "per_game": build_per_game_breakdown(raw_results, blended_results),
    }


def write_json_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote full JSON report ({len(report.get('per_game', []))} games) to {path}")


def print_brake_comparison(raw_metrics: dict, blended_metrics: dict, results_blended: list[GameResult]) -> None:
    print("========================================================")
    print(" MODE A: BLENDED (Safety Brake ON) vs MODE B: PURE RAW (Safety Brake OFF)")
    print(" -- same held-out games, same raw Monte Carlo predictions --")
    print("========================================================")
    print(f" {'Metric':<28}{'Mode B (Raw)':>14}{'Mode A (Blended)':>18}{'Delta':>12}")
    print(f" {'Win/Loss accuracy':<28}{raw_metrics['accuracy']:>13.1f}%{blended_metrics['accuracy']:>17.1f}%"
          f"{blended_metrics['accuracy'] - raw_metrics['accuracy']:>+11.1f}%")
    print(f" {'Average Brier score':<28}{raw_metrics['brier']:>14.4f}{blended_metrics['brier']:>18.4f}"
          f"{blended_metrics['brier'] - raw_metrics['brier']:>+12.4f}")
    print(f" {'Point differential MAE':<28}{raw_metrics['mae']:>13.2f} {blended_metrics['mae']:>17.2f} "
          f"{blended_metrics['mae'] - raw_metrics['mae']:>+11.2f}")
    print("--------------------------------------------------------")
    print(" (accuracy higher is better; Brier score and MAE lower is better)")
    print(" NOTE: MAE is mathematically IDENTICAL between modes by construction -- the Safety")
    print(" Brake (backend/api_simulation.py's compute_dynamic_raw_weight/apply_bayesian_shrinkage)")
    print(" only ever blends the WIN PROBABILITY, never the predicted score/margin, so any nonzero")
    print(" delta printed above is floating-point noise, not a real effect on this metric.")
    weights = [r.shrinkage_weight for r in results_blended if r.shrinkage_weight is not None]
    gaps = [r.delta_net_rating for r in results_blended if r.delta_net_rating is not None]
    if weights:
        print("--------------------------------------------------------")
        print(f" Games with a real NET_RATING match this season : {len(weights)}/{len(results_blended)}")
        print(f" Dynamic raw-sim weight actually used  : min {min(weights):.3f}  "
              f"mean {mean(weights):.3f}  max {max(weights):.3f}")
        print(f" abs(delta NET_RATING) across these games : min {min(gaps):.1f}  "
              f"mean {mean(gaps):.1f}  max {max(gaps):.1f}")
    print("========================================================\n")


def print_holdout_info(meta: dict, csv_path: Path, is_marquee_flags: list[bool]) -> None:
    split_mode = meta["split_mode"]
    info = meta["split_info"]
    report = meta["eval_report"]
    n_marquee = sum(is_marquee_flags)
    print("\n========================================================")
    print(" ML HOLD-OUT SET (fed into the hybrid pass below)")
    print("========================================================")
    print(f" Source dataset               : {csv_path}")
    if split_mode == "time":
        print(f" Split                        : chronological -- {info['n_train']} train / "
              f"{info['n_test']} held-out test games")
        print(f" Train date range             : {info['train_date_range'][0]} to {info['train_date_range'][1]}")
        print(f" Test date range (held out)   : {info['test_date_range'][0]} to {info['test_date_range'][1]}")
    else:
        print(f" Split                        : leave-one-out cross-validation over {info['n_games']} games")
        print(" (no usable game_date column was found for a chronological split)")
    print(f" ML holdout MAE / R^2         : {report['mae']:.2f} pts / {report['r2']:.3f}")
    print(f" ML holdout directional acc.  : {report['directional_accuracy'] * 100:.1f}%")
    print(f" High-leverage matchups       : {n_marquee}/{len(is_marquee_flags)} held-out games "
          f"(real: playoff, top-4-conference-contender matchup, prior-season Finals/Conf-Finals "
          f"rematch, or In-Season Tournament knockout -- see train_ml_model.py's "
          f"compute_high_leverage_flags())")
    print("========================================================")
    print(" Only these held-out games -- never used to train the ML margin")
    print(" model -- are scored below, in both the baseline and hybrid passes,")
    print(" so the two stay directly comparable and neither is leakage-tainted.")
    print(" Both the baseline and hybrid passes below apply the engine's intrinsic,")
    print(" data-calibrated effects (--home-a for team_a, --b2b-a/--b2b-b for any team on 0")
    print(" rest days, and always-on defensive resistance from real def_rating) -- these are")
    print(" part of the autonomous engine now, not hybrid-only additions. The hybrid pass")
    print(f" additionally layers on --ml-margin and --hot-hand-boost {HOT_HAND_BOOST_VALUE} for")
    print(" marquee games, the two remaining *external* overrides.")
    print("========================================================\n")


_PROGRESS_STEP = 25  # for large-N runs with row-printing suppressed
MAX_ROWS_TO_PRINT = 150  # above this game count, per-game rows collapse to periodic progress instead


def print_brake_diagnostic_table(raw_results: list[GameResult], blended_results: list[GameResult],
                                  show_rows: bool = True) -> None:
    """Deeper per-game diagnostic for the Safety Brake comparison: RAW and
    BLENDED predictions side by side, PLUS the exact real point-in-time
    cutoff DATE (D-1, see apply_safety_brake()) actually used for that
    specific game's prior -- printed explicitly, per game, so "true daily
    point-in-time granularity" is directly visible/auditable in the output
    rather than merely asserted. `raw_results`/`blended_results` must be
    the same games in the same order (apply_safety_brake()'s own
    contract).
    """
    print("\n--- Mode B (Raw) vs Mode A (Blended) -- per-game diagnostic ---")
    if not show_rows:
        print(f" ({len(blended_results)} games -- per-game rows suppressed for readability; "
              "pass --verbose for full simulator detail.)")
        return
    header = (f"{'#':>3}  {'Matchup':<10} {'Date':<11} {'Cutoff(D-1)':<11} {'Actual':<13} "
              f"{'Raw':<16} {'Blend':<16} {'Wt':>5} {'|dNetRtg|':>9}  {'R':>1}{'B':>2}")
    print(header)
    print("-" * len(header))
    for i, (r_raw, r_blend) in enumerate(zip(raw_results, blended_results), start=1):
        g = r_raw.game
        matchup = f"{g.team_a}-{g.team_b}"
        actual = f"{g.actual_winner} {g.actual_score_a:.0f}-{g.actual_score_b:.0f}"
        raw_fav = r_raw.predicted_favorite
        raw_str = f"{raw_fav} {r_raw.win_prob[raw_fav] * 100:>5.1f}%"
        blend_fav = r_blend.predicted_favorite
        blend_str = f"{blend_fav} {r_blend.win_prob[blend_fav] * 100:>5.1f}%"
        wt = f"{r_blend.shrinkage_weight:.2f}" if r_blend.shrinkage_weight is not None else "  --"
        gap = f"{r_blend.delta_net_rating:>8.1f}" if r_blend.delta_net_rating is not None else "      --"
        cutoff = r_blend.cutoff_date or "--"
        print(f"{i:>3}  {matchup:<10} {g.game_date:<11} {cutoff:<11} {actual:<13} "
              f"{raw_str:<16} {blend_str:<16} {wt:>5} {gap:>9}  "
              f"{'Y' if r_raw.correct else 'n'} {'Y' if r_blend.correct else 'n'}")


def run_backtest_pass(title: str, games: list[HistoricalGame], exe: Path, args,
                       ml_margins: Optional[list[float]],
                       is_marquee_flags: Optional[list[bool]] = None,
                       show_rows: Optional[bool] = None,
                       ) -> tuple[list[GameResult], Optional[dict]]:
    """Runs one full backtest pass. Home-court (team_a always home, per
    fetch_real_nba_data.py's convention) and back-to-back fatigue (from each
    game's real rest-days columns) are ALWAYS applied -- they're intrinsic,
    data-calibrated engine effects now (calibrated_constants.h), not
    hybrid-only additions. `ml_margins`/`is_marquee_flags`, when given, layer
    the two remaining *external* overrides (--ml-margin/--hot-hand-boost) on
    top, distinguishing the hybrid pass from the pure-engine baseline pass.

    `show_rows` (default: True for <=MAX_ROWS_TO_PRINT games, False above
    that -- a Large-Scale Backtesting concession: printing one line per
    game is fine for a 100-game sample, unreadable noise for a 400+-game
    one) prints periodic running-accuracy progress instead when False,
    both so the terminal stays readable AND so a long-running large-N pass
    still visibly proves it's making progress (each game is one real GPU
    subprocess call, taking a real, non-trivial amount of wall-clock time).
    """
    if show_rows is None:
        show_rows = len(games) <= MAX_ROWS_TO_PRINT

    print(f"\n--- {title} ---")
    if show_rows:
        header = f"{'#':>3}  {'Matchup':<10} {'Actual':<15} {'Fav':<5}{'P(win)':>7}  {'Pred Marg':>10}  {'Actual Marg':>12}  Correct"
        print(header)
        print("-" * len(header))
    else:
        print(f" ({len(games)} games -- per-game rows suppressed for readability; "
              f"pass --verbose for full detail. Running accuracy every {_PROGRESS_STEP} games:)")

    results: list[GameResult] = []
    correct_so_far = 0
    for i, game in enumerate(games, start=1):
        margin = ml_margins[i - 1] if ml_margins is not None else None
        boost = (HOT_HAND_BOOST_VALUE
                 if is_marquee_flags is not None and is_marquee_flags[i - 1]
                 else None)
        is_a_b2b = bool(game.team_a_rest_days == 0)
        is_b_b2b = bool(game.team_b_rest_days == 0)
        try:
            result = evaluate_game(game, exe, args.timeout, args.verbose, margin, boost,
                                    True, is_a_b2b, is_b_b2b)
        except subprocess.TimeoutExpired:
            if show_rows:
                print_game_row(i, game, None, f"timed out after {args.timeout:.0f}s")
            continue
        except BacktestError as e:
            if show_rows:
                print_game_row(i, game, None, str(e))
            continue

        results.append(result)
        correct_so_far += int(result.correct)
        if show_rows:
            print_game_row(i, game, result, None)
        elif i % _PROGRESS_STEP == 0 or i == len(games):
            print(f"   ...{i:>4}/{len(games)} scored, running accuracy {100.0 * correct_so_far / max(1, len(results)):.1f}%")

    print(f" (--home-a applied to every game (team_a); B2B = a team on 0 rest days, "
          f"--b2b-a/--b2b-b applied"
          + (f"; M = marquee matchup, --hot-hand-boost {HOT_HAND_BOOST_VALUE} applied"
             if is_marquee_flags is not None else "") + ")")

    metrics = print_summary(title, games, results)
    return results, metrics


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the C++/GPU NBA Monte Carlo engine against historical game results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                         help=f"Historical games CSV (default: {DEFAULT_CSV.name})")
    parser.add_argument("--exe", type=str, default=None,
                         help="Path to cuda_simulator(.exe); auto-detected under "
                              "cpp_engine/build/{Release,Debug} if omitted")
    parser.add_argument("--mode", choices=["both", "baseline", "hybrid", "brake"], default="both",
                         help="'both' runs a plain Monte Carlo baseline pass and the ML-calibrated "
                              "hybrid pass and compares them (default); 'baseline' or 'hybrid' runs "
                              "only that one pass; 'brake' runs Mode B: Pure Raw Simulation (Safety "
                              "Brake OFF) and Mode A: Blended (Safety Brake ON, backend/api_simulation.py's "
                              "dynamic NET_RATING-gap-driven weight) on the same held-out games and "
                              "compares them -- no ML margin/hot-hand involved in either mode")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL,
                         help=f"Backend player-data endpoint used by the ML step (default: {DEFAULT_API_URL})")
    parser.add_argument("--ml-dir", type=Path, default=DEFAULT_ML_DIR,
                         help=f"Where train_ml_model.py saves/loads its model+margins (default: {DEFAULT_ML_DIR.name}/)")
    parser.add_argument("--split-mode", choices=["auto", "time", "loocv"], default="auto",
                         help="Forwarded to train_ml_model.py's holdout split (--mode baseline ignores this)")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                         help="Forwarded to train_ml_model.py: fraction of games held out chronologically "
                              "(default: 0.2; --split-mode time only)")
    parser.add_argument("--timeout", type=float, default=90.0,
                         help="Per-game subprocess timeout in seconds (default: 90)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap how many games are actually backtested (the first N, "
                              "chronologically, of whichever set is being scored -- all games in "
                              "--mode baseline, only the held-out test games in --mode hybrid/both)")
    parser.add_argument("--verbose", action="store_true",
                         help="Print each game's raw simulator stdout")
    parser.add_argument("--report-json", type=Path, default=None,
                         help="(--mode brake only) Write a full structured report -- dataset info, both "
                              "modes' metrics, statistical significance, and a complete per-game "
                              "breakdown -- to this JSON path (Artifact Preservation)")
    parser.add_argument("--point-in-time-rosters", action="store_true",
                         help="(--mode brake only) Mode B's raw simulation ALSO uses a real, "
                              "point-in-time-only --custom-roster per game (team AND player level, "
                              "built from games played strictly before that game -- see "
                              "evaluate_game_point_in_time()) instead of live team abbreviations "
                              "resolved against whatever season /api/players currently serves. "
                              "Slower (fetches per-game player logs, not just team aggregates) and "
                              "some early-season games get skipped (no real rotation yet).")
    parser.add_argument("--rolling-window-games", type=int, default=None,
                         help="(--point-in-time-rosters only) Restrict each player's point-in-time "
                              "average to their most recent N real games played (trailing window) "
                              "instead of the full season-to-date -- trades sample size for recency. "
                              "Default (omitted): season-to-date, as before.")
    args = parser.parse_args(argv)

    try:
        exe = find_executable(args.exe)
    except BacktestError as e:
        print(f"[backtest_model] Fatal: {e}", file=sys.stderr)
        return 1

    print(f"Backtesting {exe} against games from {args.csv} (mode: {args.mode})...")

    baseline_results: Optional[list[GameResult]] = None
    baseline_metrics: Optional[dict] = None
    hybrid_results: Optional[list[GameResult]] = None
    hybrid_metrics: Optional[dict] = None

    if args.mode == "brake":
        # No ML training involved at all -- the Safety Brake reads each
        # game's own POINT-IN-TIME real NET_RATING/OFF_RATING/DEF_RATING/
        # PACE (see apply_safety_brake()/fetch_point_in_time_efficiency_stats()):
        # that game's own real season, using only games played strictly
        # before it, never today's live season and never a later game's
        # real result. With no ML model to leak from, every game in the
        # CSV is fair game (same "all games, no held-out subset needed"
        # convention as --mode baseline) -- this ALSO skips
        # train_ml_model.py's full ML training pass and its box-score
        # star-availability fetching entirely, since neither is used by
        # either mode here: the efficient path for a Large-Scale backtest,
        # not just a simpler one.
        try:
            games = load_games(args.csv)
        except BacktestError as e:
            print(f"[backtest_model] Fatal: {e}", file=sys.stderr)
            return 1
        if args.limit is not None:
            games = games[: args.limit]
        if not games:
            print(f"[backtest_model] No games to test in {args.csv}.", file=sys.stderr)
            return 1

        print("\n========================================================")
        print(" DATASET (scored in both Safety Brake modes below)")
        print("========================================================")
        print(f" Source dataset               : {args.csv}")
        print(f" Games                        : {len(games)}")
        print(f" Date range                   : {games[0].game_date} to {games[-1].game_date}")
        print(" (no ML training, no held-out split -- the Safety Brake has no leakage concern of")
        print(" its own; it reads each game's own POINT-IN-TIME real NET_RATING/OFF_RATING/")
        print(" DEF_RATING/PACE -- that game's own season, games played strictly before it only,")
        print(" never today's live season or a later game's real result)")
        print(" Both modes below apply the engine's intrinsic, data-calibrated effects (--home-a for")
        print(" team_a, --b2b-a/--b2b-b for any team on 0 rest days, always-on defensive resistance) --")
        print(" no --ml-margin/--hot-hand-boost in either mode, so this isolates the Safety Brake alone.")
        print("========================================================\n")

        if args.point_in_time_rosters:
            title_suffix = (f", rolling {args.rolling_window_games}-game window"
                             if args.rolling_window_games else ", season-to-date")
            raw_results, raw_metrics = run_backtest_pass_point_in_time(
                f"Mode B: Pure Raw Simulation (Safety Brake OFF, POINT-IN-TIME rosters{title_suffix})",
                games, exe, args, rolling_window_games=args.rolling_window_games)
        else:
            raw_results, raw_metrics = run_backtest_pass(
                "Mode B: Pure Raw Simulation (Safety Brake OFF)", games, exe, args, ml_margins=None)
        if not raw_results:
            print("[backtest_model] No games were successfully scored -- cannot apply the Safety Brake.",
                  file=sys.stderr)
            return 1

        print("\nApplying the Market-Prior Safety Brake using each game's own POINT-IN-TIME "
              "real team stats (fetched below, per distinct date needed -- cached, so a large "
              "dataset spanning many games on the same handful of dates costs far fewer nba_api "
              "calls than there are games):")
        try:
            blended_results = apply_safety_brake(raw_results)
        except BacktestError as e:
            print(f"[backtest_model] Fatal: {e}", file=sys.stderr)
            return 1
        show_rows = len(blended_results) <= MAX_ROWS_TO_PRINT
        print_brake_diagnostic_table(raw_results, blended_results, show_rows=show_rows)
        blended_metrics = print_summary("Mode A: Blended (Safety Brake ON)", games, blended_results)

        if raw_metrics and blended_metrics:
            print_brake_comparison(raw_metrics, blended_metrics, blended_results)
            sig = compute_statistical_significance(raw_results, blended_results)
            print_statistical_significance(sig)

            if args.report_json is not None:
                report = build_full_report(args.csv, games, raw_results, blended_results,
                                            raw_metrics, blended_metrics, sig)
                write_json_report(report, args.report_json)

        return 0 if blended_results else 1

    if args.mode == "baseline":
        # No ML component involved, so every game in the CSV is fair game --
        # there's nothing for the model to have leaked from.
        try:
            games = load_games(args.csv)
        except BacktestError as e:
            print(f"[backtest_model] Fatal: {e}", file=sys.stderr)
            return 1
        if args.limit is not None:
            games = games[: args.limit]
        if not games:
            print(f"[backtest_model] No games to test in {args.csv}.", file=sys.stderr)
            return 1

        baseline_results, baseline_metrics = run_backtest_pass(
            "Monte Carlo Baseline (pure calibrated engine, no ML)", games, exe, args, ml_margins=None)
    else:
        # hybrid or both: only the ML model's held-out test games are valid
        # to backtest, so compute those first and let them drive both passes.
        try:
            holdout_games, ml_margins, is_marquee_flags, meta = compute_ml_margins(
                args.csv, args.api_url, args.ml_dir, args.test_fraction, args.split_mode)
        except BacktestError as e:
            print(f"[backtest_model] Fatal: {e}", file=sys.stderr)
            return 1

        if args.limit is not None:
            holdout_games = holdout_games[: args.limit]
            ml_margins = ml_margins[: args.limit]
            is_marquee_flags = is_marquee_flags[: args.limit]

        if not holdout_games:
            print("[backtest_model] No held-out games to backtest.", file=sys.stderr)
            return 1

        print_holdout_info(meta, args.csv, is_marquee_flags)

        if args.mode == "both":
            # No --ml-margin/--hot-hand-boost here -- baseline is the engine's
            # pure, autonomous, data-calibrated output (home-court/fatigue/def
            # resistance still intrinsically applied), so the comparison
            # isolates exactly what the external ML layer adds on top.
            baseline_results, baseline_metrics = run_backtest_pass(
                "Monte Carlo Baseline (pure calibrated engine, held-out games)", holdout_games, exe, args,
                ml_margins=None)

        hybrid_results, hybrid_metrics = run_backtest_pass(
            "Hybrid (calibrated engine + ML margin + Hot Hand, held-out games)", holdout_games, exe, args,
            ml_margins=ml_margins, is_marquee_flags=is_marquee_flags)

    if baseline_metrics and hybrid_metrics:
        print_comparison(baseline_metrics, hybrid_metrics)

    final_results = hybrid_results if hybrid_results is not None else baseline_results
    return 0 if final_results else 1


if __name__ == "__main__":
    # Force line-buffered stdout/stderr even when redirected to a file/pipe
    # (the default on Windows is full block-buffering for a non-tty stream)
    # -- without this, a crash that kills the process abruptly (an external
    # signal, an OOM kill, a Windows environment restart) can lose every
    # buffered print() that hadn't been flushed yet, including a traceback,
    # leaving a log file that looks like the process died silently with no
    # explanation. reconfigure() is a no-op-safe best effort; some exotic
    # stream types don't support it, which is not worth failing startup over.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    try:
        exit_code = main()
    except SystemExit:
        raise
    except BaseException:
        # Catches genuinely everything (including KeyboardInterrupt) so a
        # crash ALWAYS prints its full traceback to the log before this
        # process exits, rather than exiting with a bare, unexplained
        # non-zero code -- see this block's own comment above for why that
        # was otherwise possible even for a real Python exception.
        print("[backtest_model] FATAL: unhandled exception -- full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(1)
    raise SystemExit(exit_code)
