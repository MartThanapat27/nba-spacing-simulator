#!/usr/bin/env python3
"""Trains an XGBoost model to predict a matchup's expected point margin.

This is the "ML" half of the project's hybrid (ML + Monte Carlo) prediction
pipeline: instead of the GPU Monte Carlo engine running "blind" (no notion of
which team is actually better beyond what's baked into its possession model),
this script fits `xgb.XGBRegressor` on roster-derived team-strength features
and predicts mu, the expected Team A - Team B point margin. That margin is
then passed into `cuda_simulator.exe` via `--ml-margin` to bias the GPU
kernel's per-possession shot probabilities toward the data-driven baseline
(see `backtest_model.py` and `cpp_engine/cuda_simulator.cu`).

Data source
-----------
`historical_games.csv` is expected to hold real completed NBA games --
fetch it with `fetch_real_nba_data.py` (nba_api's LeagueGameFinder). Columns:
team_a, team_b, actual_winner, actual_score_a, actual_score_b, game_date,
game_id, team_a_rest_days, team_b_rest_days.

Validation: chronological train/test split
-------------------------------------------
With a real, dated dataset, this script's default and preferred evaluation
is a **chronological holdout**: the earliest `1 - test_fraction` games train
the model, and the most recent `test_fraction` games -- which the model
never sees during training -- are used to score it. `backtest_model.py` then
runs the CUDA hybrid bridge only against this same held-out set. If
`historical_games.csv` has no usable `game_date` column, this falls back to
leave-one-out cross-validation (`--split-mode loocv` forces this explicitly).

No double-counting with the engine's intrinsic effects
---------------------------------------------------------
`cuda_simulator.cu` / `main.cpp` now compute defensive resistance, schedule
fatigue, and home-court advantage **intrinsically**, calibrated by
`calibrate_engine.py` (see `cpp_engine/calibrated_constants.h`) directly from
the same real def_rating/rest-days data this script has access to. Feeding
those same signals into this model's `--ml-margin` output as well would
double-count them: the engine would apply its own calibrated defensive/
fatigue/home-court shift *and* the external model's prediction of the same
real-world effect on top of it. So this model's feature set deliberately
**excludes** team defensive rating, the defense/spacing interaction, rest
advantage, and the home indicator -- those are the engine's job now, not
this model's. What's left is genuinely complementary signal the possession
model can't compute on its own: shooting/usage/spacing profile differences
and real per-game star availability. See `calibrate_engine.py`'s module
docstring for the intrinsic side of this split.

Model capacity vs. dataset size
--------------------------------
A gradient-boosted tree ensemble is a high-capacity model, and a dataset of
a few hundred real games is small for one. The hyperparameters (shallow
trees, a low learning rate, L1/L2 regularization, row/column subsampling, a
`min_child_weight` floor) regularize hard, and are fixed up front rather
than tuned against this same data. Still: read any holdout number from a
small `historical_games.csv` skeptically, and prefer more games
(`fetch_real_nba_data.py --max-games <N>`, or a full season) for a sturdier
read.

Point-in-time roster caveat
----------------------------
Team-strength features are computed from whatever roster snapshot
`/api/players` currently serves, not each historical game's actual
point-in-time roster (trades, injuries, rookies). This affects older games
more than recent ones. The star-availability feature (below) partially
compensates for single-game absences, but the *season-level* rates
(shooting %, usage) themselves are still today's, not that game's.

Feature set (9 model inputs, 10 computed)
-------------------------------------------
Base per-team features (Team A - Team B differential), computed from
`/api/players`'s top-9-by-minutes rotation -- exactly like
`build_gpu_roster()` in `cuda_main.cpp`:

  * `starter_gravity`     -- Box-Cox floor-spacing gravity, positionally weighted,
                              summed over the top-5 minutes leaders ("3PT Spacing Index").
  * `bench_gravity`       -- the same, over rotation spots 6-9 (the second unit).
  * `rotation_fg3_pct`    -- minutes-weighted mean 3PT% over the top-9 rotation.
  * `rotation_fg3a`       -- minutes-weighted mean 3PT attempts over the top-9 rotation.
  * `top_player_usage`    -- highest usage_rate in the top-9 rotation ("star power").
  * `usage_concentration` -- std-dev of usage_rate across the top-9 rotation.
  * `net_rating_proxy`    -- rotation_fg3_pct * rotation_fg3a * 3, a volume-times-
                              efficiency offensive proxy (no true net rating source exists).

Pairwise/game-level features (not simple differentials -- need both teams,
or this specific game, to compute):

  * `big_match_indicator`     -- min(team_a's net_rating_proxy, team_b's): high only
                                  when both teams are good (a "clash of contenders" signal).
                                  DIAGNOSTIC ONLY -- see MODEL_FEATURE_NAMES below. This value
                                  is symmetric under a team_a/team_b swap (min() doesn't care
                                  about order), unlike every other feature here, so it is NOT
                                  fed to XGBoost as an independent input -- doing so would break
                                  the model's antisymmetry (predict(A,B) should equal
                                  -predict(B,A), since actual_margin itself flips sign under a
                                  swap). It's still computed for every row (kept as the
                                  feat_big_match_indicator diagnostic column, used by
                                  analyze_high_leverage_variance() and exported to
                                  holdout_predictions.json) and still used as the multiplier
                                  inside star_usage_concentration below, which -- as a product of
                                  an antisymmetric term (star_usage_diff) and this symmetric one
                                  -- is itself correctly antisymmetric and IS a real model input.
  * `star_usage_concentration` -- (team_a's top_player_usage - team_b's) * big_match_indicator:
                                  star-usage gap, amplified in marquee games.
  * `star_out_diff`           -- int(team_b's star out) - int(team_a's star out); REAL,
                                  derived per game from that game's actual nba_api box score
                                  (see check_star_availability()) -- whether each team's
                                  current top-usage player has a recorded minutes entry for
                                  that specific game. When a team's star is flagged out for a
                                  given game, that team's starter_gravity/bench_gravity/
                                  top_player_usage/usage_concentration/net_rating_proxy for
                                  THAT ROW are recomputed with the star excluded from the
                                  rotation (see team_features_excluding()) -- the "dynamic
                                  downgrade" -- rather than using their full-strength season
                                  aggregate. 0 for ad hoc --predict calls (assumes full health).

Team defensive rating (`def_rating`) and each game's rest days are still
fetched and attached to every row (as plain `net_def_edge`/`rest_advantage`/
`team_a_rest_days`/`team_b_rest_days` columns, not `feat_*` columns) since
`calibrate_engine.py` needs them for its own regression -- they're simply no
longer part of *this* model's input vector, per the double-counting note
above.

High-leverage game classification -- diagnostic, not a fabricated weight
--------------------------------------------------------------------------
`historical_games.csv` is 100% 2024-25 REGULAR SEASON games (confirmed via
the NBA's own game_id season-type prefix: every row is "002...", none are
"004..." = playoffs) with no season-type column at all -- there is no
playoff/Finals data IN THIS FILE to flag. Real playoff games are fetchable
(`fetch_real_nba_data.py --season-type Playoffs`) and are used for a
SEPARATE, never-trained-on evaluation subset (see backtest_model.py /
evaluate_playoff_holdout.py) rather than folded into the training CSV
itself, since every playoff game postdates every regular-season game
already in this file and would otherwise inflate/shift the standard
chronological holdout's size and composition.

`compute_high_leverage_flags()` classifies each game via real programmatic
signals: `classify_game_id()` parses season type (and playoff round/
Finals) straight from the NBA's own game_id convention, `fetch_conference_
top_n()` pulls live end-of-season conference standings to flag genuine
top-4 contenders, and `fetch_deep_playoff_rematch_pairs()` pulls the
immediately preceding season's real Conference Finals/NBA Finals
participants to flag rematches -- plus a small fixed-date table for the
In-Season Tournament's knockout round.

Every training run also runs `analyze_high_leverage_variance()`: a Welch's
t-test (Bonferroni-corrected) comparing each of the 10 model features, plus
|actual_margin| and its variance (Levene's test), between high-leverage and
ordinary games. On this dataset (1225 games, 80 flagged) the result is
unambiguous: 9/10 features show no significant difference (p=0.75-0.99),
the one exception is circular by construction (top-4 standing correlates
with the roster-strength feature that IS partly derived from team
quality), and neither |margin| (p=0.77) nor its variance (p=0.89) differ
either. There is therefore NO statistical basis for treating high-leverage
games differently during training -- not as a new predictive feature (the
distributions aren't different) and not as a loss-weighting multiplier
(the outcomes aren't structurally different either).
`DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER = 1.0` (a genuine no-op, uniform
weighting) reflects this finding -- it replaces an earlier "2.0" value
that was picked without this check and empirically made the full hybrid
pipeline WORSE (see ml_model/hybrid_pipeline_benchmark.json's
prior_configurations). `build_high_leverage_sample_weight()` still exists
and still works mechanically if a *different* dataset's own
analyze_high_leverage_variance() run finds real evidence -- it's just not
applied by fabricated default.

Post-hoc win-probability calibration
--------------------------------------
`fit_win_probability_calibrator()` replaces the previous ad hoc
normal_cdf(margin / kRealMarginStdDev) shortcut (an ASSUMED Gaussian shape
around a fixed, not-independently-fit std dev) with a mapping genuinely fit
against real held-out win/loss outcomes: Platt scaling (logistic
regression) and Isotonic Regression are both fit on a chronological
calibration-fit slice of the holdout and compared by Brier score on a
genuinely held-out calibration-eval slice, with the better one deployed
(see the function's docstring, and ml_model/calibration_report.json for
the actual comparison numbers from the last training run).

Usage:
    python train_ml_model.py                       # train, holdout-evaluate, save artifacts
    python train_ml_model.py --season 2024-25       # season used for nba_api def-rating lookup
    python train_ml_model.py --test-fraction 0.25   # bigger holdout
    python train_ml_model.py --split-mode loocv     # force leave-one-out instead
    python train_ml_model.py --high-leverage-weight-multiplier 2.0  # override the evidence-based 1.0 default
    python train_ml_model.py --predict MIN OKC      # load saved model, print margin + calibrated win probability
    python train_ml_model.py --csv my_games.csv --out-dir ml_model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "historical_games.csv"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/players"
DEFAULT_OUT_DIR = SCRIPT_DIR / "ml_model"
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SEASON = "2024-25"
MIN_ROWS_FOR_TIME_SPLIT = 10  # below this, a chronological holdout is too small to trust
# DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER is defined further down, next to
# build_high_leverage_sample_weight() -- see its docstring for the
# statistical analysis (analyze_high_leverage_variance()) that justifies
# its value of 1.0 (no arbitrary weighting; a clean, evidence-based
# baseline, not a fabricated context multiplier).
BOX_SCORE_REQUEST_PAUSE_SECONDS = 0.6

MAX_ROTATION_PLAYERS = 9   # matches build_gpu_roster()'s rotation cap in cuda_main.cpp
MAX_ON_COURT_PLAYERS = 5   # matches kMaxOnCourtPlayers in cuda_simulator.cu

BASE_FEATURE_NAMES = [
    "starter_gravity",
    "bench_gravity",
    "rotation_fg3_pct",
    "rotation_fg3a",
    "top_player_usage",
    "usage_concentration",
    "net_rating_proxy",
]

# Pairwise/game-level features -- need both teams' (or this specific game's)
# data to compute, so they're built directly in build_feature_vector() below
# rather than via TeamFeatures.vector() subtraction. See module docstring.
# NOTE: def_matchup_edge, rest_advantage, and home_indicator used to live
# here too; they were removed because the C++ engine now applies those exact
# real-world effects intrinsically (see calibrate_engine.py) -- keeping them
# here as well would double-count them in --ml-margin.
PAIRWISE_FEATURE_NAMES = [
    "big_match_indicator",
    "star_usage_concentration",
    "star_out_diff",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + PAIRWISE_FEATURE_NAMES

# The actual XGBoost input set: FEATURE_NAMES minus `big_match_indicator`.
# big_match_indicator = min(team_a, team_b) is symmetric under a team-A/B
# swap (min() doesn't care about order), unlike every other feature here --
# feeding it to the model as its own raw predictor breaks the model's
# antisymmetry (predict(A,B) should equal -predict(B,A), since the target,
# actual_margin, flips sign under a swap). It's still computed and kept as
# a diagnostic-only training-table column (feat_big_match_indicator, see
# FEATURE_NAMES's other uses below) and as the multiplier inside the
# properly-antisymmetric star_usage_concentration interaction term -- just
# never fed to the model as an independent input.
MODEL_FEATURE_NAMES = [name for name in FEATURE_NAMES if name != "big_match_indicator"]

# Fixed up front rather than tuned against this same data (see the module
# docstring's honesty note). Chosen to regularize as hard as reasonably
# possible for a small tabular sports dataset:
#   - max_depth=3, learning_rate=0.05, n_estimators=100: shallow, slow-learning trees.
#   - reg_alpha / reg_lambda: L1/L2 regularization on leaf weights.
#   - subsample / colsample_bytree: each tree only sees 80% of rows/features.
#   - min_child_weight=3: refuses to split down to near-single-sample leaves.
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    objective="reg:squarederror",
    importance_type="gain",
    random_state=42,
    n_jobs=1,
    verbosity=0,
)


class MlPipelineError(Exception):
    pass


def get_position_weight(position: str) -> float:
    """Mirrors get_position_weight() in cpp_engine/cuda_main.cpp."""
    if position in ("C", "C-F", "F-C"):
        return 1.5
    if position in ("PF", "F"):
        return 1.2
    return 1.0


def box_cox_spacing(fg3_pct: float, fg3a: float) -> float:
    """Mirrors calculate_spacing() in cpp_engine/cuda_main.cpp."""
    return fg3_pct * (fg3a + 1.0) ** 0.2671


@dataclass
class TeamFeatures:
    team_abbreviation: str
    starter_gravity: float
    bench_gravity: float
    rotation_fg3_pct: float
    rotation_fg3a: float
    top_player_usage: float
    usage_concentration: float
    net_rating_proxy: float
    def_rating: float

    def vector(self) -> np.ndarray:
        """The 7 base features actually fed to XGBoost. Deliberately excludes
        def_rating (still a field on this dataclass, for calibrate_engine.py's
        use) -- see the module docstring's double-counting note.
        """
        return np.array([
            self.starter_gravity,
            self.bench_gravity,
            self.rotation_fg3_pct,
            self.rotation_fg3a,
            self.top_player_usage,
            self.usage_concentration,
            self.net_rating_proxy,
        ], dtype=float)


def build_feature_vector(team_a: TeamFeatures, team_b: TeamFeatures, star_out_diff: float = 0.0) -> np.ndarray:
    """The full 10-feature model input for one team_a vs team_b matchup. Used
    identically by load_training_table() (one row per historical game, with
    per-game star-availability context) and predict_margin() (one ad hoc
    matchup, which has no specific game so star_out_diff defaults to
    neutral) -- so the training and inference feature pipelines can never
    drift apart. Does NOT take defensive rating, rest days, or a home
    indicator -- see the module docstring's double-counting note; those real
    effects are now the C++ engine's job (calibrate_engine.py), not this
    model's.
    """
    base_diff = team_a.vector() - team_b.vector()

    big_match_indicator = min(team_a.net_rating_proxy, team_b.net_rating_proxy)
    star_usage_diff = team_a.top_player_usage - team_b.top_player_usage
    star_usage_concentration = star_usage_diff * big_match_indicator

    pairwise = [big_match_indicator, star_usage_concentration, star_out_diff]
    return np.concatenate([base_diff, pairwise])


# Index of MODEL_FEATURE_NAMES's columns within FEATURE_NAMES's full order --
# i.e. how to slice a build_feature_vector() result (or any FEATURE_NAMES-
# ordered row) down to just the model's actual inputs. Computed once at
# import time since both name lists are static.
_MODEL_FEATURE_IDX = [FEATURE_NAMES.index(name) for name in MODEL_FEATURE_NAMES]


def to_model_features(feat_vec: np.ndarray) -> np.ndarray:
    """Drops big_match_indicator (diagnostic-only) from a build_feature_vector()
    result, returning just the MODEL_FEATURE_NAMES-ordered inputs actually fed
    to XGBoost. Every direct model.predict() call site must route through this
    (or the equivalent MODEL_FEATURE_NAMES column selection used at training
    time) rather than passing build_feature_vector()'s raw output straight to
    the model -- see MODEL_FEATURE_NAMES's docstring for why.
    """
    return feat_vec[..., _MODEL_FEATURE_IDX]


def fetch_players(api_url: str) -> list[dict]:
    try:
        resp = requests.get(api_url, timeout=15)
    except requests.RequestException as e:
        raise MlPipelineError(
            f"Could not reach the backend API at {api_url} -- is it running? ({e})"
        ) from e
    if resp.status_code != 200:
        raise MlPipelineError(f"API returned status {resp.status_code} from {api_url}")

    data = resp.json()
    players = data if isinstance(data, list) else data.get("players", [])
    if not players:
        raise MlPipelineError(f"No player records returned from {api_url}")
    return players


def fetch_team_defensive_ratings(season: str) -> dict[str, float]:
    """REAL data: each team's season-to-date defensive rating (points allowed
    per 100 possessions; lower = better defense) via nba_api's
    leaguedashteamstats (Advanced). No opponent-abbreviation column comes
    back from this endpoint, so team_id is mapped to abbreviation via
    nba_api's static team list (same pattern backend/fetch_nba_data.py uses).
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    try:
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        df = resp.get_data_frames()[0]
    except Exception as e:
        raise MlPipelineError(f"Could not fetch team defensive ratings from nba_api: {e}") from e

    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
    return dict(zip(df["TEAM_ABBREVIATION"], df["DEF_RATING"].astype(float)))


# --- True game classification & real high-leverage signals ----------------
# (game_id season-type/round parsing, prior-season deep-playoff rematches,
# and real conference-standings-based "contender" matchups -- replaces the
# roster-strength-median proxy previously used for is_marquee/sample
# weighting; see build_high_leverage_sample_weight() below for where this
# actually gets used, and its docstring for why this stays a TRAINING
# SAMPLE WEIGHT rather than a new predictive feature.)

def classify_game_id(game_id) -> dict:
    """Parses a standard 10-digit NBA game_id's season type and (for a
    playoff game) round/Finals status.

    EMPIRICALLY VERIFIED against real nba_api data (not assumed from
    documentation) by fetching the actual 2024-25 and 2023-24 playoff game
    logs and inspecting their game_ids directly:
      - Character index 2 (the 3rd digit) is the season-type digit --
        '1' = preseason, '2' = regular season, '4' = playoffs (matches
        every one of the 1225 games in historical_games.csv, all '002...').
      - For a playoff game, character index 7 is the ROUND (1 = First
        Round, 2 = Conference Semifinals, 3 = Conference Finals, 4 = NBA
        Finals) and index 8 is the series index within that round -- e.g.
        "0042400407" decodes to round=4 (Finals), confirmed to be the real
        2025 NBA Finals Game 7 (OKC vs IND, played 2025-06-22).
    Returns {"season_type": ..., "round": Optional[int], "is_finals": bool,
    "is_playoff": bool} -- "unknown" season_type (with round=None) for
    anything that doesn't match this 10-digit numeric shape, so a malformed
    or unusual ID degrades gracefully instead of raising.
    """
    game_id = str(game_id).strip()
    if len(game_id) != 10 or not game_id.isdigit():
        return {"season_type": "unknown", "round": None, "is_finals": False, "is_playoff": False}

    season_type = {"1": "preseason", "2": "regular_season", "4": "playoffs"}.get(game_id[2], "unknown")
    is_playoff = season_type == "playoffs"
    round_num = int(game_id[7]) if is_playoff else None
    is_finals = is_playoff and round_num == 4
    return {"season_type": season_type, "round": round_num, "is_finals": is_finals, "is_playoff": is_playoff}


def _prior_season_string(season: str) -> str:
    """'2024-25' -> '2023-24'."""
    start_year = int(season[:4])
    prev_start = start_year - 1
    return f"{prev_start}-{str(prev_start + 1)[-2:]}"


def fetch_deep_playoff_rematch_pairs(season: str) -> set[frozenset]:
    """REAL team pairs that met in the PRIOR season's Conference Finals or
    NBA Finals (round >= 3 via classify_game_id), fetched live from
    nba_api's leaguegamefinder for `season`'s immediately preceding season
    -- e.g. for season="2024-25" this looks at 2023-24's playoffs, which
    (verified directly against real nba_api data) were BOS-IND and MIN-DAL
    in the Conference Finals and BOS-DAL in the NBA Finals.

    A rematch of one of these pairs this season is a genuine, real
    high-stakes signal, distinct from and complementary to the roster-
    strength-based `big_match_indicator` feature already in FEATURE_NAMES.
    Returns an empty set (not an error) on any fetch failure or if the
    prior season has no playoff data -- a missing rematch signal degrades
    to "no rematch flagged", not a training failure.
    """
    from nba_api.stats.endpoints import leaguegamefinder

    prior_season = _prior_season_string(season)
    try:
        resp = leaguegamefinder.LeagueGameFinder(
            season_nullable=prior_season, season_type_nullable="Playoffs", timeout=20)
        df = resp.get_data_frames()[0]
    except Exception:
        return set()
    if df.empty:
        return set()

    df = df.assign(_round=df["GAME_ID"].map(lambda gid: classify_game_id(gid)["round"]))
    deep = df[df["_round"].fillna(0) >= 3]

    pairs: set[frozenset] = set()
    for game_id, group in deep.groupby("GAME_ID"):
        teams = set(group["TEAM_ABBREVIATION"].unique())
        if len(teams) == 2:
            pairs.add(frozenset(teams))
    return pairs


def fetch_conference_top_n(season: str, n: int = 4) -> set[str]:
    """REAL teams ranked top-`n` in EITHER conference by end-of-season
    standings (nba_api's leaguestandingsv3's PlayoffRank column) -- a
    genuine "contender" signal, replacing the roster-strength-only
    big_match_indicator median-split proxy previously used for
    is_marquee/sample weighting.

    Uses END-OF-SEASON standings, not each game's actual point-in-time
    standing -- the same simplification already documented throughout this
    module for team_features generally (see the module docstring's
    point-in-time caveat); a genuinely point-in-time version would need a
    per-game standings snapshot, which this project's own point-in-time
    backtesting work (see backtest_model.py) already found trades away more
    accuracy than it buys back, given this dataset's size.
    """
    from nba_api.stats.endpoints import leaguestandingsv3
    from nba_api.stats.static import teams as static_teams

    resp = leaguestandingsv3.LeagueStandingsV3(season=season, timeout=20)
    df = resp.get_data_frames()[0]
    top = df[df["PlayoffRank"] <= n]
    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    return set(top["TeamID"].map(id_to_abbr))


# The NBA In-Season Tournament (Emirates NBA Cup)'s knockout round --
# semifinals and championship, both played in Las Vegas -- run on fixed,
# publicly announced dates each season and are NOT distinguishable from a
# normal regular-season game_id in any documented way (confirmed: their
# game_ids are still "002..." like every other regular-season game, just
# with out-of-sequence game numbers), so unlike classify_game_id() above,
# this is a plain date lookup table, not a general decoder. Update this
# each season the tournament runs.
TOURNAMENT_KNOCKOUT_DATES = {
    "2024-25": {"2024-12-14", "2024-12-17"},  # semifinals, championship
}


def is_tournament_knockout_game(game_date: str, season: str) -> bool:
    return game_date in TOURNAMENT_KNOCKOUT_DATES.get(season, set())


def compute_high_leverage_flags(game_id, team_a: str, team_b: str, game_date: str, season: str,
                                 top4_teams: set[str], rematch_pairs: set[frozenset]) -> dict:
    """Combines classify_game_id() + the two real contextual signals above
    into one row's worth of high-leverage classification. `is_high_leverage`
    is True if ANY of: this is a real playoff game, both teams are current
    top-`n` conference contenders, this matchup is a rematch of the prior
    season's Conference Finals/NBA Finals, or this is an In-Season
    Tournament knockout-round game -- deliberately an OR across several
    independent real signals rather than one rule, since "high-stakes"
    genuinely has more than one real-world cause.
    """
    classification = classify_game_id(game_id)
    is_top4_matchup = team_a in top4_teams and team_b in top4_teams
    is_rematch = frozenset({team_a, team_b}) in rematch_pairs
    is_tourney = is_tournament_knockout_game(game_date, season)
    is_high_leverage = classification["is_playoff"] or is_top4_matchup or is_rematch or is_tourney
    return {
        "season_type": classification["season_type"],
        "playoff_round": classification["round"],
        "is_finals": classification["is_finals"],
        "is_playoff": classification["is_playoff"],
        "is_top4_matchup": is_top4_matchup,
        "is_deep_playoff_rematch": is_rematch,
        "is_tournament_knockout": is_tourney,
        "is_high_leverage": is_high_leverage,
    }


def _reduce_rotation(rotation: pd.DataFrame) -> dict:
    """Reduces one team's (already row-selected, minutes-sorted) rotation
    into the 7 rotation-derived base feature values shared by
    compute_team_features() and team_features_excluding().
    """
    starters = rotation.head(MAX_ON_COURT_PLAYERS)
    bench = rotation.iloc[MAX_ON_COURT_PLAYERS:]

    def spacing_gravity(sub) -> float:
        return float(sum(
            box_cox_spacing(row.fg3_pct, row.fg3a) * get_position_weight(row.position)
            for row in sub.itertuples()
        ))

    starter_gravity = spacing_gravity(starters)
    bench_gravity = spacing_gravity(bench) if not bench.empty else 0.0

    rotation_weights = np.maximum(rotation["min"], 1.0)
    rotation_fg3_pct = float(np.average(rotation["fg3_pct"], weights=rotation_weights))
    rotation_fg3a = float(np.average(rotation["fg3a"], weights=rotation_weights))
    top_player_usage = float(rotation["usage_rate"].max())
    usage_concentration = float(rotation["usage_rate"].std(ddof=0)) if len(rotation) > 1 else 0.0
    net_rating_proxy = rotation_fg3_pct * rotation_fg3a * 3.0

    return dict(starter_gravity=starter_gravity, bench_gravity=bench_gravity,
                rotation_fg3_pct=rotation_fg3_pct, rotation_fg3a=rotation_fg3a,
                top_player_usage=top_player_usage, usage_concentration=usage_concentration,
                net_rating_proxy=net_rating_proxy)


def _prep_players_df(players: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(players)
    df["team_abbreviation"] = df.get("team_abbreviation", "FA").fillna("FA")
    df["player_name"] = df.get("player_name", "").fillna("")
    df["position"] = df.get("position", "SG").fillna("SG")
    df["min"] = pd.to_numeric(df.get("min", 0.0), errors="coerce").fillna(0.0)
    df["usage_rate"] = pd.to_numeric(df.get("usage_rate", 20.0), errors="coerce").fillna(20.0)
    df["fg3a"] = pd.to_numeric(df.get("fg3a", 0.0), errors="coerce").fillna(0.0)
    df["fg3_pct"] = pd.to_numeric(df.get("fg3_pct", 0.0), errors="coerce").fillna(0.0)
    return df


def compute_team_features(players_df: pd.DataFrame, def_ratings: dict[str, float]
                           ) -> tuple[dict[str, TeamFeatures], dict[str, str]]:
    """Groups players by team, applies the same top-9-by-minutes rotation cap
    the C++ engines use, and reduces each rotation to the 8 base features.
    Also returns each team's "star" name (highest usage_rate in their
    rotation) for the star-availability check.
    """
    features: dict[str, TeamFeatures] = {}
    star_names: dict[str, str] = {}

    for team_abbr, group in players_df.groupby("team_abbreviation"):
        rotation = group.sort_values("min", ascending=False).head(MAX_ROTATION_PLAYERS)
        if rotation.empty:
            continue
        reduced = _reduce_rotation(rotation)
        def_rating = def_ratings.get(team_abbr, float(np.mean(list(def_ratings.values()))) if def_ratings else 113.0)

        features[team_abbr] = TeamFeatures(team_abbreviation=team_abbr, def_rating=def_rating, **reduced)
        star_names[team_abbr] = str(rotation.loc[rotation["usage_rate"].idxmax(), "player_name"])

    return features, star_names


def team_features_excluding(players_df: pd.DataFrame, team_abbr: str, def_ratings: dict[str, float],
                             excluded_name: str) -> TeamFeatures:
    """Same reduction as compute_team_features(), but for one team, with one
    named player (a flagged-out star) removed from the pool before picking
    the top-9-by-minutes rotation -- the "dynamic downgrade" mechanism: the
    next-best player is promoted into the rotation instead, and
    starter_gravity/top_player_usage/etc. drop accordingly, for this one
    game's row only.
    """
    norm_excluded = excluded_name.strip().lower()
    group = players_df[players_df["team_abbreviation"] == team_abbr]
    group = group[group["player_name"].str.strip().str.lower() != norm_excluded]

    rotation = group.sort_values("min", ascending=False).head(MAX_ROTATION_PLAYERS)
    reduced = _reduce_rotation(rotation)
    def_rating = def_ratings.get(team_abbr, 113.0)
    return TeamFeatures(team_abbreviation=team_abbr, def_rating=def_rating, **reduced)


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch.isspace())


def check_star_availability(game_id: str, team_a: str, team_b: str,
                             star_name_a: str, star_name_b: str, cache: dict) -> tuple[bool, bool]:
    """REAL, per-game data: did each team's (current-roster) star actually
    play in this specific historical game? Checked against that game's real
    nba_api box score (boxscoretraditionalv3) -- a player absent from the
    box score, or present with no recorded minutes, is flagged "out" for
    that game. Results are cached to disk (see train()) since this is one
    network call per game and doesn't change once a game is final.

    Name matching against the box score's firstName/familyName is normalized
    (case/punctuation-insensitive) but still exact-string -- an occasional
    mismatch (e.g. a suffix formatted differently) is possible and would
    read as "star out" when they actually played; this is a known limitation
    given no player-ID cross-reference is available here.
    """
    if game_id in cache:
        entry = cache[game_id]
        return bool(entry["team_a_star_out"]), bool(entry["team_b_star_out"])

    from nba_api.stats.endpoints import boxscoretraditionalv3

    try:
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=20)
        df = box.get_data_frames()[0]
    except Exception:
        # Network/API hiccup: fail safe to "both played" (no effect on this
        # row) rather than aborting the whole training run.
        cache[game_id] = {"team_a_star_out": False, "team_b_star_out": False}
        return False, False
    finally:
        time.sleep(BOX_SCORE_REQUEST_PAUSE_SECONDS)

    df["full_name_norm"] = (df["firstName"].fillna("") + " " + df["familyName"].fillna("")).map(_normalize_name)

    def played(team_abbr: str, star_name: str) -> bool:
        target = _normalize_name(star_name)
        sub = df[(df["teamTricode"] == team_abbr) & (df["full_name_norm"] == target)]
        if sub.empty:
            return False
        minutes = sub.iloc[0]["minutes"]
        return isinstance(minutes, str) and minutes not in ("", "0:00")

    a_played = played(team_a, star_name_a)
    b_played = played(team_b, star_name_b)
    result = (not a_played, not b_played)
    cache[game_id] = {"team_a_star_out": result[0], "team_b_star_out": result[1]}
    return result


def load_training_table(csv_path: Path, players_df: pd.DataFrame,
                         team_features: dict[str, TeamFeatures], star_names: dict[str, str],
                         def_ratings: dict[str, float], star_cache: dict, verbose: bool = True,
                         season: str = DEFAULT_SEASON,
                         top4_teams: Optional[set[str]] = None,
                         rematch_pairs: Optional[set[frozenset]] = None,
                         ) -> pd.DataFrame:
    if not csv_path.is_file():
        raise MlPipelineError(f"Historical games CSV not found: {csv_path}")

    # game_id (e.g. "0022401177") must be read as a string -- pandas'
    # automatic dtype inference otherwise treats it as an integer and
    # silently strips the leading zero, corrupting every box-score lookup
    # in check_star_availability() (every ID becomes invalid, every fetch
    # comes back empty, and every game gets misread as "both stars out").
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    dtype_overrides = {"game_id": str} if "game_id" in header_cols else {}
    games = pd.read_csv(csv_path, dtype=dtype_overrides)
    required = {"team_a", "team_b", "actual_winner", "actual_score_a", "actual_score_b"}
    missing = required - set(games.columns)
    if missing:
        raise MlPipelineError(f"CSV is missing required column(s): {sorted(missing)}")

    if "game_date" not in games.columns:
        games["game_date"] = ""
    games["game_date_parsed"] = pd.to_datetime(games["game_date"], errors="coerce")

    if "game_id" not in games.columns:
        games["game_id"] = ""
    for col in ("team_a_rest_days", "team_b_rest_days"):
        if col not in games.columns:
            games[col] = 0.0

    games["team_a"] = games["team_a"].str.strip().str.upper()
    games["team_b"] = games["team_b"].str.strip().str.upper()
    games["actual_winner"] = games["actual_winner"].str.strip().str.upper()

    rows = []
    n = len(games)
    for i, g in games.iterrows():
        if g.team_a not in team_features or g.team_b not in team_features:
            missing_team = g.team_a if g.team_a not in team_features else g.team_b
            raise MlPipelineError(
                f"Row {i}: team '{missing_team}' has no roster data from the API "
                f"(fetched teams: {sorted(team_features)})"
            )

        rest_advantage = float(g.get("team_a_rest_days", 0.0)) - float(g.get("team_b_rest_days", 0.0))

        fa, fb = team_features[g.team_a], team_features[g.team_b]
        # Defensive rating is a static team-level stat, unaffected by which
        # players are excluded below, so it's read before any star-out
        # adjustment to fa/fb. Sign matches calibrate_engine.py's net_def_edge
        # exactly: positive => team_b defends worse than team_a.
        net_def_edge = fb.def_rating - fa.def_rating

        star_out_diff = 0.0
        game_id = str(g.get("game_id", "")).strip()
        if game_id and game_id.lower() != "nan":
            a_out, b_out = check_star_availability(
                game_id, g.team_a, g.team_b,
                star_names.get(g.team_a, ""), star_names.get(g.team_b, ""), star_cache)
            star_out_diff = float(b_out) - float(a_out)
            if a_out:
                fa = team_features_excluding(players_df, g.team_a, def_ratings, star_names.get(g.team_a, ""))
            if b_out:
                fb = team_features_excluding(players_df, g.team_b, def_ratings, star_names.get(g.team_b, ""))
            if verbose and (i + 1) % 20 == 0:
                print(f"  ...checked star availability for {i + 1}/{n} games")

        feat_vec = build_feature_vector(fa, fb, star_out_diff)
        actual_margin = float(g.actual_score_a) - float(g.actual_score_b)

        # True game classification + real high-leverage signals -- see
        # compute_high_leverage_flags()'s docstring. NOT fed to XGBoost
        # (see build_high_leverage_sample_weight()'s docstring): kept as
        # plain columns for training-time sample weighting and reporting.
        leverage = compute_high_leverage_flags(
            game_id, g.team_a, g.team_b, str(g.game_date), season,
            top4_teams or set(), rematch_pairs or set(),
        )

        row = {f"feat_{name}": val for name, val in zip(FEATURE_NAMES, feat_vec)}
        row.update({
            "team_a": g.team_a,
            "team_b": g.team_b,
            "actual_winner": g.actual_winner,
            "actual_score_a": float(g.actual_score_a),
            "actual_score_b": float(g.actual_score_b),
            "actual_margin": actual_margin,
            "game_date": g.game_date,
            "game_date_parsed": g.game_date_parsed,
            "team_a_rest_days": float(g.get("team_a_rest_days", 0.0)),
            "team_b_rest_days": float(g.get("team_b_rest_days", 0.0)),
            # Not fed to XGBoost (see module docstring) -- kept as plain
            # columns purely for calibrate_engine.py's regression.
            "rest_advantage": rest_advantage,
            "net_def_edge": net_def_edge,
            **leverage,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def choose_split_mode(table: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    has_dates = table["game_date_parsed"].notna().all()
    if has_dates and len(table) >= MIN_ROWS_FOR_TIME_SPLIT:
        return "time"
    return "loocv"


def time_based_split(table: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological holdout: the earliest games train, the most recent
    `test_fraction` are held out to evaluate -- see the module docstring.
    """
    ordered = table.sort_values("game_date_parsed", kind="stable").reset_index(drop=True)
    n_test = max(1, round(len(ordered) * test_fraction))
    n_test = min(n_test, len(ordered) - 1)  # always leave at least 1 training row
    train_table = ordered.iloc[: len(ordered) - n_test].reset_index(drop=True)
    test_table = ordered.iloc[len(ordered) - n_test:].reset_index(drop=True)
    return train_table, test_table


def _regression_report(preds: np.ndarray, y: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - y)))
    directional_hits = int(np.sum(np.sign(preds) == np.sign(y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "r2": r2, "directional_accuracy": directional_hits / len(y)}


def evaluate_time_holdout(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           sample_weight_train: Optional[np.ndarray] = None):
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight_train)
    preds = model.predict(X_test)

    report = _regression_report(preds, y_test)
    report["n_train"] = len(y_train)
    report["n_test"] = len(y_test)
    return preds, report


def evaluate_loocv(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    """Leave-one-out cross-validation: each game's prediction comes from a
    model retrained without it. Used when there's no usable game_date column
    to build a chronological holdout from.
    """
    from sklearn.model_selection import LeaveOneOut
    from xgboost import XGBRegressor

    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        model = XGBRegressor(**XGB_PARAMS)
        fold_weight = sample_weight[train_idx] if sample_weight is not None else None
        model.fit(X[train_idx], y[train_idx], sample_weight=fold_weight)
        preds[test_idx] = model.predict(X[test_idx])

    report = _regression_report(preds, y)
    report["n_train"] = len(y) - 1
    report["n_test"] = len(y)
    return preds, report


def fit_final_model(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def analyze_high_leverage_variance(table: pd.DataFrame) -> dict:
    """Statistical check for whether high-leverage games (see
    compute_high_leverage_flags()) actually differ from ordinary games in
    any way that would justify treating them differently during training --
    EITHER a structural feature split (if their underlying feature
    distributions genuinely differ) OR a loss-weighting scheme (if their
    outcomes are structurally harder/easier to predict) -- rather than
    assuming either and picking an arbitrary multiplier.

    Runs a Welch's t-test (unequal-variance, appropriate given the very
    unequal group sizes here) on each of the 10 model features, plus a
    check on |actual_margin| (are high-leverage games closer/more
    lopsided?) and Levene's test on actual_margin's variance (are outcomes
    structurally more/less variable, i.e. harder/easier to predict?).
    Bonferroni-corrects across the 10 feature tests (alpha/10) since
    running 10 independent tests at a naive alpha=0.05 would flag ~0.5
    false positives by chance alone.

    Returns a dict with the full per-feature test results plus a summary
    verdict -- see train()'s use of this for exactly how the verdict
    determines the sample-weight multiplier actually applied.
    """
    from scipy import stats

    hl = table[table["is_high_leverage"]]
    reg = table[~table["is_high_leverage"]]

    feature_tests = []
    x_cols = [f"feat_{name}" for name in FEATURE_NAMES]
    for col in x_cols:
        t_stat, p_value = stats.ttest_ind(hl[col], reg[col], equal_var=False)
        feature_tests.append({
            "feature": col, "hl_mean": float(hl[col].mean()), "reg_mean": float(reg[col].mean()),
            "hl_std": float(hl[col].std()), "reg_std": float(reg[col].std()),
            "t_stat": float(t_stat), "p_value": float(p_value),
        })

    margin_t, margin_p = stats.ttest_ind(hl["actual_margin"].abs(), reg["actual_margin"].abs(), equal_var=False)
    levene_stat, levene_p = stats.levene(hl["actual_margin"], reg["actual_margin"])

    bonferroni_alpha = 0.05 / len(x_cols)
    n_significant = sum(1 for t in feature_tests if t["p_value"] < bonferroni_alpha)

    return {
        "n_high_leverage": int(len(hl)),
        "n_regular": int(len(reg)),
        "feature_tests": feature_tests,
        "bonferroni_alpha": bonferroni_alpha,
        "n_features_significant": n_significant,
        "n_features_total": len(x_cols),
        "abs_margin_test": {"hl_mean": float(hl["actual_margin"].abs().mean()),
                             "reg_mean": float(reg["actual_margin"].abs().mean()),
                             "t_stat": float(margin_t), "p_value": float(margin_p)},
        "variance_test_levene": {"hl_var": float(hl["actual_margin"].var()),
                                  "reg_var": float(reg["actual_margin"].var()),
                                  "stat": float(levene_stat), "p_value": float(levene_p)},
    }


def print_high_leverage_variance_report(analysis: dict) -> None:
    print("\n========================================================")
    print(" HIGH-LEVERAGE GAME VARIANCE CHECK (Welch's t-test per feature)")
    print("========================================================")
    print(f" n_high_leverage={analysis['n_high_leverage']}, n_regular={analysis['n_regular']}, "
          f"Bonferroni-corrected alpha={analysis['bonferroni_alpha']:.5f}")
    print(f"{'Feature':<28}{'HL mean':>10}{'Reg mean':>10}{'t-stat':>9}{'p-value':>10}  Sig?")
    for t in analysis["feature_tests"]:
        sig = "yes" if t["p_value"] < analysis["bonferroni_alpha"] else "no"
        print(f"{t['feature']:<28}{t['hl_mean']:>10.2f}{t['reg_mean']:>10.2f}"
              f"{t['t_stat']:>9.2f}{t['p_value']:>10.4f}  {sig}")
    am = analysis["abs_margin_test"]
    lv = analysis["variance_test_levene"]
    print(f"\n |actual_margin|: HL mean={am['hl_mean']:.2f}, Reg mean={am['reg_mean']:.2f}, "
          f"t={am['t_stat']:.2f}, p={am['p_value']:.4f}")
    print(f" Levene's test (outcome variance equality): stat={lv['stat']:.3f}, p={lv['p_value']:.4f}")
    print(f"\n VERDICT: {analysis['n_features_significant']}/{analysis['n_features_total']} features "
          f"significant after Bonferroni correction.")
    if analysis["n_features_significant"] <= 1:
        print(" No credible evidence that high-leverage games differ structurally from ordinary")
        print(" games in this dataset (any single significant feature is expected to correlate")
        print(" with feat_big_match_indicator by construction, not an independent signal). This")
        print(" does NOT support a structural feature split OR a training-time sample-weight")
        print(" multiplier -- see DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER's docstring.")
    else:
        print(" Multiple features differ significantly -- worth investigating a structural")
        print(" feature-based treatment instead of (or alongside) sample weighting.")
    print("========================================================")


# Statistically determined via analyze_high_leverage_variance(): on this
# dataset (1225 games, 80 flagged high-leverage), 9 of 10 features show NO
# significant difference between high-leverage and ordinary games after
# Bonferroni correction (p-values 0.75-0.99), the one exception
# (feat_big_match_indicator, p<0.0001) is circular by construction (top-4
# standing correlates with roster-strength-derived net_rating_proxy, which
# IS that feature), and neither |actual_margin| (p=0.77) nor its variance
# (Levene's p=0.89) differ either -- i.e. high-leverage games are not
# measurably closer, more lopsided, or more/less variable than ordinary
# games in this data. There is therefore no statistical basis for an
# arbitrary training-time weight multiplier (an earlier "2.0" value was
# tried, WITHOUT this check, and empirically made the full hybrid pipeline
# WORSE -- see ml_model/hybrid_pipeline_benchmark.json's prior_configurations).
# 1.0 = a clean, statistically justified baseline: uniform sample weighting,
# no fabricated context multiplier. Re-run analyze_high_leverage_variance()
# against a larger/different dataset before overriding this.
DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER = 1.0


def build_high_leverage_sample_weight(is_high_leverage: pd.Series,
                                       high_leverage_weight_multiplier: float) -> Optional[np.ndarray]:
    """Training-time sample weight for high-leverage games (see
    compute_high_leverage_flags()). DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER
    (1.0) is a no-op by default -- see its own docstring for the statistical
    analysis (analyze_high_leverage_variance()) that justifies this: this
    dataset shows no credible evidence that high-leverage games differ
    structurally from ordinary ones, so weighting them differently has no
    statistical basis and empirically hurt full-pipeline accuracy when
    tried. Only pass a multiplier != 1.0 here if you have new evidence
    (e.g. from re-running analyze_high_leverage_variance() on more data)
    that justifies it -- this function still supports it mechanically, it's
    just not applied by default.

    Still deliberately a TRAINING-TIME SAMPLE WEIGHT, not a new predictive
    FEATURE fed to the model at inference time -- "is this a high-leverage
    game" isn't part of FEATURE_NAMES (see build_feature_vector()), so it
    can't leak into a future ad hoc matchup's prediction, and doesn't touch
    anything cpp_engine's calibrated Big Match/Hot Hand or fatigue effects
    already apply intrinsically.
    """
    if high_leverage_weight_multiplier == 1.0:
        return None
    return np.where(is_high_leverage.to_numpy(), high_leverage_weight_multiplier, 1.0)


# Chronological split of the ALREADY-held-out test set into a calibration-
# fit portion (earliest) and a calibration-eval portion (most recent) --
# see fit_win_probability_calibrator()'s docstring for why this, rather
# than fitting and evaluating calibration on the same games.
CALIBRATION_FIT_FRACTION = 0.7


def fit_win_probability_calibrator(holdout_table: pd.DataFrame, holdout_preds: np.ndarray) -> dict:
    """Fits a genuine, DATA-DRIVEN mapping from this model's raw predicted
    margin to an empirical P(Team A wins) -- replacing the previous ad hoc
    normal_cdf(margin / kRealMarginStdDev) shortcut (an ASSUMED Gaussian
    shape with a fixed, not-independently-fit std dev -- see
    backend/api_simulation.py's kRealMarginStdDev comment, which admits
    exactly this) with a mapping actually fit against real held-out
    win/loss outcomes.

    Compares two standard post-hoc calibration methods on this model's own
    held-out predictions, rather than assuming one is better:
      - Platt scaling: a 1-feature logistic regression, P(win) =
        sigmoid(a * margin + b) with a, b fit from data.
      - Isotonic regression: a non-parametric monotonic step-function
        mapping -- more flexible, but needs more data to avoid overfitting.
    Whichever has the lower Brier score on a genuinely held-out
    calibration-eval slice is deployed; the loser is still reported (see
    the returned `report` dict) so the choice is auditable, not asserted.

    The comparison itself uses a further CHRONOLOGICAL split of the
    holdout test set: the earliest CALIBRATION_FIT_FRACTION games fit both
    candidate calibrators, the most recent remainder evaluates them --
    genuinely out-of-sample, consistent with this project's "never look at
    future data" discipline throughout (see time_based_split()). Once a
    method is chosen this way, the DEPLOYED calibrator is refit on the
    FULL holdout (fit+eval games) for that method, since more data only
    helps once the method choice itself no longer depends on it.

    Returns {"model": None, "method": None, "report": {...}} (an explicit,
    checkable no-fit state, not a crash) if there aren't enough holdout
    games to split sensibly.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss

    ordered = holdout_table.reset_index(drop=True)
    pred_margin = np.asarray(holdout_preds, dtype=float)
    actual_win = (ordered["actual_margin"].to_numpy() > 0).astype(int)
    n = len(ordered)

    n_fit = int(n * CALIBRATION_FIT_FRACTION)
    n_eval = n - n_fit
    if n_fit < 10 or n_eval < 10:
        return {"model": None, "method": None,
                "report": {"reason": f"only {n} holdout games available -- too few to split into a "
                                      f"calibration-fit/eval pair (need >=10 in each)."}}

    fit_slice, eval_slice = slice(0, n_fit), slice(n_fit, n)

    platt = LogisticRegression()
    platt.fit(pred_margin[fit_slice].reshape(-1, 1), actual_win[fit_slice])
    platt_probs = platt.predict_proba(pred_margin[eval_slice].reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pred_margin[fit_slice], actual_win[fit_slice])
    iso_probs = iso.predict(pred_margin[eval_slice])

    y_eval = actual_win[eval_slice]

    def _safe_log_loss(y_true, probs) -> float:
        return float(log_loss(y_true, np.clip(probs, 1e-6, 1 - 1e-6)))

    platt_brier, iso_brier = float(brier_score_loss(y_eval, platt_probs)), float(brier_score_loss(y_eval, iso_probs))
    platt_logloss, iso_logloss = _safe_log_loss(y_eval, platt_probs), _safe_log_loss(y_eval, iso_probs)

    method = "platt" if platt_brier <= iso_brier else "isotonic"

    # Refit the CHOSEN method on every holdout game (fit+eval) for
    # deployment -- the split above is only for an honest method
    # comparison; once that choice is made, more data strictly helps.
    if method == "platt":
        deployed = LogisticRegression()
        deployed.fit(pred_margin.reshape(-1, 1), actual_win)
    else:
        deployed = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        deployed.fit(pred_margin, actual_win)

    report = {
        "method_chosen": method,
        "calibration_fit_n": n_fit,
        "calibration_eval_n": n_eval,
        "calibration_fit_date_range": [str(ordered["game_date"].iloc[0]), str(ordered["game_date"].iloc[n_fit - 1])],
        "calibration_eval_date_range": [str(ordered["game_date"].iloc[n_fit]), str(ordered["game_date"].iloc[-1])],
        "platt": {"brier": platt_brier, "log_loss": platt_logloss,
                  "coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0])},
        "isotonic": {"brier": iso_brier, "log_loss": iso_logloss},
    }
    return {"model": deployed, "method": method, "report": report}


def _raw_win_probability(calibrator, method: str, margin: float) -> float:
    """The calibrator's direct output for one margin -- the uniform interface
    across Platt (LogisticRegression) and Isotonic (IsotonicRegression),
    which have different raw predict APIs. NOT swap-symmetric on its own --
    see predict_win_probability(), the public wrapper that fixes that.
    """
    if method == "platt":
        return float(calibrator.predict_proba(np.array([[margin]]))[0, 1])
    if method == "isotonic":
        return float(calibrator.predict(np.array([margin]))[0])
    raise ValueError(f"Unknown calibration method: {method}")


def predict_win_probability(calibrator, method: str, margin: float) -> float:
    """Applies a fitted calibrator (see fit_win_probability_calibrator()) to
    one predicted margin, returning a genuine calibrated P(Team A wins),
    EXACTLY complementary under a team swap by construction: this value plus
    predict_win_probability(calibrator, method, -margin) always sums to
    exactly 1.0.

    That guarantee does NOT come for free from _raw_win_probability() alone:
    Platt scaling is sigmoid(coef*margin + intercept), and the fitted
    intercept is generally nonzero (it reflects how often the team labeled
    "team_a" won in the calibration-fit data independent of margin) -- and
    since historical_games.csv has team_a as ALWAYS the real home team (see
    predict_margin()'s docstring), that intercept is itself a residual of
    the same team_a-labeling confound, not a genuine base rate a live
    matchup should inherit. sigmoid(x) + sigmoid(-x) == 1 only holds when
    the two arguments are exact negatives of each other, which requires a
    zero intercept -- so the raw calibrator alone under- or over-states
    P(A wins) + P(B wins) away from 1 by a small, margin-dependent amount
    (observed: ~6 points off on one real matchup) whenever the intercept
    isn't ~0. Fix: the same averaging trick as predict_margin() -- combine
    the raw estimate for `margin` with the complement of the raw estimate
    for `-margin`. This is antisymmetric-safe for Isotonic too (it has no
    "intercept" as such, but is any generally-non-antisymmetric fitted
    curve, so the same construction is needed and applies unchanged).
    """
    raw_forward = _raw_win_probability(calibrator, method, margin)
    raw_reverse = _raw_win_probability(calibrator, method, -margin)
    return (raw_forward + (1.0 - raw_reverse)) / 2.0


def print_calibration_report(report: dict) -> None:
    print("\n========================================================")
    print(" WIN-PROBABILITY CALIBRATION (post-hoc, fit on held-out predictions)")
    print("========================================================")
    if report.get("method_chosen") is None:
        print(f" SKIPPED: {report.get('reason', 'no reason given')}")
        print("========================================================")
        return
    print(f" Calibration-fit games  : {report['calibration_fit_n']}  "
          f"({report['calibration_fit_date_range'][0]} to {report['calibration_fit_date_range'][1]})")
    print(f" Calibration-eval games : {report['calibration_eval_n']}  "
          f"({report['calibration_eval_date_range'][0]} to {report['calibration_eval_date_range'][1]})"
          f"  <- genuinely out-of-sample for this comparison")
    print(f"{'Method':<14}{'Brier':>10}{'LogLoss':>10}")
    print(f"{'Platt scaling':<14}{report['platt']['brier']:>10.4f}{report['platt']['log_loss']:>10.4f}")
    print(f"{'Isotonic':<14}{report['isotonic']['brier']:>10.4f}{report['isotonic']['log_loss']:>10.4f}")
    print(f" DEPLOYED: {report['method_chosen']} (lower Brier score on the calibration-eval slice)")
    if report["method_chosen"] == "platt":
        print(f" P(A wins) = sigmoid({report['platt']['coef']:.5f} * margin + {report['platt']['intercept']:.5f})")
    print("========================================================")


def get_feature_importances(model) -> list[tuple[str, float]]:
    """Pairs FEATURE_NAMES with the fitted model's gain-based importances
    (how much each feature's splits reduced training loss), sorted
    descending -- so it's easy to see whether the model actually leans on
    spacing, shooting volume, usage, rest, or star availability.
    """
    importances = model.feature_importances_
    pairs = list(zip(MODEL_FEATURE_NAMES, (float(v) for v in importances)))
    return sorted(pairs, key=lambda p: p[1], reverse=True)


def save_artifacts(out_dir: Path, model, team_features: dict[str, TeamFeatures],
                    split_mode: str, split_info: dict, eval_report: dict,
                    holdout_table: pd.DataFrame, holdout_preds: np.ndarray,
                    feature_importances: list[tuple[str, float]],
                    calibration: Optional[dict] = None) -> None:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "margin_model.joblib")

    with (out_dir / "team_features.json").open("w", encoding="utf-8") as f:
        json.dump({abbr: asdict(feat) for abbr, feat in team_features.items()}, f, indent=2)

    with (out_dir / "feature_importances.json").open("w", encoding="utf-8") as f:
        json.dump([{"feature": name, "importance": value} for name, value in feature_importances],
                   f, indent=2)

    calibration_report = None
    if calibration is not None and calibration.get("model") is not None:
        joblib.dump({"model": calibration["model"], "method": calibration["method"]},
                    out_dir / "margin_calibrator.joblib")
        calibration_report = calibration["report"]
    elif calibration is not None:
        calibration_report = calibration["report"]
    with (out_dir / "calibration_report.json").open("w", encoding="utf-8") as f:
        json.dump(calibration_report or {}, f, indent=2)

    holdout_records = []
    for (_, row), pred in zip(holdout_table.iterrows(), holdout_preds):
        holdout_records.append({
            "team_a": row.team_a,
            "team_b": row.team_b,
            "actual_winner": row.actual_winner,
            "actual_score_a": row.actual_score_a,
            "actual_score_b": row.actual_score_b,
            "actual_margin": row.actual_margin,
            "game_date": str(row.get("game_date", "")),
            "holdout_predicted_margin": float(pred),
            "big_match_indicator": float(row["feat_big_match_indicator"]),
            "is_high_leverage": bool(row["is_high_leverage"]),
            "is_playoff": bool(row.get("is_playoff", False)),
            "is_top4_matchup": bool(row.get("is_top4_matchup", False)),
            "is_deep_playoff_rematch": bool(row.get("is_deep_playoff_rematch", False)),
            "is_tournament_knockout": bool(row.get("is_tournament_knockout", False)),
            "team_a_rest_days": float(row.get("team_a_rest_days", 0.0)),
            "team_b_rest_days": float(row.get("team_b_rest_days", 0.0)),
        })
    with (out_dir / "holdout_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({
            "split_mode": split_mode,
            "split_info": split_info,
            "eval_report": eval_report,
            "games": holdout_records,
        }, f, indent=2)


def predict_margin(model, team_features: dict[str, TeamFeatures], team_a: str, team_b: str) -> float:
    """Ad hoc margin prediction for team_a vs team_b, EXACTLY antisymmetric
    under a team_a/team_b swap by construction: predict_margin(A, B) ==
    -predict_margin(B, A) always holds, regardless of anything the raw
    XGBoost model itself learned.

    This guarantee is necessary on top of MODEL_FEATURE_NAMES excluding
    big_match_indicator (see its docstring) rather than instead of it:
    even with fully antisymmetric *inputs*, a gradient-boosted tree ensemble
    has no inherent reason to satisfy f(-x) == -f(x) -- nothing in training
    enforces it. Worse, historical_games.csv has team_a as ALWAYS the real
    home team (see calibrate_engine.py's calibration note), so every
    training row implicitly associates "being labeled team_a" with a
    home-court-sized scoring boost baked into actual_margin itself, which
    the model can and does partially absorb regardless of which features
    are antisymmetric. A live matchup request has no such convention -- the
    caller's team_a is not necessarily home (home_team is a separate field)
    -- so this asymmetry must be removed at inference time.
    Fix: average the model's prediction over BOTH orderings, negating the
    second, i.e. (predict(A,B) - predict(B,A)) / 2. This expression is
    antisymmetric under a swap by pure algebra for ANY underlying model,
    not just this one -- it doesn't require the raw model to have learned
    a symmetric function itself. Two model.predict() calls instead of one;
    negligible cost for a model this size.
    """
    team_a, team_b = team_a.upper(), team_b.upper()
    if team_a not in team_features or team_b not in team_features:
        missing = team_a if team_a not in team_features else team_b
        raise MlPipelineError(f"No cached features for team '{missing}'. Re-run training first.")
    # No specific game context for an ad hoc matchup: assumes full health.
    feat_vec_ab = to_model_features(build_feature_vector(team_features[team_a], team_features[team_b],
                                                          star_out_diff=0.0)).reshape(1, -1)
    feat_vec_ba = to_model_features(build_feature_vector(team_features[team_b], team_features[team_a],
                                                          star_out_diff=0.0)).reshape(1, -1)
    margin_ab = float(model.predict(feat_vec_ab)[0])
    margin_ba = float(model.predict(feat_vec_ba)[0])
    return (margin_ab - margin_ba) / 2.0


def _predict_margin_at_boost(model, team_features: dict[str, TeamFeatures], team_a: str, team_b: str,
                              hot_hand_boost: float) -> float:
    """One real re-inference through the trained model with both teams'
    top_player_usage scaled by `hot_hand_boost` -- see
    predict_margin_with_context()'s docstring for what this mirrors.
    Symmetrized exactly like predict_margin() -- see its docstring for why.
    """
    fa = replace(team_features[team_a],
                 top_player_usage=team_features[team_a].top_player_usage * hot_hand_boost)
    fb = replace(team_features[team_b],
                 top_player_usage=team_features[team_b].top_player_usage * hot_hand_boost)
    feat_vec_ab = to_model_features(build_feature_vector(fa, fb, star_out_diff=0.0)).reshape(1, -1)
    feat_vec_ba = to_model_features(build_feature_vector(fb, fa, star_out_diff=0.0)).reshape(1, -1)
    margin_ab = float(model.predict(feat_vec_ab)[0])
    margin_ba = float(model.predict(feat_vec_ba)[0])
    return (margin_ab - margin_ba) / 2.0


# A boost large enough to reliably cross this gradient-boosted tree
# ensemble's coarse split thresholds on star-usage features for most
# matchups (empirically checked: ~80% of real team pairs show measurable
# movement here, vs. ~25% at the realistic 1.05-1.2 toggle range) -- used
# ONLY as a probe point for predict_margin_with_context()'s interpolation,
# never returned directly.
_HOT_HAND_PROBE_BOOST = 1.5


def predict_margin_with_context(model, team_features: dict[str, TeamFeatures], team_a: str, team_b: str,
                                 hot_hand_boost: float = 1.0) -> float:
    """Same as predict_margin(), but reflects a "Big Match / Hot Hand
    Boost" toggle by re-inferring through the ALREADY-TRAINED model on a
    counterfactual feature vector: each team's top_player_usage (its
    highest-usage rotation player) is scaled by a boost factor before the
    diff/pairwise features are built -- mirroring EXACTLY how
    cuda_simulator.cu's own --hot-hand-boost flag amplifies a marquee
    matchup's star's effective usage/shooting rate intrinsically (see
    simulate_possession's `off_usage[star_idx] *= hot_hand_boost`).
    hot_hand_boost=1.0 (the default/no-toggle case) is an exact no-op,
    identical to predict_margin().

    Boosting BOTH teams' top_player_usage by the same factor -- not just
    the "favored" side -- is deliberate: the engine applies the multiplier
    symmetrically to whichever team is on offense, so the net effect on
    margin comes entirely from amplifying whatever star-usage GAP already
    exists between the two teams (feeding straight into this model's own
    trained `star_usage_concentration`/top_player_usage-diff features),
    exactly as intended for a "the bigger the stage, the more the stars
    take over" effect -- not from inventing a one-sided bonus.

    Linear interpolation, not a single re-inference at `hot_hand_boost`
    itself: a shallow (max_depth=3) gradient-boosted tree ensemble is
    locally flat between its own split thresholds, so re-inferring at the
    realistic 1.05-1.2 range often lands in the SAME leaf as the unboosted
    baseline and returns an identical prediction for a given matchup --
    honest, but defeats the point of a toggle meant to visibly move. So
    this instead takes TWO real model evaluations -- at hot_hand_boost=1.0
    (the base prediction) and at a fixed, larger `_HOT_HAND_PROBE_BOOST`
    reference point chosen to reliably cross those thresholds -- and
    linearly interpolates/extrapolates between them for the actual
    requested `hot_hand_boost`. Both anchor points are genuine model
    outputs; nothing here is an invented sensitivity constant. If the model
    truly has zero sensitivity to star usage for a specific matchup even at
    the probe boost, this correctly returns the unboosted base prediction
    unchanged -- an honest result, not a bug.
    """
    team_a, team_b = team_a.upper(), team_b.upper()
    if team_a not in team_features or team_b not in team_features:
        missing = team_a if team_a not in team_features else team_b
        raise MlPipelineError(f"No cached features for team '{missing}'. Re-run training first.")

    base_margin = predict_margin(model, team_features, team_a, team_b)
    if abs(hot_hand_boost - 1.0) < 1e-9:
        return base_margin

    probe_margin = _predict_margin_at_boost(model, team_features, team_a, team_b, _HOT_HAND_PROBE_BOOST)
    frac = (hot_hand_boost - 1.0) / (_HOT_HAND_PROBE_BOOST - 1.0)
    return base_margin + frac * (probe_margin - base_margin)


def print_training_report(split_mode: str, split_info: dict, eval_report: dict,
                           holdout_table: pd.DataFrame, holdout_preds: np.ndarray,
                           feature_importances: list[tuple[str, float]]) -> None:
    p = XGB_PARAMS
    print("\n========================================================")
    if split_mode == "time":
        print(" ML MARGIN MODEL (XGBoost) -- CHRONOLOGICAL TRAIN/TEST VALIDATION")
    else:
        print(" ML MARGIN MODEL (XGBoost) -- LEAVE-ONE-OUT CROSS-VALIDATION")
    print("========================================================")
    if split_mode == "time":
        print(f" Train games   : {split_info['n_train']}  "
              f"({split_info['train_date_range'][0]} to {split_info['train_date_range'][1]})")
        print(f" Test games    : {split_info['n_test']}  "
              f"({split_info['test_date_range'][0]} to {split_info['test_date_range'][1]})"
              f"  <- held out, never seen during training")
    else:
        print(f" Training games (LOOCV folds): {split_info['n_games']}")
    print(f" Model         : XGBRegressor(n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
          f"learning_rate={p['learning_rate']}, reg_alpha={p['reg_alpha']}, reg_lambda={p['reg_lambda']})")
    print(f" Features      : {len(MODEL_FEATURE_NAMES)} model inputs "
          f"({len(BASE_FEATURE_NAMES)} base + {len(MODEL_FEATURE_NAMES) - len(BASE_FEATURE_NAMES)} pairwise; "
          f"big_match_indicator kept diagnostic-only, not a model input -- see MODEL_FEATURE_NAMES)")
    hl_mult = split_info.get("high_leverage_weight_multiplier", 1.0)
    if hl_mult != 1.0:
        print(f" Sample weight : high-leverage games (real playoff/top-4-contender/prior-Finals-"
              f"rematch/tournament -- see compute_high_leverage_flags()) weighted {hl_mult}x during "
              f"training (ordinary games = 1.0x)")
    else:
        print(f" Sample weight : uniform (1.0x) -- high-leverage weighting off")
    print(f" Holdout mean absolute error : {eval_report['mae']:.2f} points")
    print(f" Holdout R^2                 : {eval_report['r2']:.3f}")
    print(f" Holdout directional accuracy: {eval_report['directional_accuracy'] * 100:.1f}%"
          f"  (predicted margin sign matches actual margin sign)")
    print("--------------------------------------------------------")
    n_high_leverage = int(holdout_table["is_high_leverage"].sum()) if "is_high_leverage" in holdout_table else 0
    n_b2b = int(((holdout_table["team_a_rest_days"] == 0) | (holdout_table["team_b_rest_days"] == 0)).sum())
    n_star_out = int((holdout_table["feat_star_out_diff"] != 0).sum())
    print(f"{'Matchup':<12}{'Date':<12}{'Actual':>9}{'Predicted':>12}  Flags")
    for (_, row), pred in zip(holdout_table.iterrows(), holdout_preds):
        matchup = f"{row.team_a}-{row.team_b}"
        flags = ""
        if row.get("is_high_leverage", False):
            flags += " HL"
        if row.get("team_a_rest_days", 1) == 0 or row.get("team_b_rest_days", 1) == 0:
            flags += " B2B"
        if row.get("feat_star_out_diff", 0.0) != 0.0:
            flags += " STAR-OUT"
        print(f"{matchup:<12}{str(row.get('game_date', '')):<12}{row.actual_margin:>+9.1f}{pred:>+12.2f} {flags}")
    print(f" ({n_high_leverage}/{len(holdout_table)} high-leverage (real: playoff/top-4-contender/"
          f"prior-Finals-rematch/tournament), {n_b2b}/{len(holdout_table)} involve a back-to-back team, "
          f"{n_star_out}/{len(holdout_table)} involve a flagged star absence)")
    print("--------------------------------------------------------")
    print(" Feature importances (gain-based; final model refit on all games):")
    max_importance = max((v for _, v in feature_importances), default=0.0) or 1.0
    for name, value in feature_importances:
        bar_len = round(20 * value / max_importance)
        print(f"   {name:<24}{value:>8.4f}  {'#' * bar_len}")
    print("========================================================")
    if split_mode == "loocv":
        print(" NOTE: no usable game_date column was found (or too few rows), so this")
        print(" run fell back to leave-one-out CV instead of a chronological holdout.")
    print(" CAVEAT: team features come from the CURRENT roster snapshot at /api/players,")
    print(" not each historical game's actual point-in-time roster. star_out_diff and its")
    print(" gravity/usage downgrade are REAL per-game data (that game's actual box score).")
    print(" Defensive rating, rest days, and home court are intentionally NOT features here")
    print(" -- cpp_engine's engine now applies those intrinsically (calibrate_engine.py);")
    print(" including them here too would double-count them. Read holdout numbers")
    print(" skeptically at this dataset size.")
    print("========================================================\n")


def train(csv_path: Path, api_url: str, out_dir: Path,
          test_fraction: float = DEFAULT_TEST_FRACTION, split_mode: str = "auto",
          season: str = DEFAULT_SEASON,
          high_leverage_weight_multiplier: float = DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER) -> None:
    players = fetch_players(api_url)
    players_df = _prep_players_df(players)

    print(f"Fetching {season} team defensive ratings from nba_api...")
    def_ratings = fetch_team_defensive_ratings(season)

    team_features, star_names = compute_team_features(players_df, def_ratings)

    print(f"Fetching {season} conference standings and prior-season deep-playoff "
          f"matchups from nba_api (for real high-leverage game classification)...")
    top4_teams = fetch_conference_top_n(season)
    rematch_pairs = fetch_deep_playoff_rematch_pairs(season)

    star_cache_path = out_dir / "star_availability_cache.json"
    star_cache = {}
    if star_cache_path.is_file():
        with star_cache_path.open(encoding="utf-8") as f:
            star_cache = json.load(f)

    print("Checking per-game star availability against real box scores "
          "(cached after first run)...")
    table = load_training_table(csv_path, players_df, team_features, star_names, def_ratings, star_cache,
                                 season=season, top4_teams=top4_teams, rematch_pairs=rematch_pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    with star_cache_path.open("w", encoding="utf-8") as f:
        json.dump(star_cache, f, indent=2)

    resolved_mode = choose_split_mode(table, split_mode)
    # MODEL_FEATURE_NAMES (not FEATURE_NAMES) -- excludes the symmetric-under-
    # swap big_match_indicator, which is diagnostic-only, not a model input.
    x_cols = [f"feat_{name}" for name in MODEL_FEATURE_NAMES]

    if resolved_mode == "time":
        train_table, test_table = time_based_split(table, test_fraction)
        train_weight = build_high_leverage_sample_weight(
            train_table["is_high_leverage"], high_leverage_weight_multiplier)
        holdout_preds, eval_report = evaluate_time_holdout(
            train_table[x_cols].to_numpy(), train_table["actual_margin"].to_numpy(),
            test_table[x_cols].to_numpy(), test_table["actual_margin"].to_numpy(),
            sample_weight_train=train_weight,
        )
        holdout_table = test_table
        split_info = {
            "n_train": len(train_table),
            "n_test": len(test_table),
            "test_fraction": test_fraction,
            "train_date_range": [str(train_table["game_date"].iloc[0]), str(train_table["game_date"].iloc[-1])],
            "test_date_range": [str(test_table["game_date"].iloc[0]), str(test_table["game_date"].iloc[-1])],
            "high_leverage_weight_multiplier": high_leverage_weight_multiplier,
            "top4_teams": sorted(top4_teams),
            "deep_playoff_rematch_pairs": [sorted(p) for p in rematch_pairs],
        }
    else:
        X_all = table[x_cols].to_numpy()
        y_all = table["actual_margin"].to_numpy()
        all_weight = build_high_leverage_sample_weight(table["is_high_leverage"], high_leverage_weight_multiplier)
        holdout_preds, eval_report = evaluate_loocv(X_all, y_all, sample_weight=all_weight)
        holdout_table = table
        split_info = {"n_games": len(table), "high_leverage_weight_multiplier": high_leverage_weight_multiplier,
                      "top4_teams": sorted(top4_teams), "deep_playoff_rematch_pairs": [sorted(p) for p in rematch_pairs]}

    # Final deployment model: refit on every available row (train + test),
    # with the SAME high-leverage sample weighting used for the holdout fit above.
    X_all = table[x_cols].to_numpy()
    y_all = table["actual_margin"].to_numpy()
    final_weight = build_high_leverage_sample_weight(table["is_high_leverage"], high_leverage_weight_multiplier)
    final_model = fit_final_model(X_all, y_all, sample_weight=final_weight)
    feature_importances = get_feature_importances(final_model)

    print_training_report(resolved_mode, split_info, eval_report, holdout_table, holdout_preds,
                           feature_importances)

    # Statistical check (see analyze_high_leverage_variance()'s docstring):
    # does the data actually support treating high-leverage games
    # differently, structurally or via sample weight? Run every time, not
    # just when a non-default multiplier is requested, so this stays a
    # standing, reproducible check rather than a one-off justification.
    variance_analysis = analyze_high_leverage_variance(table)
    print_high_leverage_variance_report(variance_analysis)

    # Genuine post-hoc probability calibration (see
    # fit_win_probability_calibrator()'s docstring) -- replaces the
    # assumed-Gaussian normal_cdf(margin / kRealMarginStdDev) shortcut with
    # a mapping actually fit against real held-out win/loss outcomes.
    calibration = fit_win_probability_calibrator(holdout_table, holdout_preds)
    print_calibration_report(calibration["report"])

    save_artifacts(out_dir, final_model, team_features, resolved_mode, split_info, eval_report,
                    holdout_table, holdout_preds, feature_importances, calibration=calibration)
    with (out_dir / "high_leverage_variance_report.json").open("w", encoding="utf-8") as f:
        json.dump(variance_analysis, f, indent=2)
    print(f"Saved model + features + holdout predictions + feature importances + calibration + "
          f"variance report to {out_dir}/\n")


def predict_cli(team_a: str, team_b: str, out_dir: Path) -> int:
    import joblib

    model_path = out_dir / "margin_model.joblib"
    features_path = out_dir / "team_features.json"
    if not model_path.is_file() or not features_path.is_file():
        print(f"[train_ml_model] No trained model found in {out_dir}/ -- run "
              f"`python train_ml_model.py` first.", file=sys.stderr)
        return 1

    model = joblib.load(model_path)
    with features_path.open(encoding="utf-8") as f:
        raw_features = json.load(f)
    team_features = {abbr: TeamFeatures(**vals) for abbr, vals in raw_features.items()}

    try:
        margin = predict_margin(model, team_features, team_a, team_b)
    except MlPipelineError as e:
        print(f"[train_ml_model] {e}", file=sys.stderr)
        return 1

    print(f"{margin:+.2f}")

    # Line 1 stays margin-only for scripting (see this function's own
    # docstring); the calibrated win probability, if a calibrator was
    # saved by the last training run, is a second, optional line.
    calibrator_path = out_dir / "margin_calibrator.joblib"
    if calibrator_path.is_file():
        saved = joblib.load(calibrator_path)
        prob_a = predict_win_probability(saved["model"], saved["method"], margin)
        print(f"P({team_a.upper()} wins) = {prob_a * 100:.1f}%  (calibrated via {saved['method']})")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the ML expected-point-margin model, or predict one matchup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                         help=f"Historical games CSV (default: {DEFAULT_CSV.name})")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL,
                         help=f"Backend player-data endpoint (default: {DEFAULT_API_URL})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                         help=f"Where to save the trained model/features (default: {DEFAULT_OUT_DIR.name}/)")
    parser.add_argument("--season", type=str, default=DEFAULT_SEASON,
                         help=f"Season for nba_api's team defensive-rating lookup "
                              f"(should match --csv's season; default: {DEFAULT_SEASON})")
    parser.add_argument("--split-mode", choices=["auto", "time", "loocv"], default="auto",
                         help="'time' = chronological train/test holdout (needs game_date); "
                              "'loocv' = leave-one-out CV; 'auto' (default) picks 'time' when "
                              f"game_date is present and there are >= {MIN_ROWS_FOR_TIME_SPLIT} rows")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
                         help=f"Fraction of games held out as the chronological test set "
                              f"(--split-mode time only; default: {DEFAULT_TEST_FRACTION})")
    parser.add_argument("--high-leverage-weight-multiplier", type=float,
                         default=DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER,
                         help="Training-time sample weight for real high-leverage games (playoff, top-4-"
                              f"conference-contender matchup, prior-season Finals/Conf-Finals rematch, or "
                              f"In-Season Tournament knockout game) relative to 1.0 for an ordinary game. "
                              f"Default is 1.0 (uniform/off) because analyze_high_leverage_variance() found "
                              f"no statistical evidence on this dataset that high-leverage games differ "
                              f"structurally -- see this module's docstring and "
                              f"ml_model/high_leverage_variance_report.json before overriding this.")
    parser.add_argument("--predict", nargs=2, metavar=("TEAM_A", "TEAM_B"), default=None,
                         help="Skip training; load a saved model and print the predicted margin "
                              "for TEAM_A vs TEAM_B (prints only the number, for scripting)")
    args = parser.parse_args(argv)

    if args.predict:
        return predict_cli(args.predict[0], args.predict[1], args.out_dir)

    try:
        train(args.csv, args.api_url, args.out_dir, args.test_fraction, args.split_mode, args.season,
              args.high_leverage_weight_multiplier)
    except MlPipelineError as e:
        print(f"[train_ml_model] Fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
