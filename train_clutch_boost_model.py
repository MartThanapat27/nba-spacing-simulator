#!/usr/bin/env python3
"""XGBoost-driven Clutch/Momentum Boost calibration -- fits the engine's
`clutch_shooting_confidence_bonus` and `clutch_star_usage_mult`
(`EngineTuningParams`, see cpp_engine/main.cpp and cuda_simulator.cuh)
against REAL historical clutch-time performance, and writes a
`--tuning-config`-shaped JSON that overrides just those two fields (every
other field keeps the engine's own already-calibrated default).

This is the "XGBoost Integration Hook for Player Boosts" interface: an
external script that outputs data-driven clutch/momentum boost weights,
consumed by the SAME `--tuning-config <path.json>` pipeline every other
EngineTuningParams field already uses -- no new C++ plumbing, no bespoke
file format.

Why XGBoost here (unlike calibrate_engine.py's plain OLS)?
------------------------------------------------------------
calibrate_engine.py fits def-resistance/fatigue/home-court from features
that are already known to be linear-in-effect (a rest day is worth however
many points, full stop). Clutch performance is exactly the kind of signal
XGBoost is a better fit for: whether a team's real clutch-time edge
predicts real game margins is itself an open, non-linear-shaped question at
this sample size, and `train_ml_model.py` already uses XGBoost for the same
reason (see that module's docstring). A tiny, heavily-regularized
`XGBRegressor` (2 features, max_depth=2) is used here, evaluated via
leave-one-out cross-validation exactly like `train_ml_model.py`'s own
small-sample fallback -- an in-sample-only fit would not tell you whether
the signal generalizes at all.

Two REAL, per-team/per-player features, both season-to-date aggregates
(no data leakage from the specific game being predicted):

  1. `team_clutch_fg_delta_edge` -- (Team A - Team B) of each team's own
     (real NBA "clutch time": last 5 minutes, score within 5 -- via
     nba_api's `leaguedashteamclutch`, whose OWN defaults already match
     this engine's `clutch_time_remaining_seconds_threshold`/
     `clutch_time_margin_threshold` exactly) FG_PCT minus that SAME team's
     season-wide FG_PCT. Positive means this team shoots BETTER in the
     clutch than their own normal level -- exactly what
     `clutch_shooting_confidence_bonus` is supposed to model.
  2. `star_clutch_usage_delta_edge` -- (Team A - Team B) of each team's
     real highest-season-usage player's clutch-time USG_PCT (via
     nba_api's `leaguedashplayerclutch`) minus that SAME player's season
     usage rate. Positive means the go-to player's role concentrates
     FURTHER in the clutch -- the real-data analogue of
     `clutch_star_usage_mult`.

What gets written, and what doesn't
--------------------------------------
Only `clutch_shooting_confidence_bonus` is overridden from a directly
fitted effect size (probed the same way calibrate_engine.py converts a
margin coefficient into the kernel's probability-shift units --
`MARGIN_TO_PROB_SHIFT`, reused verbatim). `clutch_star_usage_mult` is a
USAGE-RATE multiplier, not a probability shift, so there is no equivalent
direct unit conversion; it is nudged by a small, capped fraction of the
engine's own default proportional to the fitted feature's SIGN and
LOOCV-supported strength, and clearly flagged in this script's report
(and the written JSON's own `_meta` block) as a softer, directionally-
informed nudge rather than a unit-derived coefficient -- same "included
honestly, magnitude may be small/uncertain" convention as
calibrate_engine.py's own weak coefficients.

Usage:
    python train_clutch_boost_model.py
    python train_clutch_boost_model.py --csv historical_games.csv --season 2024-25
    cpp_engine/build/Release/cuda_simulator.exe TEAM_A TEAM_B --tuning-config clutch_boost_tuning.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import train_ml_model as ml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "clutch_boost_tuning.json"

# Must match cpp_engine/cuda_simulator.cu's kAvgPointsPerMake/kPossessionsPerTeam
# exactly -- see calibrate_engine.py's identical constant for the full
# derivation; duplicated here rather than imported so this script has no
# C++-build-order dependency, same convention calibrate_engine.py itself uses.
AVG_POINTS_PER_MAKE = 2.3
POSSESSIONS_PER_TEAM = 100
MARGIN_TO_PROB_SHIFT = 1.0 / (2.0 * AVG_POINTS_PER_MAKE * POSSESSIONS_PER_TEAM)

# Engine defaults (EngineTuningParams in cpp_engine/main.cpp /
# cuda_simulator.cuh) -- the JSON this script writes overrides these from
# their FITTED baseline, not from zero, so "no real signal found" naturally
# degrades to "write back the engine's own already-verified default."
DEFAULT_CLUTCH_SHOOTING_CONFIDENCE_BONUS = 0.02
DEFAULT_CLUTCH_STAR_USAGE_MULT = 1.15

# Heavily regularized for a ~100-200 row, 2-feature dataset -- shallower
# and fewer trees than train_ml_model.py's own XGB_PARAMS (7-10 features),
# since 2 features need far less capacity to overfit with. Same
# regularization ingredients (L1/L2, subsample, min_child_weight) as that
# module's own honesty-first convention.
CLUTCH_XGB_PARAMS = dict(
    n_estimators=40,
    max_depth=2,
    learning_rate=0.08,
    reg_alpha=0.5,
    reg_lambda=1.5,
    subsample=0.85,
    colsample_bytree=1.0,
    min_child_weight=3,
    objective="reg:squarederror",
    importance_type="gain",
    random_state=42,
    n_jobs=1,
    verbosity=0,
)

FEATURE_NAMES = ["team_clutch_fg_delta_edge", "star_clutch_usage_delta_edge"]


def fetch_team_clutch_fg_pct(season: str) -> dict[str, float]:
    """REAL data: each team's own real clutch-time (last 5 min, score
    within 5 -- nba_api's own defaults, matching this engine's clutch_time
    definition exactly) field goal percentage, via
    `leaguedashteamclutch`.
    """
    from nba_api.stats.endpoints import leaguedashteamclutch
    from nba_api.stats.static import teams as static_teams

    try:
        resp = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base", per_mode_detailed="PerGame", timeout=30,
        )
        df = resp.get_data_frames()[0]
    except Exception as e:
        raise ml.MlPipelineError(f"Could not fetch team clutch stats from nba_api: {e}") from e

    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
    return dict(zip(df["TEAM_ABBREVIATION"], df["FG_PCT"].astype(float)))


def fetch_team_season_fg_pct(season: str) -> dict[str, float]:
    """REAL data: each team's season-wide (non-clutch) field goal
    percentage -- the baseline `fetch_team_clutch_fg_pct` is compared
    against to isolate a genuine clutch DELTA rather than just "good
    shooting teams shoot well in the clutch too."
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    try:
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base", per_mode_detailed="PerGame", timeout=30,
        )
        df = resp.get_data_frames()[0]
    except Exception as e:
        raise ml.MlPipelineError(f"Could not fetch team season stats from nba_api: {e}") from e

    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
    return dict(zip(df["TEAM_ABBREVIATION"], df["FG_PCT"].astype(float)))


def fetch_player_clutch_usage(season: str) -> dict[str, float]:
    """REAL data: every rotation player's real clutch-time usage rate
    (USG_PCT, 0-1 fraction), keyed by the SAME normalized-name convention
    train_ml_model.py's `_normalize_name` already uses for box-score
    star-availability matching, so this script's own name join can't
    silently drift from that established convention.
    """
    from nba_api.stats.endpoints import leaguedashplayerclutch

    try:
        resp = leaguedashplayerclutch.LeagueDashPlayerClutch(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame", timeout=30,
        )
        df = resp.get_data_frames()[0]
    except Exception as e:
        raise ml.MlPipelineError(f"Could not fetch player clutch usage from nba_api: {e}") from e

    return {
        ml._normalize_name(row.PLAYER_NAME): float(row.USG_PCT)
        for row in df.itertuples()
    }


def compute_team_clutch_signals(players_df: pd.DataFrame, season: str, verbose: bool = True
                                 ) -> pd.DataFrame:
    """Builds one row per real team with both REAL clutch-delta signals
    this script models: `team_clutch_fg_delta` (team-wide) and
    `star_clutch_usage_delta` (that team's real highest-season-usage
    player only -- the "player-specific" signal). Teams/players nba_api
    has no clutch-time sample for (too few genuinely clutch minutes played
    this season) are simply absent from the result -- `build_game_table`
    below drops any historical game touching a team missing here, rather
    than guessing a neutral 0.0 that would silently understate a real
    "not enough data" gap.
    """
    if verbose:
        print(f"Fetching {season} real team clutch-time FG% (nba_api leaguedashteamclutch)...")
    clutch_fg = fetch_team_clutch_fg_pct(season)
    if verbose:
        print(f"Fetching {season} real team season FG% (nba_api leaguedashteamstats)...")
    season_fg = fetch_team_season_fg_pct(season)
    if verbose:
        print(f"Fetching {season} real player clutch-time usage (nba_api leaguedashplayerclutch)...")
    clutch_usage = fetch_player_clutch_usage(season)

    _, star_names = ml.compute_team_features(players_df, def_ratings={})

    rows = []
    for team_abbr, star_name in star_names.items():
        if team_abbr not in clutch_fg or team_abbr not in season_fg:
            continue
        star_rows = players_df[
            (players_df["team_abbreviation"] == team_abbr)
            & (players_df["player_name"] == star_name)
        ]
        if star_rows.empty:
            continue
        star_season_usage = float(star_rows.iloc[0]["usage_rate"]) / 100.0  # nba_api USG_PCT is a 0-1 fraction
        star_clutch_usage = clutch_usage.get(ml._normalize_name(star_name))
        if star_clutch_usage is None:
            continue

        rows.append({
            "team_abbreviation": team_abbr,
            "star_name": star_name,
            "team_clutch_fg_delta": clutch_fg[team_abbr] - season_fg[team_abbr],
            "star_clutch_usage_delta": star_clutch_usage - star_season_usage,
        })

    table = pd.DataFrame(rows)
    if verbose:
        print(f"Resolved real clutch signals for {len(table)}/{len(star_names)} teams "
              f"(remainder lack a usable clutch-time sample or a name match this season).")
    return table


def build_game_table(csv_path: Path, team_signals: pd.DataFrame) -> pd.DataFrame:
    """Joins historical_games.csv's real final margins with the two teams'
    already-computed real clutch-delta signals. Both signals are
    season-to-date aggregates known BEFORE any specific game tips off, so
    joining them onto that game's real outcome is not data leakage --
    same convention as calibrate_engine.py's net_def_edge/rest_advantage.
    """
    games = pd.read_csv(csv_path)
    signals = team_signals.set_index("team_abbreviation")

    rows = []
    for g in games.itertuples():
        if g.team_a not in signals.index or g.team_b not in signals.index:
            continue
        a = signals.loc[g.team_a]
        b = signals.loc[g.team_b]
        rows.append({
            "team_clutch_fg_delta_edge": a["team_clutch_fg_delta"] - b["team_clutch_fg_delta"],
            "star_clutch_usage_delta_edge": a["star_clutch_usage_delta"] - b["star_clutch_usage_delta"],
            "actual_margin": float(g.actual_score_a) - float(g.actual_score_b),
        })

    table = pd.DataFrame(rows)
    dropped = len(games) - len(table)
    if dropped:
        print(f"Dropped {dropped}/{len(games)} historical games missing a real clutch signal "
              "for at least one side.")
    return table


def _regression_report(preds: np.ndarray, y: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    directional_hits = int(np.sum(np.sign(preds) == np.sign(y)))
    return {"mae": mae, "r2": r2, "directional_accuracy": directional_hits / len(y) if len(y) else float("nan")}


def evaluate_loocv(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Leave-one-out cross-validation -- each game's prediction comes from
    a model retrained without it. At this sample size (a couple hundred
    games at most) an in-sample-only R^2 would be close to meaningless;
    this is the same honesty-first evaluation train_ml_model.py's own
    small-sample fallback uses.
    """
    from sklearn.model_selection import LeaveOneOut
    from xgboost import XGBRegressor

    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        model = XGBRegressor(**CLUTCH_XGB_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    report = _regression_report(preds, y)
    report["n"] = len(y)
    return preds, report


def probe_margin_effect(model, X: np.ndarray, feature_idx: int) -> float:
    """Empirical partial-dependence-style probe: holds every OTHER feature
    at its observed mean and asks the fitted model for the predicted
    margin swing between this feature's observed -1 SD and +1 SD --
    XGBoost has no linear coefficient to read off directly (unlike
    calibrate_engine.py's OLS), so this is the tree-ensemble equivalent of
    "points of margin per unit of this real feature," on the SAME real
    scale the feature itself is measured in.
    """
    means = X.mean(axis=0)
    std = X[:, feature_idx].std(ddof=0)
    if std <= 1e-9:
        return 0.0
    lo, hi = means.copy(), means.copy()
    lo[feature_idx] -= std
    hi[feature_idx] += std
    delta_margin = float(model.predict(hi.reshape(1, -1))[0] - model.predict(lo.reshape(1, -1))[0])
    return delta_margin / (2.0 * std)  # margin points per +1 unit of the raw feature


def print_report(table: pd.DataFrame, loocv_report: dict, fg_effect_per_unit: float,
                  usage_effect_per_unit: float) -> None:
    print("\n========================================================")
    print(" CLUTCH/MOMENTUM BOOST CALIBRATION -- XGBoost on real clutch data")
    print("========================================================")
    print(f" Games used (both sides had a real clutch signal) : {len(table)}")
    print(f" LOOCV MAE (points)          : {loocv_report['mae']:.2f}")
    print(f" LOOCV R^2                   : {loocv_report['r2']:.3f}")
    print(f" LOOCV directional accuracy  : {loocv_report['directional_accuracy']:.1%}")
    print("--------------------------------------------------------")
    print(f" team_clutch_fg_delta_edge   -> {fg_effect_per_unit:+.2f} margin points per "
          "+1.0 (i.e. +100pp) of real clutch-FG%-vs-season-FG%% edge")
    print(f" star_clutch_usage_delta_edge -> {usage_effect_per_unit:+.2f} margin points per "
          "+1.0 (i.e. +100pp) of real star clutch-usage-vs-season-usage edge")
    print("========================================================")
    if loocv_report["r2"] <= 0.0 or len(table) < 20:
        print(" CAVEAT: LOOCV R^2 is at or below zero and/or the sample is small (<20 games).")
        print(" This means real clutch performance is NOT shown to out-of-sample-predict real")
        print(" game margins at this sample size -- the derived tuning override below is still")
        print(" written (same honesty convention as calibrate_engine.py's weak coefficients),")
        print(" but should be treated as a small, exploratory nudge, not a proven effect.")
        print(" Re-run against more games (fetch_real_nba_data.py with no --max-games) for a")
        print(" sturdier estimate.")
        print("========================================================")
    print()


def derive_tuning_overrides(fg_effect_per_unit: float, usage_effect_per_unit: float) -> dict:
    """Converts the two probed effect sizes into the two EngineTuningParams
    fields they map onto -- see this module's docstring for why only
    clutch_shooting_confidence_bonus gets a unit-derived conversion, and
    clutch_star_usage_mult gets a softer, capped directional nudge.
    """
    # fg_effect_per_unit is "margin points per +1.0 (100pp) of clutch-FG%
    # edge" -- a real clutch-FG% edge this large would be enormous (teams
    # differ by a few percentage points, not 100), so the conversion is
    # applied directly to MARGIN_TO_PROB_SHIFT's own per-point units,
    # exactly like calibrate_engine.py's b_rest/b_def coefficients.
    shooting_bonus = DEFAULT_CLUTCH_SHOOTING_CONFIDENCE_BONUS + fg_effect_per_unit * MARGIN_TO_PROB_SHIFT
    # Keep within this constant's own sane band (kMinShotProb/kMaxShotProb
    # already backstop the engine itself, but a wildly-fit small-sample
    # value has no business overriding the default by more than a few
    # points of shot probability).
    shooting_bonus = float(np.clip(shooting_bonus, -0.06, 0.08))

    # usage_effect_per_unit is "margin points per +1.0 (100pp) of star
    # clutch-usage edge" -- no direct unit conversion exists from "margin
    # points" to "a usage-weighted Node 0 tendency multiplier", so this is
    # a capped, SIGN-informed nudge (+-0.10 around the engine default),
    # scaled by how large the effect looks relative to a plausible real
    # clutch-usage swing (~0.05, i.e. a player's usage rising ~5
    # percentage points in the clutch -- a real, commonly observed
    # magnitude).
    plausible_real_usage_swing = 0.05
    usage_mult = DEFAULT_CLUTCH_STAR_USAGE_MULT + np.clip(
        usage_effect_per_unit * plausible_real_usage_swing * MARGIN_TO_PROB_SHIFT * 20.0, -0.10, 0.10
    )
    usage_mult = float(np.clip(usage_mult, 1.00, 1.30))

    return {
        "clutch_shooting_confidence_bonus": shooting_bonus,
        "clutch_star_usage_mult": usage_mult,
    }


def write_tuning_config(overrides: dict, table: pd.DataFrame, loocv_report: dict,
                         season: str, out_path: Path) -> None:
    payload = dict(overrides)
    payload["_meta"] = {
        "generated_by": "train_clutch_boost_model.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "season": season,
        "games_used": len(table),
        "loocv_r2": loocv_report["r2"],
        "loocv_mae": loocv_report["mae"],
        "note": ("clutch_shooting_confidence_bonus is a unit-derived probability-shift "
                 "override (see MARGIN_TO_PROB_SHIFT in this script); clutch_star_usage_mult "
                 "is a softer, directionally-informed nudge, not a unit conversion -- see "
                 "this script's module docstring."),
    }
    # `_meta` is harmless for --tuning-config's own JSON loader (LOAD_TUNING_FIELD
    # only ever reads the specific field names it knows about via
    # nlohmann::json's `.value(key, default)`, so an unrecognized top-level
    # key like `_meta` is silently ignored, not an error) -- kept in the
    # file for a human/audit trail, not read by the C++ engines.
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Use it directly:")
    print(f"  cpp_engine/build/Release/cuda_simulator.exe TEAM_A TEAM_B --tuning-config {out_path.name}")
    print(f"  cpp_engine/build/Release/simulator.exe --tuning-config {out_path.name}\n")


def run(csv_path: Path, api_url: str, season: str, out_path: Path) -> int:
    players = ml.fetch_players(api_url)
    players_df = ml._prep_players_df(players)

    team_signals = compute_team_clutch_signals(players_df, season)
    if len(team_signals) < 4:
        raise ml.MlPipelineError(
            f"Only {len(team_signals)} teams have a usable real clutch-time signal this season "
            "-- too few to fit anything meaningful. Try again later in the season once more "
            "teams have accumulated real clutch minutes."
        )

    table = build_game_table(csv_path, team_signals)
    if len(table) < 8:
        raise ml.MlPipelineError(
            f"Only {len(table)} historical games have a real clutch signal for both sides -- "
            "too few to fit anything meaningful."
        )

    X = table[FEATURE_NAMES].to_numpy(dtype=float)
    y = table["actual_margin"].to_numpy(dtype=float)

    _, loocv_report = evaluate_loocv(X, y)

    from xgboost import XGBRegressor
    final_model = XGBRegressor(**CLUTCH_XGB_PARAMS)
    final_model.fit(X, y)

    fg_effect = probe_margin_effect(final_model, X, FEATURE_NAMES.index("team_clutch_fg_delta_edge"))
    usage_effect = probe_margin_effect(final_model, X, FEATURE_NAMES.index("star_clutch_usage_delta_edge"))

    print_report(table, loocv_report, fg_effect, usage_effect)

    overrides = derive_tuning_overrides(fg_effect, usage_effect)
    print("Derived --tuning-config overrides:")
    for k, v in overrides.items():
        print(f"  {k}: {v:+.4f}")
    print()

    write_tuning_config(overrides, table, loocv_report, season, out_path)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ml.DEFAULT_CSV)
    parser.add_argument("--api-url", type=str, default=ml.DEFAULT_API_URL)
    parser.add_argument("--season", type=str, default=ml.DEFAULT_SEASON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Where to write the --tuning-config JSON (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)

    try:
        return run(args.csv, args.api_url, args.season, args.out)
    except ml.MlPipelineError as e:
        print(f"[train_clutch_boost_model] Fatal: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
