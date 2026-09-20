#!/usr/bin/env python3
"""Fits the C++/CUDA engine's intrinsic defensive-resistance, fatigue, and
home-court coefficients from real historical data, and generates
`cpp_engine/calibrated_constants.h` -- the single source of truth both
`cuda_simulator.cu` and `main.cpp` `#include` at compile time. There is no
manual copy-paste step: running this script is the only way those constants
change, and both engines read the exact same generated values.

Why not just read these off XGBoost feature importances?
----------------------------------------------------------
`train_ml_model.py`'s gain-based XGBoost importances answer "how much did
splits on this feature reduce loss" -- a useful *ranking* signal, but not a
coefficient with real-world units. They can't tell you "shooting probability
drops by X percentage points per back-to-back," which is what a possession
simulator needs to compute an effect intrinsically. For that, this script
fits plain **ordinary least squares** (a genuinely interpretable model: each
coefficient is directly "points of margin per unit of X, holding the others
fixed") on real, already-engineered features (real rest days, real
defensive ratings) that reuse `train_ml_model.py`'s exact feature pipeline
(and its box-score cache) so nothing here can silently drift from that
script's own data.

What this fits
---------------
    actual_margin ~ intercept + b_rest * rest_advantage + b_def * net_def_edge

  * `rest_advantage` = team_a_rest_days - team_b_rest_days
  * `net_def_edge`   = team_b.def_rating - team_a.def_rating (positive =>
                        team_b defends worse than team_a, favoring team_a
                        on both ends of the floor)

Where does home-court advantage come from, if it's not a regressor?
----------------------------------------------------------------------
`fetch_real_nba_data.py` always orders team_a as the HOME team -- every row
in `historical_games.csv` has team_a at home. That means "is team_a home"
has **zero within-dataset variation**: there is no "team_a away" case to
compare against, so a differential regressor for it would be perfectly
collinear with nothing and simply inestimable. What *is* estimable, and
exactly captures the same quantity, is the regression's **intercept**: with
rest_advantage and net_def_edge held at zero, the model's predicted margin
for team_a (home) vs. team_b (away) *is* the fitted home-court effect, by
construction of how the dataset was built. So the intercept is reported and
converted exactly like the other two coefficients -- not a special case,
just the correct estimator given this dataset's structure. (If a future
dataset mixes home/away order, add an explicit home_indicator regressor and
drop this reasoning.)

Every coefficient (including the intercept) is reported with its standard
error and t-statistic, because with ~100-200 games a point estimate alone
isn't trustworthy. A coefficient with |t| < 2 is written to the generated
header at its fitted magnitude anyway (not zeroed out or replaced with a
guess) but flagged clearly as not statistically distinguishable from zero,
both in this script's report and in the header's own comments -- so nothing
downstream can mistake "the data says ~0" for "untested."

Converting to the kernel's probability space
----------------------------------------------
`cuda_simulator.cu`'s kernel operates on a per-possession shot-probability
shift, not points of game margin. The conversion is physically motivated: a
made shot is worth ~2.3 points on average (a blend of 2PT/3PT), there are
~100 possessions/team/game, and a symmetric +delta/-delta shift across both
teams doubles the margin swing, so
    delta_p = margin_coefficient / (2 * avg_points_per_make * possessions_per_team)
This is the same derivation the engine already uses for its external
`--ml-margin` flag (see `kMlMarginToProbShift` in `cuda_simulator.cu`), reused
here so intrinsic and external calibrations agree on one possession model.

No double-counting: what train_ml_model.py's XGBoost model does NOT see
--------------------------------------------------------------------------
Defensive rating, rest days, and home court are now handled *intrinsically*
by the C++ engine (this script's whole point). `train_ml_model.py`'s
external `--ml-margin` model has had those exact features removed from its
own input set for this reason -- if it kept them, its predicted margin would
double-count the same real-world signal the engine now already applies on
its own, which is precisely the "double-counted external margin" this
refactor eliminates. See train_ml_model.py's module docstring.

Usage:
    python calibrate_engine.py
    python calibrate_engine.py --csv historical_games.csv --season 2024-25
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import train_ml_model as ml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HEADER_PATH = SCRIPT_DIR / "cpp_engine" / "calibrated_constants.h"

# Must match cpp_engine/cuda_simulator.cu's kAvgPointsPerMake / kPossessionsPerTeam
# exactly -- see that file's kMlMarginToProbShift derivation comment, reused
# here so both the kernel's own external-margin conversion and this script's
# intrinsic-coefficient conversion agree on the same possession model.
AVG_POINTS_PER_MAKE = 2.3
POSSESSIONS_PER_TEAM = 100
MARGIN_TO_PROB_SHIFT = 1.0 / (2.0 * AVG_POINTS_PER_MAKE * POSSESSIONS_PER_TEAM)

REGRESSORS = ["rest_advantage", "net_def_edge"]  # + implicit "intercept" = home-court, see module docstring


def fit_ols_with_stats(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Plain OLS via the normal equations, with standard errors and t-stats
    from the usual sampling theory. `X` should NOT include an intercept
    column -- one is added here, and (per this module's docstring) IS the
    home-court coefficient given this dataset's all-team_a-is-home structure.
    """
    n, k = X.shape
    X_design = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    resid = y - X_design @ beta
    rss = float(np.sum(resid ** 2))
    dof = n - (k + 1)
    if dof <= 0:
        raise ml.MlPipelineError(f"Not enough games ({n}) to fit {k} coefficients + intercept.")
    sigma2 = rss / dof

    xtx_inv = np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_stats = beta / se

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    all_names = ["intercept (home-court)"] + names
    coefficients = {
        name: {"beta": float(b), "se": float(s), "t_stat": float(t)}
        for name, b, s, t in zip(all_names, beta, se, t_stats)
    }
    return {"coefficients": coefficients, "r2": r2, "n": n, "dof": dof, "rss": rss}


def print_report(fit: dict) -> None:
    print("\n========================================================")
    print(" ENGINE PARAMETER CALIBRATION -- OLS on real historical data")
    print("========================================================")
    print(f" Games used   : {fit['n']}")
    print(f" R^2          : {fit['r2']:.3f}")
    print("--------------------------------------------------------")
    print(f"{'Term':<26}{'Coefficient':>14}{'Std. Error':>14}{'t-stat':>10}  Significant?")
    for name in fit["coefficients"]:
        c = fit["coefficients"][name]
        sig = "yes (|t|>2)" if abs(c["t_stat"]) > 2.0 else "NO -- weak/no evidence"
        print(f"{name:<26}{c['beta']:>+14.4f}{c['se']:>14.4f}{c['t_stat']:>10.2f}  {sig}")
    print("========================================================")
    print(" CAVEAT: n is small (a couple hundred games at most). A coefficient")
    print(" flagged 'NO' is not distinguishable from zero at this sample size --")
    print(" it is still written to calibrated_constants.h at its fitted, honest")
    print(" magnitude (which may be near zero) rather than replaced with a guess,")
    print(" but should not be read as a proven effect. Re-run against more games")
    print(" (fetch_real_nba_data.py with no --max-games) for a sturdier estimate.")
    print("========================================================\n")


def build_regression_table(csv_path: Path, api_url: str, season: str, ml_dir: Path):
    players = ml.fetch_players(api_url)
    players_df = ml._prep_players_df(players)

    print(f"Fetching {season} team defensive ratings from nba_api...")
    def_ratings = ml.fetch_team_defensive_ratings(season)
    team_features, star_names = ml.compute_team_features(players_df, def_ratings)

    import json
    star_cache_path = ml_dir / "star_availability_cache.json"
    star_cache = {}
    if star_cache_path.is_file():
        with star_cache_path.open(encoding="utf-8") as f:
            star_cache = json.load(f)

    print("Building regression table (reusing train_ml_model.py's feature pipeline "
          "and cached box-score star-availability lookups)...")
    table = ml.load_training_table(csv_path, players_df, team_features, star_names, def_ratings,
                                    star_cache, verbose=True)

    ml_dir.mkdir(parents=True, exist_ok=True)
    with star_cache_path.open("w", encoding="utf-8") as f:
        json.dump(star_cache, f, indent=2)

    return table


def calibrate(csv_path: Path, api_url: str, season: str, ml_dir: Path) -> dict:
    table = build_regression_table(csv_path, api_url, season, ml_dir)

    rest_advantage = table["rest_advantage"].to_numpy()
    net_def_edge = table["net_def_edge"].to_numpy()
    y = table["actual_margin"].to_numpy()

    X = np.column_stack([rest_advantage, net_def_edge])
    fit = fit_ols_with_stats(X, y, REGRESSORS)
    print_report(fit)

    b_home = fit["coefficients"]["intercept (home-court)"]["beta"]
    b_rest = fit["coefficients"]["rest_advantage"]["beta"]
    b_def = fit["coefficients"]["net_def_edge"]["beta"]

    # rest_advantage and net_def_edge are Team A - Team B differentials, so
    # their fitted coefficients already represent the full symmetric margin
    # swing; the intercept is a one-sided home-court bonus for team_a, so it
    # gets the SAME conversion (it's still "points of margin", just not a
    # differential -- kMlMarginToProbShift's derivation doesn't care which).
    prob_per_rest_day = b_rest * MARGIN_TO_PROB_SHIFT
    prob_per_def_point = b_def * MARGIN_TO_PROB_SHIFT
    prob_home_court = b_home * MARGIN_TO_PROB_SHIFT

    print("========================================================")
    print(" CONVERTED TO KERNEL PROBABILITY-SPACE CONSTANTS")
    print(" (points-of-margin coefficient x kMlMarginToProbShift)")
    print("========================================================")
    print(f" kDefResistanceProbPerRating : {prob_per_def_point:+.6f}  (shot-prob shift per net_def_edge point)")
    print(f" kFatigueProbPerRestDay      : {prob_per_rest_day:+.6f}  (shot-prob shift per day of rest ADVANTAGE;")
    print("                                a back-to-back is -1 unit of this -- see C++ usage)")
    print(f" kHomeCourtProbShift         : {prob_home_court:+.6f}  (shot-prob shift for the home team)")
    print("========================================================\n")

    return {
        "fit": fit,
        "prob_per_def_point": prob_per_def_point,
        "prob_per_rest_day": prob_per_rest_day,
        "prob_home_court": prob_home_court,
    }


HEADER_TEMPLATE = """\
#pragma once

// ============================================================================
// AUTO-GENERATED by calibrate_engine.py -- DO NOT EDIT BY HAND.
// Regenerate with: python calibrate_engine.py
//
// Generated : {timestamp}
// Dataset   : {csv_name} ({n} real games)
// Model     : actual_margin ~ intercept + b_rest*rest_advantage + b_def*net_def_edge
//             (OLS; intercept = home-court effect -- see this script's module
//             docstring for why home-court is estimated via the intercept
//             rather than a regressor, given this dataset's all-team_a-home
//             structure)
// R^2       : {r2:.3f}
//
//   Term                       Coefficient   Std.Error   t-stat   Significant?
//   intercept (home-court)     {home_beta:>+11.4f} {home_se:>11.4f} {home_t:>8.2f}   {home_sig}
//   rest_advantage             {rest_beta:>+11.4f} {rest_se:>11.4f} {rest_t:>8.2f}   {rest_sig}
//   net_def_edge               {def_beta:>+11.4f} {def_se:>11.4f} {def_t:>8.2f}   {def_sig}
//
// A coefficient marked "NO" above is not statistically distinguishable from
// zero at this sample size. It is still included below at its fitted,
// honest magnitude rather than zeroed out or replaced with a guess -- but
// should not be read as a proven real-world effect. Re-run
// `calibrate_engine.py` against more games for a sturdier estimate.
//
// Included by both cpp_engine/cuda_simulator.cu and cpp_engine/main.cpp, so
// the GPU batch engine and CPU narrative engine always share one calibration.
// ============================================================================

namespace engine_calibration {{

// Shot-probability shift per net_def_edge point (team_b.def_rating -
// team_a.def_rating). Always applied intrinsically -- every roster carries
// its own real def_rating (see GPURoster::def_rating / Team::def_rating),
// no external flag required.
constexpr double kDefResistanceProbPerRating = {def_shift:.8f};

// Shot-probability shift per day of REST ADVANTAGE (fit as the
// rest_advantage regression coefficient, positive = more rest is better).
// A team on a back-to-back (0 rest days) has roughly -1 unit of rest
// advantage relative to a normally-rested opponent, so the engine must
// apply it as -kFatigueProbPerRestDay (a penalty), not add it directly --
// requires the --b2b-a/--b2b-b flag, since "is this team on a
// back-to-back" is game-context information no roster stat can supply.
constexpr double kFatigueProbPerRestDay = {fatigue_shift:.8f};

// Shot-probability shift applied to the home team -- requires the
// --home-a/--home-b flag for the same reason as fatigue above (venue is
// game context, not a roster property).
constexpr double kHomeCourtProbShift = {home_shift:.8f};

}}  // namespace engine_calibration
"""


def format_header(csv_path: Path, calibration: dict) -> str:
    fit = calibration["fit"]
    c = fit["coefficients"]

    def sig(name: str) -> str:
        return "yes" if abs(c[name]["t_stat"]) > 2.0 else "NO"

    return HEADER_TEMPLATE.format(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        csv_name=csv_path.name,
        n=fit["n"],
        r2=fit["r2"],
        home_beta=c["intercept (home-court)"]["beta"], home_se=c["intercept (home-court)"]["se"],
        home_t=c["intercept (home-court)"]["t_stat"], home_sig=sig("intercept (home-court)"),
        rest_beta=c["rest_advantage"]["beta"], rest_se=c["rest_advantage"]["se"],
        rest_t=c["rest_advantage"]["t_stat"], rest_sig=sig("rest_advantage"),
        def_beta=c["net_def_edge"]["beta"], def_se=c["net_def_edge"]["se"],
        def_t=c["net_def_edge"]["t_stat"], def_sig=sig("net_def_edge"),
        def_shift=calibration["prob_per_def_point"],
        fatigue_shift=calibration["prob_per_rest_day"],
        home_shift=calibration["prob_home_court"],
    )


def write_header(csv_path: Path, calibration: dict, header_path: Path) -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(format_header(csv_path, calibration), encoding="utf-8")
    print(f"Wrote {header_path}")
    print("Rebuild the C++ engines to pick up the new constants:")
    print("  cmake --build cpp_engine/build --config Release --target cuda_simulator simulator\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ml.DEFAULT_CSV)
    parser.add_argument("--api-url", type=str, default=ml.DEFAULT_API_URL)
    parser.add_argument("--season", type=str, default=ml.DEFAULT_SEASON)
    parser.add_argument("--out-dir", type=Path, default=ml.DEFAULT_OUT_DIR)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER_PATH,
                         help=f"Where to write the generated C++ header (default: {DEFAULT_HEADER_PATH})")
    args = parser.parse_args(argv)

    try:
        calibration = calibrate(args.csv, args.api_url, args.season, args.out_dir)
        write_header(args.csv, calibration, args.header)
    except ml.MlPipelineError as e:
        print(f"[calibrate_engine] Fatal: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
