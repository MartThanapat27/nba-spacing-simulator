#!/usr/bin/env python3
"""Head-to-Head (H2H) Tactical Counter-Strategies -- a preprocessing utility
that analyzes real style-matchup metrics (and any literal historical
head-to-head meetings) between two specific teams, and writes a pair-
specific `--tuning-config` JSON (`EngineTuningParams` overrides) so the
C++/CUDA engines simulate localized tactical friction for THAT exact
rivalry instead of generic league-baseline stats.

Three real, interpretable style-matchup signals
------------------------------------------------
  1. **Pace factor** -- these two specific teams' real blended PACE
     (possessions/48, nba_api `leaguedashteamstats` Advanced) vs the
     league-average pace both engines are calibrated to
     (`kPossessionsPerTeam`=100). A faster blended pace shortens the
     neutral-state possession-length range (`min_possession_seconds`/
     `max_possession_seconds`); a slower one lengthens it.
  2. **Shooting-efficiency (environment) adjustment** -- the average of
     both teams' real OPP_FG3_PCT (3PT% allowed, nba_api `Opponent`
     measure type) relative to the LEAGUE-AVERAGE OPP_FG3_PCT (computed
     from this same fetch, not a hardcoded assumption). When these two
     teams' defenses run leakier than average against the 3, this pairing
     tends to be a more perimeter-friendly environment than the generic
     baseline, nudged in via `league_avg_shot_prob` (the Node 2
     compression's own center -- see cpp_engine's comment block above that
     field).
  3. **Star-neutralization (interior physicality) adjustment** -- the
     average of both teams' real best rim protector (max blocks/game
     across the top-8-by-minutes rotation, mirroring `rim_protection_best`
     exactly) relative to the league-average figure. When this specific
     pairing brings above-average shot-blocking on both ends, interior
     possessions face more real friction than the generic baseline,
     nudged in via `rim_protect_suppression_weight`.

Literal head-to-head games, when they exist
---------------------------------------------
`historical_games.csv` is checked for any real meetings between these
exact two teams (either home/away order). With a 120-game single-season
sample this is usually 0-2 games -- far too few to fit anything on its
own -- so it is reported as a diagnostic and, when at least one real
meeting exists, blended in as a SMALL (capped, sample-size-scaled)
additional nudge to the pace factor specifically (the one signal a single
box score's own real combined score can speak to directly), rather than
silently ignored.

Which EngineTuningParams fields this touches, and which it deliberately
doesn't
--------------------------------------------------------------------------
Only fields with IDENTICAL default values on both cpp_engine/main.cpp (CPU)
and cpp_engine/cuda_simulator.cuh (GPU) are targeted here
(`min_possession_seconds`/`max_possession_seconds`, `league_avg_shot_prob`,
`rim_protect_suppression_weight`). `paint_finish_fg_pct_bonus`/
`open_shot_bonus_multiplier`/`contested_iso_fg_pct_penalty` are
DELIBERATELY different between the two engines (see cuda_simulator.cu's
own comment block on those fields -- the GPU kernel's floor-spacing
gravity has no fatigue/substitution model, so it runs systematically
higher than the CPU's fatigue-aware version), so writing one shared
absolute value into a `--tuning-config` file either engine might load
would silently erase that already-established, documented CPU/GPU
calibration difference. Every field this script DOES touch has no such
landmine.

Because EngineTuningParams fields are engine-wide constants applied
symmetrically to whichever side is on offense/defense each possession (not
per-team), a pair-specific modifier here means "how should THIS MATCHUP's
overall tactical texture shift" (pace, contest tightness, interior
physicality) -- not "boost only Team A" -- an honest scoping given the
engine's existing architecture, not a limitation this script papers over.

Usage:
    python compute_h2h_tactics.py SAS OKC
    python compute_h2h_tactics.py GSW SAS --out h2h_gsw_sas.json
    cpp_engine/build/Release/cuda_simulator.exe SAS OKC --tuning-config h2h_sas_okc.json
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

# Engine defaults (EngineTuningParams in cpp_engine/main.cpp /
# cuda_simulator.cuh) this script may override -- see module docstring for
# why ONLY these three (identical CPU/GPU defaults).
DEFAULT_MIN_POSSESSION_SECONDS = 11
DEFAULT_MAX_POSSESSION_SECONDS = 18
DEFAULT_LEAGUE_AVG_SHOT_PROB = 0.42
DEFAULT_RIM_PROTECT_SUPPRESSION_WEIGHT = 0.09

# Real possession-count target both engines are calibrated to (see
# kPossessionsPerTeam in cuda_simulator.cu/main.cpp).
LEAGUE_AVG_PACE = 100.0

MAX_ROTATION_PLAYERS = 8  # matches build_gpu_roster()'s rotation cap in cuda_main.cpp

# Caps on how far each real signal is allowed to move its engine field --
# these are pair-specific TEXTURE nudges layered on top of the engine's
# own already-verified calibration, not a wholesale re-tune.
MAX_POSSESSION_SHIFT_SECONDS = 3.0
MAX_SHOT_PROB_SHIFT = 0.02
MAX_RIM_SUPPRESSION_SHIFT = 0.02


def fetch_team_pace_and_opp_fg3(season: str) -> tuple[dict[str, float], dict[str, float]]:
    """REAL data: every team's season-to-date real PACE (Advanced measure
    type) and real OPP_FG3_PCT -- 3PT% allowed (Opponent measure type),
    via nba_api's leaguedashteamstats. Two separate real endpoints because
    nba_api doesn't expose both in one measure type.
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    id_to_abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame", timeout=30,
    ).get_data_frames()[0]
    adv["abbr"] = adv["TEAM_ID"].map(id_to_abbr)
    pace = dict(zip(adv["abbr"], adv["PACE"].astype(float)))

    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star="Regular Season",
        measure_type_detailed_defense="Opponent", per_mode_detailed="PerGame", timeout=30,
    ).get_data_frames()[0]
    opp["abbr"] = opp["TEAM_ID"].map(id_to_abbr)
    opp_fg3_pct = dict(zip(opp["abbr"], opp["OPP_FG3_PCT"].astype(float)))

    return pace, opp_fg3_pct


def compute_team_rim_protection(players_df: pd.DataFrame, team_abbr: str) -> float:
    """REAL data: this team's best real shot-blocker's blocks/game across
    the top-`MAX_ROTATION_PLAYERS`-by-minutes rotation -- mirrors
    GPURoster::rim_protection_best / Team::get_rim_protection_best()
    exactly (max, not mean -- rim protection is disproportionately driven
    by one elite shot-blocker).
    """
    rotation = players_df[players_df["team_abbreviation"] == team_abbr].sort_values(
        "min", ascending=False).head(MAX_ROTATION_PLAYERS)
    if rotation.empty:
        return 0.5  # engine's own neutral default
    return float(rotation["blk_per_gm"].max())


def find_h2h_games(csv_path: Path, team_a: str, team_b: str) -> pd.DataFrame:
    """REAL data: any literal historical meetings between these exact two
    teams (either home/away order) in historical_games.csv.
    """
    games = pd.read_csv(csv_path)
    mask = (
        ((games["team_a"] == team_a) & (games["team_b"] == team_b))
        | ((games["team_a"] == team_b) & (games["team_b"] == team_a))
    )
    return games[mask]


def compute_h2h_tactics(team_a: str, team_b: str, api_url: str, season: str, csv_path: Path,
                         verbose: bool = True) -> dict:
    players = ml.fetch_players(api_url)
    players_df = ml._prep_players_df(players)
    # blk isn't in _prep_players_df's narrower column set -- pull it
    # straight from the raw payload (same "already real, no fabrication"
    # convention as everything else this project derives from /api/players).
    players_df["blk_raw"] = [p.get("blk", 0.0) for p in players]
    players_df["gp_raw"] = [p.get("gp", 1.0) or 1.0 for p in players]
    players_df["blk_per_gm"] = players_df["blk_raw"] / players_df["gp_raw"]

    if verbose:
        print(f"Fetching {season} real team pace and opponent-3PT% (nba_api leaguedashteamstats)...")
    pace_by_team, opp_fg3_by_team = fetch_team_pace_and_opp_fg3(season)

    for abbr in (team_a, team_b):
        if abbr not in pace_by_team:
            raise ml.MlPipelineError(f"No real season stats found for team '{abbr}' this season.")

    league_avg_opp_fg3 = float(np.mean(list(opp_fg3_by_team.values())))
    league_avg_rim_protection = float(np.mean([
        compute_team_rim_protection(players_df, abbr) for abbr in pace_by_team if abbr in players_df["team_abbreviation"].values
    ]))

    pace_a, pace_b = pace_by_team[team_a], pace_by_team[team_b]
    blended_pace = (pace_a + pace_b) / 2.0

    opp_fg3_a = opp_fg3_by_team.get(team_a, league_avg_opp_fg3)
    opp_fg3_b = opp_fg3_by_team.get(team_b, league_avg_opp_fg3)
    blended_opp_fg3 = (opp_fg3_a + opp_fg3_b) / 2.0

    rim_a = compute_team_rim_protection(players_df, team_a)
    rim_b = compute_team_rim_protection(players_df, team_b)
    blended_rim_protection = (rim_a + rim_b) / 2.0

    h2h_games = find_h2h_games(csv_path, team_a, team_b)
    h2h_avg_combined_score: Optional[float] = None
    if not h2h_games.empty:
        h2h_avg_combined_score = float((h2h_games["actual_score_a"] + h2h_games["actual_score_b"]).mean())

    if verbose:
        print(f"\nReal style signals for {team_a} vs {team_b}:")
        print(f"  Pace (a/b/blended)              : {pace_a:.1f} / {pace_b:.1f} / {blended_pace:.1f} "
              f"(league avg target: {LEAGUE_AVG_PACE:.1f})")
        print(f"  Opp 3PT% allowed (a/b/blended)   : {opp_fg3_a:.3f} / {opp_fg3_b:.3f} / {blended_opp_fg3:.3f} "
              f"(league avg: {league_avg_opp_fg3:.3f})")
        print(f"  Best rim protector blk/gm (a/b)  : {rim_a:.2f} / {rim_b:.2f} "
              f"(pairing avg: {blended_rim_protection:.2f}, league avg: {league_avg_rim_protection:.2f})")
        if h2h_games.empty:
            print(f"  Literal head-to-head games found : 0 (in this {len(pd.read_csv(csv_path))}-game sample -- "
                  "style-matchup signals above are this script's whole basis)")
        else:
            print(f"  Literal head-to-head games found : {len(h2h_games)}, avg real combined score "
                  f"{h2h_avg_combined_score:.1f}")

    return {
        "team_a": team_a, "team_b": team_b,
        "pace_a": pace_a, "pace_b": pace_b, "blended_pace": blended_pace,
        "opp_fg3_a": opp_fg3_a, "opp_fg3_b": opp_fg3_b, "blended_opp_fg3": blended_opp_fg3,
        "league_avg_opp_fg3": league_avg_opp_fg3,
        "rim_a": rim_a, "rim_b": rim_b, "blended_rim_protection": blended_rim_protection,
        "league_avg_rim_protection": league_avg_rim_protection,
        "n_h2h_games": len(h2h_games),
        "h2h_avg_combined_score": h2h_avg_combined_score,
    }


def derive_tuning_overrides(signals: dict) -> dict:
    """Converts the three real style-matchup signals into the three
    EngineTuningParams fields they map onto -- see this module's docstring
    for why exactly these three fields, and why every shift is capped
    (a pair-specific TEXTURE nudge, not a wholesale re-tune).
    """
    # 1. Pace factor -- a faster real blended pace than the league-average
    # target SHORTENS the neutral possession-length range (more real
    # possessions/game); a slower one lengthens it. Scaled linearly and
    # capped at MAX_POSSESSION_SHIFT_SECONDS so an unusually extreme real
    # pace outlier can't blow the range out to something implausible.
    pace_ratio = LEAGUE_AVG_PACE / signals["blended_pace"] if signals["blended_pace"] > 0 else 1.0
    center = (DEFAULT_MIN_POSSESSION_SECONDS + DEFAULT_MAX_POSSESSION_SECONDS) / 2.0
    half_width = (DEFAULT_MAX_POSSESSION_SECONDS - DEFAULT_MIN_POSSESSION_SECONDS) / 2.0
    center_shift = np.clip((pace_ratio - 1.0) * center, -MAX_POSSESSION_SHIFT_SECONDS, MAX_POSSESSION_SHIFT_SECONDS)

    # Literal head-to-head evidence, when any exists: a real combined score
    # notably above/below a league-average combined-score baseline
    # (~2 * 110 = 220, this engine's own ~105-115/team target) nudges the
    # pace shift further in the SAME direction, scaled down hard (capped at
    # a THIRD of the style-based cap) since 1-2 real games is a tiny,
    # high-variance sample on its own -- corroborating evidence, not a
    # primary signal.
    if signals["h2h_avg_combined_score"] is not None:
        league_avg_combined_score = 220.0
        h2h_pace_hint = (signals["h2h_avg_combined_score"] - league_avg_combined_score) / league_avg_combined_score
        h2h_shift = np.clip(-h2h_pace_hint * center, -MAX_POSSESSION_SHIFT_SECONDS / 3.0, MAX_POSSESSION_SHIFT_SECONDS / 3.0)
        center_shift = np.clip(center_shift + h2h_shift, -MAX_POSSESSION_SHIFT_SECONDS, MAX_POSSESSION_SHIFT_SECONDS)

    new_center = center + center_shift
    min_possession_seconds = int(round(np.clip(new_center - half_width, 6, 30)))
    max_possession_seconds = int(round(np.clip(new_center + half_width, min_possession_seconds + 2, 34)))

    # 2. Shooting-efficiency (environment) adjustment -- this pairing's
    # blended real OPP_FG3_PCT relative to the REAL league average (not an
    # assumed constant) shifts the Node 2 compression center
    # (league_avg_shot_prob): a leakier-than-average perimeter environment
    # for this specific pairing nudges it up (shots run hotter before
    # compression engages); a stingier one nudges it down.
    opp_fg3_edge = signals["blended_opp_fg3"] - signals["league_avg_opp_fg3"]
    # A real OPP_FG3_PCT edge of ~0.02 (2 percentage points) is already a
    # meaningfully leaky/stingy defense at the team level -- scaled so that
    # magnitude alone reaches the cap.
    league_avg_shot_prob = DEFAULT_LEAGUE_AVG_SHOT_PROB + np.clip(
        opp_fg3_edge * (MAX_SHOT_PROB_SHIFT / 0.02), -MAX_SHOT_PROB_SHIFT, MAX_SHOT_PROB_SHIFT)

    # 3. Star-neutralization (interior physicality) adjustment -- this
    # pairing's blended real best-rim-protector figure relative to the
    # REAL league average shifts rim_protect_suppression_weight: more
    # real shot-blocking on both ends of this specific pairing means
    # interior possessions face more real friction than the generic
    # baseline.
    rim_edge = signals["blended_rim_protection"] - signals["league_avg_rim_protection"]
    # A real rim-protection edge of ~1.0 blk/gm above league average is
    # already a genuinely elite-anchored pairing -- scaled so that
    # magnitude alone reaches the cap.
    rim_protect_suppression_weight = DEFAULT_RIM_PROTECT_SUPPRESSION_WEIGHT + np.clip(
        rim_edge * MAX_RIM_SUPPRESSION_SHIFT, -MAX_RIM_SUPPRESSION_SHIFT, MAX_RIM_SUPPRESSION_SHIFT)

    return {
        "min_possession_seconds": min_possession_seconds,
        "max_possession_seconds": max_possession_seconds,
        "league_avg_shot_prob": float(league_avg_shot_prob),
        "rim_protect_suppression_weight": float(rim_protect_suppression_weight),
    }


def write_tuning_config(overrides: dict, signals: dict, out_path: Path) -> None:
    payload = dict(overrides)
    payload["_meta"] = {
        "generated_by": "compute_h2h_tactics.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "team_a": signals["team_a"], "team_b": signals["team_b"],
        "blended_pace": signals["blended_pace"],
        "blended_opp_fg3_pct": signals["blended_opp_fg3"], "league_avg_opp_fg3_pct": signals["league_avg_opp_fg3"],
        "blended_rim_protection": signals["blended_rim_protection"],
        "league_avg_rim_protection": signals["league_avg_rim_protection"],
        "n_h2h_games": signals["n_h2h_games"], "h2h_avg_combined_score": signals["h2h_avg_combined_score"],
        "note": ("Pair-specific tactical texture nudges (pace/shooting-environment/interior-physicality), "
                 "layered on top of the engine's own default calibration -- see this script's module "
                 "docstring for exactly which EngineTuningParams fields are safe to override this way."),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("Use it directly:")
    print(f"  cpp_engine/build/Release/cuda_simulator.exe {signals['team_a']} {signals['team_b']} "
          f"--tuning-config {out_path.name}")
    print(f"  cpp_engine/build/Release/simulator.exe {signals['team_a']} {signals['team_b']} "
          f"--tuning-config {out_path.name}\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("team_a", type=str)
    parser.add_argument("team_b", type=str)
    parser.add_argument("--api-url", type=str, default=ml.DEFAULT_API_URL)
    parser.add_argument("--season", type=str, default=ml.DEFAULT_SEASON)
    parser.add_argument("--csv", type=Path, default=ml.DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=None,
                         help="Where to write the --tuning-config JSON "
                              "(default: h2h_tactics_<TEAM_A>_<TEAM_B>.json)")
    args = parser.parse_args(argv)

    team_a, team_b = args.team_a.upper(), args.team_b.upper()
    out_path = args.out or SCRIPT_DIR / f"h2h_tactics_{team_a}_{team_b}.json"

    try:
        signals = compute_h2h_tactics(team_a, team_b, args.api_url, args.season, args.csv)
        overrides = derive_tuning_overrides(signals)
    except ml.MlPipelineError as e:
        print(f"[compute_h2h_tactics] Fatal: {e}")
        return 1

    print("\nDerived --tuning-config overrides:")
    for k, v in overrides.items():
        print(f"  {k}: {v}")

    write_tuning_config(overrides, signals, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
