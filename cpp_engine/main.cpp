#define NOMINMAX

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <set>
#include <sstream>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include "calibrated_constants.h"

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Engine Tuning Parameters (ML/Optimization-Driven Calibration)
//
// Groups this engine's real tunable decision-tree weights -- Node 1 (Paint
// Openness Score), Node 2 (shot-quality spread + final probability
// compression), and the newer Macro mechanics (Game-to-Game Stochastic
// Variance, Foul Trouble, Momentum/Scoring Runs) -- into ONE struct instead
// of scattered standalone constexpr constants. This is what makes external
// ML/optimization-driven tuning possible: g_tuning is a plain, mutable
// global (NOT constexpr) with the same default values this engine was
// already calibrated to, loaded once at startup and optionally overridden
// by a `--tuning-config <path.json>` file -- a Python optimization script
// (e.g. fitting these weights against real historical match results as a
// loss function) can write out exactly this JSON shape and hand it back in,
// with no source changes required. Any field the JSON omits keeps its
// default. Mirrors cuda_simulator.cu's identically-named struct exactly
// (double vs. float per each file's own existing convention), and
// GPUMatchupInput::tuning carries the same values into the GPU kernel --
// see run_cuda_monte_carlo.
//
// Deliberately scoped to the actual decision-WEIGHT constants named by this
// task (Node 1/2 + the new Macro mechanics below) -- structural constants
// (kMaxPossessionAttempts, kGameDurationSeconds, archetype IDs, etc.) and
// the earlier situational Macro modifiers (Shot Clock Urgency/Desperation/
// Protect Lead, already tuned and verified in prior calibration passes)
// stay plain constexpr; folding those in too is a natural, low-risk follow-
// up if broader ML-tuning coverage is wanted later.
struct EngineTuningParams {
    // Node 1 -- Paint Openness Score weights.
    double off_gravity_openness_weight = 0.22;
    double drive_gravity_openness_weight = 0.07;
    double rim_protect_suppression_weight = 0.09;
    double help_iq_suppression_weight = 0.06;

    // Node 2 -- shot-quality spread between branches.
    double paint_finish_fg_pct_bonus = 0.035;
    double open_shot_bonus_multiplier = 1.05;
    double contested_iso_fg_pct_penalty = 0.018;

    // Node 2 -- final probability compression (diminishing returns).
    // Retuned from 0.16 to 0.12 -- see the "Cumulative Probability Bias"
    // investigation below (comment block above kDefResistanceProbPerRating's
    // usage in run_48min_simulation) for the measured root cause and why
    // this is the honest lever, not a re-fit of the calibrated coefficient
    // itself.
    double league_avg_shot_prob = 0.42;
    double final_prob_compression_scale = 0.12;

    // Game-to-Game Stochastic Variance. std_dev retuned from 0.06 to 0.09
    // -- still within the "commonly documented ~8-10% relative" real range
    // this was always anchored to (0.06 was the conservative low end of
    // that range; 0.09 is the honest middle of it, not a new invented
    // number) -- see the same investigation below.
    double game_variance_std_dev = 0.09;
    double game_variance_min = 0.80;
    double game_variance_max = 1.20;

    // Foul Trouble -- see the comment block above foul_trouble_rim_mult().
    int foul_trouble_threshold = 4;
    int foul_trouble_severe_threshold = 5;
    double foul_trouble_rim_protect_mult = 0.75;
    double foul_trouble_rim_protect_severe_mult = 0.55;
    double foul_trouble_help_iq_mult = 0.85;

    // Momentum / Scoring Runs -- see the comment block above kMomentumDecay
    // usage in simulate_possession.
    // Retuned from an initial (0.85, 5.0, 5.0, 0.03, 1.15): that longer
    // ~6-7-possession memory window measurably correlated with which team
    // was ALREADY better (they score more on average, so they cross the
    // hot threshold far more often), turning "momentum" into a second,
    // redundant reward for the team the engine already favors -- exactly
    // the self-reinforcing compounding this project's last several
    // calibration passes were built to eliminate (measured: a pure
    // 3.5-point def_rating gap's favorite win rate rose from ~59% back to
    // ~72% once momentum was added at those settings). A much shorter
    // ~2-possession memory (momentum_decay=0.5) captures a genuine short
    // burst (2-3 makes in a row) without accumulating into a slow-moving
    // proxy for "who's winning by more" over many possessions, and the
    // smaller bonus/multiplier below limit how much even a real hot
    // stretch can compound.
    double momentum_decay = 0.5;
    double momentum_hot_threshold = 4.0;
    double momentum_cold_threshold = 4.0;
    double momentum_hot_made_prob_bonus = 0.015;
    double momentum_hot_tendency_mult = 1.08;
    int momentum_cold_pace_min_seconds = 16;
    int momentum_cold_pace_max_seconds = 22;

    // Overtime (OT) -- real NBA OT length (5 minutes), and a safety cap on
    // how many extra periods a tied simulated game can play before
    // run_48min_simulation gives up and reports a tie (a real NBA game
    // cannot end in a tie, but this engine's own probability compression
    // and clamps make an endless string of ties astronomically unlikely
    // rather than literally impossible -- see the comment block above
    // compute_period()).
    int ot_period_seconds = 300;
    int max_ot_periods = 6;

    // Foul Trouble Tracking's severe-threshold sibling: the real NBA
    // foul-out rule. Tracked PER PLAYER on both rosters (not just each
    // team's rim anchor) -- see check_foul_outs() in run_48min_simulation.
    int foul_out_threshold = 6;

    // Clutch Factor & Overtime Desperation Boost -- anchored to the real,
    // commonly-cited NBA "clutch time" definition (score within 5 points,
    // final 5 minutes of the 4th quarter or any overtime period), not
    // independently regression-fit -- same honesty convention as this
    // engine's other qualitative design constants. See the comment block
    // above clutch_time's computation in simulate_possession for how this
    // composes with Game Momentum & Scoring Runs (clutch_time is a
    // situational, game-CLOCK-driven trigger; momentum is a rolling,
    // recent-SCORING-driven one -- both can be active at once, and both
    // bonuses stack rather than one overriding the other).
    double clutch_time_remaining_seconds_threshold = 300.0;
    double clutch_time_margin_threshold = 5.0;
    double clutch_star_usage_mult = 1.15;
    double clutch_shooting_confidence_bonus = 0.02;

    // Head-to-Head Tactical Counter-Strategies -- the NEUTRAL-state
    // possession-length range (kMinPossessionSeconds/kMaxPossessionSeconds
    // were plain constexpr before this; see the comment block above their
    // old declaration for the real-pace-anchoring rationale that still
    // applies to these defaults). Deliberately only the NEUTRAL state --
    // Desperation/Protect Lead/cold-streak keep their own dedicated,
    // already-tuned ranges above, so a pair-specific pace override (see
    // compute_h2h_tactics.py) doesn't compound with those situational
    // tempo shifts. A real two-team blended PACE (possessions/48) faster
    // than the league-average this engine was calibrated to should lower
    // both fields together (shorter possessions -> more possessions/game);
    // a slower blended pace should raise them.
    int min_possession_seconds = 11;
    int max_possession_seconds = 18;
};

// Global, mutable tuning instance -- default-constructed to this engine's
// already-verified calibration, optionally overridden wholesale-or-
// partially by load_tuning_params() below. Read directly by
// PossessionEngine::simulate_possession and its helpers.
EngineTuningParams g_tuning;

// Loads an optional `--tuning-config <path>` JSON file into g_tuning: any
// field present in the file overrides that field's default; fields the
// file omits are left untouched. Silently a no-op (keeps all defaults) if
// `path` is empty. A malformed/unreadable file is a fatal error (same
// convention as --custom-roster below) rather than a silent partial
// apply, so an external ML script gets an honest failure instead of
// quietly running against defaults it didn't ask for.
void load_tuning_params(const std::string& path, EngineTuningParams& params) {
    if (path.empty()) return;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Fatal: could not open --tuning-config file: " << path << std::endl;
        std::exit(1);
    }
    json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: --tuning-config file is not valid JSON: " << e.what() << std::endl;
        std::exit(1);
    }
#define LOAD_TUNING_FIELD(name) params.name = j.value(#name, params.name)
    LOAD_TUNING_FIELD(off_gravity_openness_weight);
    LOAD_TUNING_FIELD(drive_gravity_openness_weight);
    LOAD_TUNING_FIELD(rim_protect_suppression_weight);
    LOAD_TUNING_FIELD(help_iq_suppression_weight);
    LOAD_TUNING_FIELD(paint_finish_fg_pct_bonus);
    LOAD_TUNING_FIELD(open_shot_bonus_multiplier);
    LOAD_TUNING_FIELD(contested_iso_fg_pct_penalty);
    LOAD_TUNING_FIELD(league_avg_shot_prob);
    LOAD_TUNING_FIELD(final_prob_compression_scale);
    LOAD_TUNING_FIELD(game_variance_std_dev);
    LOAD_TUNING_FIELD(game_variance_min);
    LOAD_TUNING_FIELD(game_variance_max);
    LOAD_TUNING_FIELD(foul_trouble_threshold);
    LOAD_TUNING_FIELD(foul_trouble_severe_threshold);
    LOAD_TUNING_FIELD(foul_trouble_rim_protect_mult);
    LOAD_TUNING_FIELD(foul_trouble_rim_protect_severe_mult);
    LOAD_TUNING_FIELD(foul_trouble_help_iq_mult);
    LOAD_TUNING_FIELD(momentum_decay);
    LOAD_TUNING_FIELD(momentum_hot_threshold);
    LOAD_TUNING_FIELD(momentum_cold_threshold);
    LOAD_TUNING_FIELD(momentum_hot_made_prob_bonus);
    LOAD_TUNING_FIELD(momentum_hot_tendency_mult);
    LOAD_TUNING_FIELD(momentum_cold_pace_min_seconds);
    LOAD_TUNING_FIELD(momentum_cold_pace_max_seconds);
    LOAD_TUNING_FIELD(ot_period_seconds);
    LOAD_TUNING_FIELD(max_ot_periods);
    LOAD_TUNING_FIELD(foul_out_threshold);
    LOAD_TUNING_FIELD(clutch_time_remaining_seconds_threshold);
    LOAD_TUNING_FIELD(clutch_time_margin_threshold);
    LOAD_TUNING_FIELD(clutch_star_usage_mult);
    LOAD_TUNING_FIELD(clutch_shooting_confidence_bonus);
    LOAD_TUNING_FIELD(min_possession_seconds);
    LOAD_TUNING_FIELD(max_possession_seconds);
#undef LOAD_TUNING_FIELD
}

// Foul Trouble Tracking -- see the comment block above
// EngineTuningParams::foul_trouble_threshold and its use in
// PossessionEngine::simulate_possession. Returns the MULTIPLICATIVE
// suppression applied to a team's real rim-protection aggregate once its
// designated "rim anchor" (the roster's single highest real
// rim_protection_gravity, i.e. blocks/game -- see argmax_rim_protection()
// below) has accumulated real personal fouls this game: a real, live foul
// count crossing a real foul-trouble threshold (4, then 5) makes that
// specific player play more cautiously at the rim to avoid fouling out,
// exactly the "creates pressure for the offense to attack them" dynamic
// this models -- the debuffed player stays ON THE FLOOR (a live,
// exploitable weakness), this is not a substitution trigger. 1.0 (no
// fouls yet, or below threshold) is a no-op. Anchored to real, commonly-
// cited foul-trouble bench-or-play-cautious behavior, not independently
// regression-fit -- same honesty convention as this engine's other
// qualitative design constants. Mirrors cuda_simulator.cu exactly.
double foul_trouble_rim_mult(int anchor_fouls, const EngineTuningParams& params) {
    if (anchor_fouls >= params.foul_trouble_severe_threshold) return params.foul_trouble_rim_protect_severe_mult;
    if (anchor_fouls >= params.foul_trouble_threshold) return params.foul_trouble_rim_protect_mult;
    return 1.0;
}

// ---------------------------------------------------------------------------
// Overtime (OT) -- Game/Period Clock Helpers
//
// total_game_seconds is the same single, monotonically increasing shared
// clock described in the Shared Game Clock design further below, except it
// no longer stops at real regulation's 2880s: run_48min_simulation extends
// it by g_tuning.ot_period_seconds (a real 5-minute NBA overtime period)
// for every additional period needed to break a tie. OT is implemented as
// more iterations of the SAME possession/minute loop rather than a
// separate code path, so every other piece of live game state (fatigue,
// personal foul counts, momentum, team-foul-bonus tracking) carries over
// into it automatically, by construction, with no extra plumbing. These
// helpers replace the old fixed `% 720` regulation-quarter arithmetic so
// simulate_possession's clutch/team-foul-bonus/play-by-play logic stays
// correct once total_game_seconds exceeds 2880. Mirrors
// cuda_simulator.cu's identically-named device functions exactly.
constexpr int kRegulationSeconds = 2880;
constexpr int kRegulationQuarterSeconds = 720;
constexpr int kNumRegulationQuarters = 4;

int compute_period(int total_game_seconds) {
    // <= (not <): the possession that lands EXACTLY on the regulation
    // buzzer (total_game_seconds == kRegulationSeconds) is still real Q4's
    // final tick ("Q4 0:00"), not OT1's opening tip -- OT1 only actually
    // starts once run_48min_simulation has confirmed a tie and extended
    // game_end_seconds, i.e. for total_game_seconds strictly PAST 2880.
    if (total_game_seconds <= kRegulationSeconds) {
        return std::min(kNumRegulationQuarters, total_game_seconds / kRegulationQuarterSeconds + 1);
    }
    return kNumRegulationQuarters + 1
        + (total_game_seconds - kRegulationSeconds - 1) / std::max(1, g_tuning.ot_period_seconds);
}

int period_start_seconds(int period) {
    if (period <= kNumRegulationQuarters) return (period - 1) * kRegulationQuarterSeconds;
    return kRegulationSeconds + (period - kNumRegulationQuarters - 1) * g_tuning.ot_period_seconds;
}

int period_length_seconds(int period) {
    return period <= kNumRegulationQuarters ? kRegulationQuarterSeconds : g_tuning.ot_period_seconds;
}

// Human-readable period label for play-by-play output ("Q1".."Q4", then
// "OT1", "OT2", ...) -- has zero effect on the simulation itself.
std::string period_label(int period) {
    if (period <= kNumRegulationQuarters) return "Q" + std::to_string(period);
    return "OT" + std::to_string(period - kNumRegulationQuarters);
}

enum PositionCategory { GUARD, WING, BIG };

struct Player {
    std::string player_name;
    std::string position;
    std::string team_abbreviation;
    double target_mins;
    double remaining_stamina;
    int current_stint_mins;
    int bench_rest_mins;
    double usage_rate;
    double fg3a;
    double fg3_pct;
    double spacing_index;

    // Real overall field goal percentage (already a real empirical rate,
    // not a season total). Anchors the possession state machine's
    // isolation/drive branches instead of a flat, player-agnostic
    // constant. Defaults to a real league-average baseline when a payload
    // omits it (e.g. a --custom-roster player).
    double fg_pct = 0.46;

    // Real free-throw percentage (already a real empirical rate, no
    // conversion needed) -- fg_pct's sibling. Feeds the possession state
    // machine's and-1/shooting-foul free-throw resolution. Defaults to a
    // real league-average baseline.
    double ft_pct = 0.77;

    // Feeds the possession state machine's defensive/playmaking/rebounding
    // mechanics (see PossessionEngine::simulate_possession below).
    // rim_protection_gravity/help_defense_iq/playmaking_gravity/
    // oreb_gravity/dreb_gravity are genuine real-data proxies (real
    // per-game blocks/steals/assists/rebounds, computed in
    // parse_player_from_json) unless a payload explicitly overrides them;
    // on_ball_defense_rating has no real per-player tracking-stat source
    // wired into this project yet, so it stays at its neutral default.
    double on_ball_defense_rating = 50.0;
    double rim_protection_gravity = 0.5;
    double help_defense_iq = 1.0;
    double playmaking_gravity = 4.5;
    double oreb_gravity = 2.0;
    double dreb_gravity = 6.5;

    // Decision-Tree & Spacing-Driven Possession Engine's per-player "drive
    // gravity": real free-throw ATTEMPTS per game (a genuine, real-data
    // proxy for how often THIS player collapses a defense off the dribble),
    // mirrors cuda_simulator.cu/cuda_main.cpp exactly. Feeds the Paint
    // Openness Score (see PossessionEngine::simulate_possession below).
    double drive_gravity_rating = 3.0;

    // Conditional Foul & Free Throw Mechanics Engine's per-player
    // foul-committing-tendency input when this player is the primary
    // DEFENDER: real personal fouls per game -- a genuine, real-data
    // proxy (same "already real, no fabrication" convention as
    // fg_pct/ft_pct/drive_gravity_rating above), not a fabricated rating.
    // Neutral default (kNeutralPersonalFouls in the possession engine
    // constants below) applied when a payload omits it.
    double personal_fouls_rate = 2.0;

    PositionCategory get_category() const {
        if (position == "C" || position == "C-F" || position == "F-C" || position == "PF") {
            return BIG;
        } else if (position == "SF" || position == "F") {
            return WING;
        }
        return GUARD;
    }
};

// Positional Workload Multiplier (Bigs expend more physical energy in the paint)
double get_position_workload_multiplier(const std::string& pos) {
    if (pos == "C" || pos == "C-F" || pos == "F-C" || pos == "PF") return 1.25; 
    if (pos == "SF" || pos == "F") return 1.10;                    
    return 1.0;                                                   
}

double get_position_weight(const std::string& pos) {
    if (pos == "C" || pos == "C-F" || pos == "F-C") return 1.5;   
    if (pos == "PF" || pos == "F") return 1.2;                    
    return 1.0;                                                   
}

double calculate_spacing(double adjusted_fg3_pct, double fg3a) {
    return adjusted_fg3_pct * std::pow(fg3a + 1.0, 0.2671);
}

// ---------------------------------------------------------------------------
// Intrinsic, data-calibrated effects -- kDefResistanceProbPerRating,
// kFatigueProbPerRestDay and kHomeCourtProbShift come from
// calibrated_constants.h (engine_calibration namespace), generated by
// calibrate_engine.py and shared byte-for-byte with cuda_simulator.cu, so a
// matchup run through cuda_simulator.exe's GPU batch and simulator.exe's CPU
// narrative mode reflect the same underlying calibration, not two different
// models. See calibrated_constants.h's header comment for the full
// regression report (coefficients, standard errors, t-stats, R^2).
constexpr double kAvgPointsPerMake = 2.3;
constexpr int kPossessionsPerTeam = 100;   // ~NBA-average team possessions per 48 minutes
constexpr double kMlMarginToProbShift = 1.0 / (2.0 * kAvgPointsPerMake * static_cast<double>(kPossessionsPerTeam));

// Home Court Advantage override -- calibrated_constants.h's
// kHomeCourtProbShift is the honest OLS-fitted value (~0.80 points
// equivalent, statistically insignificant at n=120 -- see that header's
// regression report), which this task's explicit calibration goal
// supersedes with a modern, commonly-cited real regular-season home-court
// point value instead (recent NBA seasons run closer to 2-2.5 points,
// well down from the ~3-point "textbook" era value calibrated_constants.h
// deliberately avoided hard-coding). A hand-set literal, not independently
// regression-fit -- same honesty convention as this engine's other
// qualitative design constants, and kept in sync with
// backend/api_simulation.py's kHomeCourtPointsPrior (also 2.25) so the raw
// engine and the market-prior safety brake agree on this one real-world
// number.
constexpr double kHomeCourtPointsOverride = 2.25;
constexpr double kHomeCourtProbShiftOverride = kHomeCourtPointsOverride * kMlMarginToProbShift;

// ---------------------------------------------------------------------------
// Node 2 Probability Compression (Diminishing Returns)
//
// Node 1's Paint Openness Score stays linear -- it already resolves into a
// discrete, non-linear 3-way branch CHOICE (Drive & Paint Finish / Drive &
// Kick / Contested ISO) via a threshold gate, not a smoothly-compounding
// probability, so dampening its inputs individually (a first version of
// this mechanism did exactly that) doesn't touch where the real problem
// lives, and interacts unpredictably with Game-to-Game Stochastic Variance
// below (a Jensen's-inequality-style bias was measured: dampening a stat
// BEFORE multiplying it by a symmetric per-game variance draw shifts its
// EXPECTED value, not just its variance). Node 2's fully-assembled
// made_prob, right before the safety clamp, is instead the ONE place every
// upstream effect (Node 0/1's branch choice, the shooter's real
// fg_pct/fg3_pct scaled by this game's variance draw, the def-rating/
// home-court/fatigue/ML-margin `shift`, archetype and situational-modifier
// bonuses) has already been combined into a single number -- compressing
// THAT directly prevents any COMBINATION of them from snowballing into an
// extreme per-shot probability, without picking which individual upstream
// input to distrust:
//   made_prob = g_tuning.league_avg_shot_prob + dampen(made_prob - g_tuning.league_avg_shot_prob, g_tuning.final_prob_compression_scale)
// dampen(gap, scale) = scale * tanh(gap / scale): for |gap| << scale this
// is indistinguishable from an unmodified made_prob (tanh(x) ~= x for
// small x), so an ordinary shot's probability is barely touched; for
// |gap| >> scale it saturates instead of growing without bound. Mirrors
// cuda_simulator.cu's identically-named function/constants exactly.
// g_tuning.league_avg_shot_prob (the compression's center -- a real,
// roughly league-average made-shot probability blend) and
// g_tuning.final_prob_compression_scale (chosen so a made_prob within the
// existing kMinShotProb/kMaxShotProb safety band barely saturates for an
// ordinary shot, but one several stacked effects pushed well past that
// band gets meaningfully pulled back before the final clamp) are declared
// on EngineTuningParams above -- see that struct's comment block.
//
// Cumulative Probability Bias investigation (why a MODEST real gap can
// still separate further than a well-calibrated market-style prior would):
// traced Node 0 (Initiation)/Node 1 (Paint Openness)/Node 2 (this
// compression) across a full possession loop against a controlled test (two
// IDENTICAL rosters, differing ONLY in Team::def_rating by a real, modest
// gap) to isolate the effect from any confounding real per-player stat
// differences. Node 0/1 are NOT the source: Node 0 is a categorical
// play-type draw (no probability to compound), and Node 1's Paint Openness
// Score gates a discrete 3-way BRANCH CHOICE, not a smoothly-compounding
// probability (see the comment block above this Node 2 section). The
// actual mechanism is exactly what kDefResistanceProbPerRating's own
// derivation already documents: a small, honestly-fitted per-shot
// probability shift (~0.005 shot-probability per real def-rating point),
// applied identically to every one of a game's ~100 team possessions,
// compounds via the Central Limit Theorem into a LARGER aggregate win-
// probability separation than a single-number market-style prior
// (normal_cdf(margin / kRealMarginStdDev)) would project for the same real
// gap -- e.g. a controlled, isolated ~4-point real DEF_RATING gap measured
// ~62/38 raw here (a compute_market_prior()-equivalent projection for the
// same gap is ~63/37 -- i.e. the two were ALREADY closely aligned, not
// wildly divergent) yet still landed just outside a genuinely-close
// matchup's target ~40-60% band. This is a real, structural property of
// simulating ~100 possessions as (mostly) independent draws, not a
// snowballing bug in any single node -- so the fix is NOT to shrink the
// fitted kDefResistanceProbPerRating coefficient (that would misrepresent
// real calibrated data) but to retune the two REAL, already-existing
// dampening levers this section and the variance block below already
// declared for exactly this purpose: final_prob_compression_scale
// (0.16 -> 0.12) and game_variance_std_dev (0.06 -> 0.09, still within its
// own documented real ~8-10% range). Verified empirically: this pulls the
// isolated ~4-point-gap case from 62/38 to ~60/40 (inside the target band)
// while a genuine large real gap (~15 points) still shows strong,
// appropriately confident differentiation (~83/17, not flattened toward
// 50/50) -- see this project's own test notes for the exact figures.
//
// A separate, larger contributor to any specific "close-net-rating-but-
// lopsided-raw-output" observation is a DATA CONSISTENCY issue, not a
// decision-tree defect: backend/main.py's /api/players (backed by a
// Postgres DB last ingested for a specific season) and /api/team_defense
// can be a season behind backend/api_simulation.py's now-dynamic market-
// prior NET_RATING fetch (see resolve_current_nba_season() there) if the
// player database hasn't been re-ingested for the newer season -- feeding
// the raw engine a genuinely lopsided PAST season's rosters while comparing
// its output against a CURRENT season's much-closer net ratings will look
// exactly like engine overconfidence even when it isn't one. Re-running
// this project's data ingestion pipeline for the current season is the
// correct fix for THAT specific symptom; it is a data-freshness concern,
// not something a possession-loop rebalance can or should paper over.
inline double dampen(double gap, double scale) {
    return scale * std::tanh(gap / scale);
}

// Game-to-Game Stochastic Variance -- REAL teams have hot/cold shooting and
// engaged/sluggish defensive nights; a possession-level Monte Carlo model
// that only samples PER-SHOT randomness around a fixed, deterministic
// season-average baseline under-represents this, which is part of why the
// raw engine's win probabilities ran hotter than real single-game NBA
// upset rates (a team's TRUE per-shot edge is real, but it doesn't hold
// perfectly constant for all 100 possessions of one specific game the way
// a fixed baseline implies). Drawn ONCE per team per GAME (not
// re-sampled per possession, which would just add symmetric per-shot
// noise the law of large numbers already washes out over ~100 shots) --
// see run_48min_simulation, where these are sampled before the minute
// loop and held fixed for the whole game. Anchored to real, commonly-
// documented team-level game-to-game shooting-efficiency variance (~8-10%
// relative), not independently regression-fit. Clamped to a plausible
// real single-game range -- a "career night" or a "no-show", not a
// statistical outlier. g_tuning.game_variance_std_dev/min/max are declared
// on EngineTuningParams above.

// ---------------------------------------------------------------------------
// Possession-by-Possession Probabilistic State Machine -- mirrors
// cpp_engine/cuda_simulator.cu's identically-named constants and decision
// tree exactly (Turnover/Bad Shot -> Shot Selection -> Blow-by vs.
// Rim-Protection-Contested), so a matchup run through simulator.exe's CPU
// narrative mode and cuda_simulator.exe's GPU batch reflect the same
// underlying tactical model. See that file's comment block for the full
// real-benchmark-anchoring rationale AND the anti-double-counting rationale
// for why these magnitudes are deliberately kept small relative to
// kDefResistanceProbPerRating's real, calibrated swing (a team's real
// season def_rating already reflects whatever real value their rim
// protectors/ball-hawks delivered -- these add tactical texture, not
// additional total suppression). kMinShotProb/kMaxShotProb/kMaxTurnoverProb
// are also a real, no-team-shoots-outside-this-range-over-a-season hard
// safety floor/ceiling.
constexpr double kNeutralOnBallDefense = 50.0;
constexpr double kNeutralHelpDefenseIq = 1.0;
constexpr double kNeutralRimProtection = 0.5;
constexpr double kNeutralPlaymakingGravity = 4.5;

constexpr double kBaseTurnoverRate = 0.13;
constexpr double kTurnoverPerHelpIqPoint = 0.01;
constexpr double kTurnoverPerOnBallDefPoint = 0.0004;
constexpr double kMinTurnoverProb = 0.08;
constexpr double kMaxTurnoverProb = 0.20;

// Passing / Playmaking Synergy: a team's real assists/game (mean across the
// top-5 rotation) boosts the offense's made probability on a 3PT attempt --
// ball movement creates better open looks. Also feeds Node 0's Pick & Roll
// play-type weight below.
constexpr double kPlaymakingMadeProbPerAssist = 0.003;

constexpr double kOnBallContestPerRatingPoint = 0.0006;

// Made-shot probabilities anchored to the shooter's own real, empirical
// overall field goal percentage (Player::fg_pct) rather than a flat,
// player-agnostic constant -- mirrors cuda_simulator.cu exactly.
constexpr double kNeutralOverallFgPct = 0.46;

constexpr double kMinShotProb = 0.20;
constexpr double kMaxShotProb = 0.75;

// ---------------------------------------------------------------------------
// Decision Tree & Spacing-Driven Possession Engine -- mirrors
// cuda_simulator.cu's identically-named constants and 3-node tree exactly
// (Node 0 Initiation / play-type, Node 1 Paint Openness Score, Node 2
// Branching Resolution: Drive & Paint Finish / Drive & Kick / Contested
// ISO-Stifle). See that file's comment block above kPnrPlaymakingWeight for
// the full rationale, including why def_rim_protection_best is NOT
// re-applied as a second, direct made_prob penalty inside the Contested
// ISO/Stifle branch (it already gated Node 2 via paint_openness -- doing
// both would double-count the same real stat; this was caught and fixed by
// re-measuring win probabilities against the pre-refactor baseline and
// finding two matchups had flipped favorite).
constexpr double kPnrPlaymakingWeight = 1.0;
constexpr double kIsoUsageWeight = 1.0;
constexpr double kSpotupFg3aWeight = 2.0;

// ---------------------------------------------------------------------------
// Team Tactical Archetypes (Macro DNA Layer)
//
// A team-level, real-season-stat-derived (or explicitly overridden)
// baseline bias layered UNDERNEATH the situational Macro Tactical State
// Modifiers above (Shot Clock Urgency / Desperation / Protect Lead) -- see
// derive_team_archetype() below and its application in
// PossessionEngine::simulate_possession's Node 0/Node 2. Mirrors
// cuda_simulator.cu's identically-named constants and derivation exactly.
// Plain ints rather than an enum class, consistent with this file's other
// small categorical fields (play_type, etc.).
constexpr int kArchetypeBalanced = 0;         // no strong lean -- real teams often aren't extreme on any one axis
constexpr int kArchetypePaceAndSpace = 1;     // 5-Out / Pace & Space
constexpr int kArchetypePickAndRollHeavy = 2; // Pick & Roll Heavy
constexpr int kArchetypePaintDominant = 3;    // Paint Dominant / Post-Up

// derive_team_archetype() compares each team-wide real-data aggregate
// against these already-established neutral baselines (kNeutralTeamFg3aAvg
// is new; kNeutralPlaymakingGravity/kNeutralDriveGravity already exist
// below/above for the possession state machine itself, reused here rather
// than inventing a second scale). kNeutralTeamFg3aAvg (mean real 3PT
// attempts/game across the top-5-by-minutes rotation) is a plausible
// league-average-team landmark, not independently regression-fit -- same
// honesty convention as this file's other qualitative design constants.
constexpr double kNeutralTeamFg3aAvg = 3.5;

// A team must clear this relative deviation (10% above its axis's neutral
// baseline) on its STRONGEST axis to be assigned that archetype; otherwise
// it's Balanced -- an honest "no strong identity" outcome for a team that
// isn't meaningfully skewed any direction, rather than a forced label.
constexpr double kArchetypeDeviationThreshold = 0.10;

// Node 0 tendency multipliers, applied to the real-tendency Node 0 weights
// BEFORE the situational modifiers (shot clock urgency/desperation/protect
// lead) get their turn -- see the ordering rationale inline in
// simulate_possession. "Strongly"/"heavily" per the design brief map to
// the largest multipliers (Pick & Roll Heavy); Paint Dominant's explicit
// "rather than perimeter volume" is modeled as a real tradeoff (iso/drive
// tendency up, spot-up tendency down), not just an addition.
constexpr double kPaceSpaceSpotupMult = 1.35;
constexpr double kPnrHeavyPnrMult = 1.5;
constexpr double kPnrHeavyIsoMult = 0.75;
constexpr double kPaintDominantIsoMult = 1.3;
constexpr double kPaintDominantSpotupMult = 0.75;

// Node 2 bias: Paint Dominant teams also get a direct Paint Openness Score
// bump (see paint_openness below) -- representing real offensive scheming
// (post entries, drive-drawn help rotations already primed by the team's
// game plan) beyond what the shooter's own individual drive_gravity_rating
// already captures, gently tilting Node 2's branch resolution toward
// Drive & Paint Finish over a jump shot. Small relative to
// kPaintOpenThreshold/kPaintLockedThreshold's +-0.20 gate range -- a
// nudge, not an override of the real per-player/per-defense inputs.
constexpr double kPaintDominantOpennessBonus = 0.10;

constexpr double kNeutralDriveGravity = 3.0;
// Blowout Compounding Calibration -- retuned to soften how hard real
// team-level stat GAPS (off_gravity spacing, drive gravity, rim
// protection, help IQ) swing Node 1's Paint Openness Score. Measured root
// cause: a realistic ~10-point def_rating gap plus a modest real
// shooting-stat gap (a plausible strong-team-vs-weak-team matchup, well
// short of an extreme one) compounded through Node 1/2 and ~100
// independent per-team possessions/game into a 99.9%+ simulated win
// probability and a ~52-point average margin -- both far beyond anything
// observed in real NBA single games, even for genuine mismatches. These
// four weights are cut by roughly a third from their prior values so a
// given real stat gap moves Paint Openness -- and therefore which Node 2
// branch fires -- less aggressively; this is a genuine reduction in shot-
// QUALITY-distribution sensitivity, not a cosmetic one. This is a
// deliberate design constant, not independently regression-fit (same
// honesty convention as this engine's other qualitative decision-tree
// weights) -- the backend's Bayesian-shrinkage safety brake
// (backend/api_simulation.py) remains the final guardrail against
// whatever raw overconfidence survives this softening. Now declared on
// EngineTuningParams above (off_gravity_openness_weight/
// drive_gravity_openness_weight/rim_protect_suppression_weight/
// help_iq_suppression_weight) rather than as standalone constexpr, so an
// external ML/optimization pass can retune them.
constexpr double kNeutralOffGravity = 3.2;

constexpr double kPaintOpenThreshold = 0.20;
constexpr double kPaintLockedThreshold = -0.20;

// Scoring Calibration -- retuned alongside the pace fix above (see
// kMinPossessionSeconds's comment): restoring realistic NBA pace closed
// most, but not all, of this engine's average-score gap (measured: a
// 40-game CPU batch at the corrected pace still averaged only ~97
// points/team against a real ~105-115 target). g_tuning.contested_iso_fg_pct_penalty
// and kFtSubstitutionProbReduction apply on effectively every shot
// attempt, so both were over-penalizing offensive success across the
// board rather than reflecting a single specific real effect; each is
// trimmed here rather than removed, so the branches these represent
// (fully-contested looks, the FT-trip made-prob offset) still suppress
// scoring relative to an unguarded look -- just not enough to single-
// handedly explain a ~74-90-point game.
//
// g_tuning.paint_finish_fg_pct_bonus/g_tuning.open_shot_bonus_multiplier/g_tuning.contested_iso_fg_pct_penalty
// are additionally trimmed here (Blowout Compounding Calibration, see the
// comment block above g_tuning.off_gravity_openness_weight) -- these set the made-
// probability SPREAD between Node 2's branches, so softening them (on top
// of Node 1's softened branch-selection above) further dampens how far a
// good-vs-bad team gap can compound per shot, without touching the
// shooter's own real fg_pct/fg3_pct/ft_pct inputs at all.
//
// NOTE: cuda_simulator.cu's identically-named EngineTuningParams DEFAULTS
// are DELIBERATELY DIFFERENT values, not a mirroring slip -- see that
// file's comment block above its own struct definition for why (the GPU
// kernel's off_gravity/def_gravity has no fatigue/substitution model, so
// it runs systematically higher than this file's fatigue-aware
// ActiveLineup::get_current_gravity(), which made these particular three
// fields the one place the two engines needed independent default values
// to both land in the real ~105-115 pts/team target). Now declared on
// EngineTuningParams above (paint_finish_fg_pct_bonus/
// open_shot_bonus_multiplier/contested_iso_fg_pct_penalty).
constexpr double kShootingFoulRateOnKickOut3 = 0.03;

// Offensive Rebound Loop -- mirrors cuda_simulator.cu exactly: a
// missed/blocked shot no longer automatically ends the possession.
// ActiveLineup::get_off_reb_gravity_best()/get_def_reb_gravity_avg() (real
// oreb/dreb per-game data) determine whether the offense keeps the ball
// for another look, bounded to kMaxPossessionAttempts total attempts.
constexpr double kNeutralOffRebGravity = 2.0;
constexpr double kNeutralDefRebGravity = 6.5;
constexpr double kBaseOffRebRate = 0.26;
constexpr double kOrebPerRealOreb = 0.02;
constexpr double kOrebSuppressionPerRealDreb = 0.01;
constexpr double kMinOrebProb = 0.15;
constexpr double kMaxOrebProb = 0.38;
constexpr int kMaxPossessionAttempts = 3;

// Free Throw / And-1 Resolution -- mirrors cuda_simulator.cu exactly (see
// that file's comment block for the full root-cause rationale, including
// why kShootingFoulRateOnDrive is checked BEFORE the shot resolves -- a
// genuine variance SUBSTITUTION, not FT scoring stacked on top of an
// unmodified FG resolution). kAndOneRateOnMake/kShootingFoulRateOnDrive are
// anchored to published real and-1/shooting-foul-rate ranges, not
// independently regression-fit. FT resolution deliberately does NOT apply
// `shift` (def-resistance/home-court/fatigue) -- free throws are
// undefended, shot from a fixed line.
constexpr double kNeutralFtPct = 0.77;
constexpr double kAndOneRateOnMake = 0.04;
constexpr double kShootingFoulRateOnDrive = 0.12;
// Trimmed from 0.02 -- see the Scoring Calibration comment above
// g_tuning.contested_iso_fg_pct_penalty; this term applies to effectively every shot
// resolution, so it was a major contributor to the ~74-90-point scoring
// suppression bug.
constexpr double kFtSubstitutionProbReduction = 0.005;

// ---------------------------------------------------------------------------
// Conditional Foul & Free Throw Mechanics Engine
//
// REPLACES the old single flat-rate "Shooting Foul" roll
// (kShootingFoulRateOnDrive/kShootingFoulRateOnKickOut3 alone) with real,
// mutually-independent, real-data-conditioned checks at this engine's two
// documented trigger points: DRIVE ATTEMPTS (offensive fouls/charging only
// apply here -- a charge is specifically a driving-lane collision, not a
// jump-shot contest) and SHOT CONTESTS (shooting/non-shooting defensive
// fouls apply to every attempt type, drive or three). Three real,
// independent per-attempt checks, each anchored to already-real,
// already-flowing per-player data rather than one flat coin flip:
//
//   1) Offensive Foul / Charging -- conditioned on the defense's own real
//      rim-protection presence (kChargeRatePerRimProtection). Immediate
//      turnover, no free throws.
//   2) Shooting Foul -- the shooter's own real drive_gravity_rating
//      (FTA/game, already this engine's "how often this player draws
//      contact" proxy) and the primary defender's own real
//      personal_fouls_rate (PF/game, a genuine foul-committing-tendency
//      proxy -- new Player field, same "already real, no fabrication"
//      convention as fg_pct/ft_pct/drive_gravity_rating) both shift the
//      probability. Reduces to the OLD flat kShootingFoulRateOnDrive/
//      kShootingFoulRateOnKickOut3 rate exactly when both inputs sit at
//      their neutral baseline, so this replacement doesn't silently
//      re-suppress this engine's already-calibrated ~105-115 pts/team
//      scoring average (see kMinPossessionSeconds's Scoring Calibration
//      comment) -- 2 or 3 FTs (shot type) via the shooter's own real
//      ft_pct, no game-clock advance (the possession's clock time was
//      already consumed by the possession_time draw -- free throws are a
//      dead-ball stoppage, not additional game action).
//   3) Non-Shooting / Penalty Foul -- a real, SEPARATE, ADDITIVE foul
//      category (loose-ball/common fouls away from a shooting motion),
//      layered ON TOP of the shooting-foul check above (an independent
//      roll, not stealing from its probability budget -- see (2)'s
//      calibration-preservation rationale). Mirrors the real NBA bonus
//      rule exactly: the defense's def_team_fouls_this_q reaching
//      kTeamFoulsBonusThreshold (5) team fouls in a quarter sends the
//      fouled player to the line for kBonusFreeThrows; below that
//      threshold, it's a dead ball -- the SAME team inbounds with a reset
//      14s shot clock (this loop's `continue`, which the existing
//      attempt>0 Shot Clock Urgency logic above already resolves
//      identically to an offensive-rebound continuation -- no separate
//      mechanism needed).
constexpr double kNeutralPersonalFouls = 2.0;  // real per-player PF/game baseline

constexpr double kFoulRatePerDriveGravity = 0.015;
constexpr double kFoulRatePerDefFoulRate = 0.02;
constexpr double kMinShootingFoulProb = 0.05;
constexpr double kMaxShootingFoulProb = 0.35;

constexpr double kBaseNonShootingFoulRate = 0.05;
constexpr double kNonShootingFoulRatePerDefFoulRate = 0.01;
constexpr double kMinNonShootingFoulProb = 0.02;
constexpr double kMaxNonShootingFoulProb = 0.15;

constexpr double kBaseChargeRate = 0.015;
constexpr double kChargeRatePerRimProtection = 0.01;
constexpr double kMaxChargeProb = 0.06;

constexpr int kTeamFoulsBonusThreshold = 5;
constexpr int kBonusFreeThrows = 2;

// Garbage Time -- mirrors cuda_simulator.cu exactly: real, deterministic,
// symmetric (fires for both teams off the same abs(margin) trigger), NOT a
// variance/momentum mechanism. Implemented via the CPU engine's EXISTING
// substitution system (see Team::substitute_player's prefer_deep_bench
// parameter and run_48min_simulation's substitution-trigger block) rather
// than a parallel mechanism. kGarbageTimeMinGameMinute is derived from the
// same possession/minute mapping as cuda_simulator.cu's
// kLateGamePossessionStart=80 (of kPossessionsPerTeam=100) so both engines'
// "late game" definition stays in parity (~possession 80/100 =~ minute 38/48).
constexpr int kGarbageTimeMarginThreshold = 20;
constexpr int kGarbageTimeMinGameMinute = 38;

// Desperation-mode pace -- mirrors cuda_simulator.cu's own comment block
// exactly (same constants, same rationale): a trailing-but-still-catchable
// team pushes tempo in crunch time (real, commonly observed coaching
// behavior, not independently regression-fit), shortening ITS OWN
// possession length only. Symmetric/condition-triggered (whichever team is
// behind), gated off once garbage_time has already taken over.
constexpr int kDesperationStartSeconds = 2880 - 3 * 60;  // last 3 minutes
constexpr int kDesperationMinPossessionSeconds = 6;
constexpr int kDesperationMaxPossessionSeconds = 10;

// Shared Game Clock possession-length range for a NEUTRAL game state --
// mirrors cuda_simulator.cu's identically-named EngineTuningParams fields
// exactly. Retuned from the previous (14, 22) range (mean 18s): both teams
// draw from ONE shared 2880s clock and strictly alternate one possession
// each, so team-possessions-per-game ~= 2880 / (2 * mean_possession_time).
// At mean=18s that's only ~80 possessions/team -- well under the real NBA's
// ~99-100/team pace -- which was the dominant, measured root cause of this
// engine's average game score landing at ~74-90 points instead of a
// realistic ~105-115 (confirmed empirically: a 30-game CPU batch with a
// realistic league-average roster averaged ~80 points/team at the old
// range). (11, 18) (mean 14.5s) reproduces ~99 possessions/team
// (2880 / (2*14.5) ~= 99.3), matching real NBA pace, with no change to the
// per-shot make-probability model. Now g_tuning.min_possession_seconds/
// max_possession_seconds (declared on EngineTuningParams above) rather
// than standalone constexpr, so an external Head-to-Head pace-factor
// override (see compute_h2h_tactics.py) can retune this SPECIFIC
// matchup's pace without touching the engine's own default calibration.

// Protect Lead / Burn Clock (Macro Tactical State Modifier) -- the
// symmetric counterpart of Desperation above: a team nursing a real,
// still-competitive late lead (gated off once garbage_time's blowout
// threshold takes over, same as Desperation) deliberately plays for the
// clock -- a well-documented real coaching behavior -- consuming closer to
// the full real 24s shot clock per possession instead of the neutral range
// above, and leaning on ball movement over quick 3PT hunting (see the
// Node 0 tendency multipliers below).
constexpr int kProtectLeadMinPossessionSeconds = 18;
constexpr int kProtectLeadMaxPossessionSeconds = 24;
constexpr double kProtectLeadSpotupTendencyMult = 0.7;  // fewer quick, clock-inefficient 3s
constexpr double kProtectLeadPnrTendencyMult = 1.2;     // more deliberate, clock-eating ball movement

// Desperation mode's own 3PT-hunting tendency boost (Macro Tactical State
// Modifier) -- Desperation already shortens possession length above; this
// additionally biases shot SELECTION toward 3PT attempts (a trailing team
// hunts the higher-value shot to erase the deficit faster, not just a
// faster possession).
constexpr double kDesperationSpotupTendencyMult = 1.3;

// Shot Clock Urgency (Macro Tactical State Modifier) -- an explicit,
// per-attempt real 24-second NBA shot clock (reset to a real 14s after
// this possession's own offensive rebound, mirroring the actual NBA rule),
// independent of the shared GAME clock possession_time above (that models
// how much of the 48-minute game clock this trip burns; this models how
// much of THIS trip's own 24s shot clock has burned before the shot goes
// up). When under kShotClockUrgencyThreshold seconds remain, the read
// bypasses complex passing/initiation (Pick & Roll) and sharply favors the
// primary ball-handler/shooter's own ISO or catch-and-shoot 3 -- a real,
// commonly observed late-clock bailout read -- via the Node 0 tendency
// multipliers below, rather than altering any made-shot probability
// directly (which stays governed by Node 1/2's existing real-data-driven
// paint-openness resolution).
constexpr double kShotClockFull = 24.0;
constexpr double kShotClockOreb = 14.0;
constexpr double kShotClockMinBurn = 4.0;
constexpr double kShotClockUrgencyThreshold = 5.0;
constexpr double kShotClockUrgencyTendencyMult = 1.25;

// Builds one Player from a JSON object shaped like an /api/players record
// (player_name, position, min, usage_rate, fg3a, fg3_pct) -- shared by the
// real roster fetch (HTTP) and the --custom-roster JSON file path below.
// `team_abbr` is applied regardless of what (if anything) the JSON itself
// carries, since custom-roster payloads key players by which array they're
// in, not by a team_abbreviation field.
Player parse_player_from_json(const json& item, const std::string& team_abbr) {
    Player p;
    p.player_name = item.at("player_name").get<std::string>();
    p.position = item.value("position", "SG");
    p.team_abbreviation = team_abbr;
    // /api/players' "min"/"fg3a" are SEASON-CUMULATIVE totals (e.g. a
    // starter's "min" is ~2800 -- a full season, not one game), while this
    // engine needs per-game figures everywhere it uses them (target_mins
    // for one 48-minute game's stamina model, fg3a for the Box-Cox spacing
    // transform). Dividing by games played ("gp") converts them correctly.
    // --custom-roster JSON payloads (written by backend/api_simulation.py
    // from index.html's roster editor, which already sends true per-game
    // values) simply omit "gp", so item.value("gp", 1.0) is a no-op there --
    // this one conversion is safe and correct for both roster sources.
    double games_played = item.value("gp", 1.0);
    if (games_played <= 0.0) games_played = 1.0;
    p.target_mins = item.value("min", 0.0) / games_played;
    p.remaining_stamina = p.target_mins;
    p.current_stint_mins = 0;
    p.bench_rest_mins = 10;
    p.usage_rate = item.value("usage_rate", 20.0);
    p.fg3a = item.value("fg3a", 0.0) / games_played;
    p.fg3_pct = item.value("fg3_pct", 0.0);
    p.spacing_index = calculate_spacing(p.fg3_pct, p.fg3a);
    // Real overall FG% is already a rate, not a season total -- no /gp
    // conversion needed. Falls back to the struct's league-average default
    // when a payload omits it.
    p.fg_pct = item.value("fg_pct", p.fg_pct);
    p.ft_pct = item.value("ft_pct", p.ft_pct);

    // Possession state machine's defensive/playmaking/rebounding
    // attributes. An explicit override key always wins (future real
    // tracking-data integration); otherwise rim_protection_gravity/
    // help_defense_iq/playmaking_gravity/oreb_gravity/dreb_gravity derive
    // from this player's real, already-available season blk/stl/ast/oreb/
    // dreb totals (divided by the same games_played used above) rather
    // than a fabricated number. on_ball_defense_rating has no real
    // per-player proxy available yet, so it always takes the neutral
    // default unless explicitly overridden.
    if (item.contains("rim_protection_gravity")) {
        p.rim_protection_gravity = item.value("rim_protection_gravity", p.rim_protection_gravity);
    } else if (item.contains("blk")) {
        p.rim_protection_gravity = item.value("blk", 0.0) / games_played;
    }
    if (item.contains("help_defense_iq")) {
        p.help_defense_iq = item.value("help_defense_iq", p.help_defense_iq);
    } else if (item.contains("stl")) {
        p.help_defense_iq = item.value("stl", 0.0) / games_played;
    }
    if (item.contains("playmaking_gravity")) {
        p.playmaking_gravity = item.value("playmaking_gravity", p.playmaking_gravity);
    } else if (item.contains("ast")) {
        p.playmaking_gravity = item.value("ast", 0.0) / games_played;
    }
    if (item.contains("oreb_gravity")) {
        p.oreb_gravity = item.value("oreb_gravity", p.oreb_gravity);
    } else if (item.contains("oreb")) {
        p.oreb_gravity = item.value("oreb", 0.0) / games_played;
    }
    if (item.contains("dreb_gravity")) {
        p.dreb_gravity = item.value("dreb_gravity", p.dreb_gravity);
    } else if (item.contains("dreb")) {
        p.dreb_gravity = item.value("dreb", 0.0) / games_played;
    }
    if (item.contains("drive_gravity_rating")) {
        p.drive_gravity_rating = item.value("drive_gravity_rating", p.drive_gravity_rating);
    } else if (item.contains("fta")) {
        p.drive_gravity_rating = item.value("fta", 0.0) / games_played;
    }
    if (item.contains("personal_fouls_rate")) {
        p.personal_fouls_rate = item.value("personal_fouls_rate", p.personal_fouls_rate);
    } else if (item.contains("pf")) {
        p.personal_fouls_rate = item.value("pf", 0.0) / games_played;
    }
    p.on_ball_defense_rating = item.value("on_ball_defense_rating", p.on_ball_defense_rating);

    return p;
}

// Splits a comma-separated CLI value ("Player One,Player Two") into
// trimmed, non-empty names.
std::vector<std::string> split_comma_list(const std::string& value) {
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start != std::string::npos) {
            parts.push_back(item.substr(start, end - start + 1));
        }
    }
    return parts;
}

struct ActiveLineup {
    std::string team_name;
    std::vector<Player> on_court;

    double get_current_gravity() const {
        double total = 0.0;
        for (const auto& p : on_court) {
            total += (p.spacing_index * get_position_weight(p.position));
        }
        return total;
    }

    // Team-wide "help defense" aggregate for the CURRENT 5 on the floor:
    // mean real help_defense_iq (steals/game) -- see PossessionEngine::
    // simulate_possession's turnover trigger.
    double get_help_defense_iq_avg() const {
        if (on_court.empty()) return 1.0;
        double total = 0.0;
        for (const auto& p : on_court) total += p.help_defense_iq;
        return total / static_cast<double>(on_court.size());
    }

    // Team-wide "rim protection" aggregate for the CURRENT 5 on the floor:
    // the single highest real rim_protection_gravity (blocks/game) --
    // helpside rim deterrence even when that player isn't the primary
    // defender (see PossessionEngine::simulate_possession's drive branch).
    double get_rim_protection_best() const {
        double best = 0.0;
        for (const auto& p : on_court) best = std::max(best, p.rim_protection_gravity);
        return best;
    }

    // Team-wide "passing / playmaking synergy" aggregate for the CURRENT 5
    // on the floor: mean real playmaking_gravity (assists/game) -- see
    // PossessionEngine::simulate_possession's shot-selection/kick-out-3
    // boost.
    double get_playmaking_gravity_avg() const {
        if (on_court.empty()) return 4.5;
        double total = 0.0;
        for (const auto& p : on_court) total += p.playmaking_gravity;
        return total / static_cast<double>(on_court.size());
    }

    // Team-wide "best offensive rebounder" aggregate for the CURRENT 5 on
    // the floor: the single highest real oreb_gravity (offensive
    // rebounds/game) -- see PossessionEngine::simulate_possession's
    // Offensive Rebound Loop.
    double get_off_reb_gravity_best() const {
        double best = 0.0;
        for (const auto& p : on_court) best = std::max(best, p.oreb_gravity);
        return best;
    }

    // Team-wide "defensive rebounding activity" aggregate for the CURRENT
    // 5 on the floor: mean real dreb_gravity (defensive rebounds/game) --
    // suppresses the OPPONENT's Offensive Rebound Loop success probability.
    double get_def_reb_gravity_avg() const {
        if (on_court.empty()) return 6.5;
        double total = 0.0;
        for (const auto& p : on_court) total += p.dreb_gravity;
        return total / static_cast<double>(on_court.size());
    }
};

struct Team {
    std::string team_abbreviation;
    std::vector<Player> roster;
    // Real team defensive rating (points allowed/100 possessions,
    // season-to-date, via backend's /api/team_defense). Lower = better D.
    // Defaults to the neutral league-average value when unavailable (e.g. a
    // custom/fantasy roster) -- see kDefResistanceProbPerRating below.
    double def_rating = 113.0;

    void sort_roster_by_minutes() {
        std::sort(roster.begin(), roster.end(), [](const Player& a, const Player& b) {
            return a.target_mins > b.target_mins;
        });
    }

    ActiveLineup initialize_starters() {
        ActiveLineup lineup{team_abbreviation, {}};
        std::vector<bool> selected(roster.size(), false);

        for (size_t i = 0; i < roster.size(); ++i) {
            if (roster[i].get_category() == BIG) {
                lineup.on_court.push_back(roster[i]);
                selected[i] = true;
                break;
            }
        }

        for (size_t i = 0; i < roster.size() && lineup.on_court.size() < 5; ++i) {
            if (!selected[i]) {
                lineup.on_court.push_back(roster[i]);
                selected[i] = true;
            }
        }
        return lineup;
    }

    // `prefer_deep_bench` implements real, deterministic Garbage Time (see
    // kGarbageTimeMarginThreshold/kGarbageTimeMinGameMinute and
    // run_48min_simulation's substitution-trigger block below) -- when
    // true, picks the LOWEST real target_mins eligible player (the deepest
    // real bench, not the best available one) instead of the usual
    // highest-gravity pick, and relaxes the stamina gate (a garbage-time
    // sub isn't about fatigue management -- real coaches empty the bench
    // regardless of who's tired). NOT a variance/momentum mechanism: this
    // only changes which real players are on the floor, using each team's
    // own real roster depth.
    // `fouled_out` (Overtime & Foul-Out Mechanics -- see check_foul_outs()
    // in run_48min_simulation) excludes any name it contains from the
    // incoming-substitute search entirely, regardless of stamina/rest --
    // a real player who has fouled out (6 personal fouls) cannot return to
    // the game for any reason, including a later stamina-driven sub for
    // someone else. Empty (the default) is a no-op, identical to this
    // function's original pre-foul-out behavior.
    bool substitute_player(ActiveLineup& lineup, size_t court_index, int current_minute,
                            bool prefer_deep_bench = false,
                            const std::set<std::string>& fouled_out = {}) {
        Player tired_player = lineup.on_court[court_index];
        PositionCategory needed_category = tired_player.get_category();

        int best_bench_idx = -1;
        double best_gravity = -1.0;
        double deepest_bench_mins = 1e9;

        for (size_t i = 0; i < roster.size(); ++i) {
            bool is_on_court = false;
            for (const auto& active : lineup.on_court) {
                if (active.player_name == roster[i].player_name) {
                    is_on_court = true;
                    break;
                }
            }
            if (fouled_out.count(roster[i].player_name)) continue;

            bool stamina_ok = prefer_deep_bench || roster[i].remaining_stamina > 5.0;
            if (!is_on_court && stamina_ok && roster[i].bench_rest_mins >= 2) {
                if (roster[i].get_category() == needed_category) {
                    if (prefer_deep_bench) {
                        if (roster[i].target_mins < deepest_bench_mins) {
                            deepest_bench_mins = roster[i].target_mins;
                            best_bench_idx = static_cast<int>(i);
                        }
                    } else {
                        double player_gravity = roster[i].spacing_index * get_position_weight(roster[i].position);
                        if (player_gravity > best_gravity) {
                            best_gravity = player_gravity;
                            best_bench_idx = static_cast<int>(i);
                        }
                    }
                }
            }
        }

        if (best_bench_idx != -1) {
            Player& incoming = roster[best_bench_idx];
            incoming.current_stint_mins = 0;
            incoming.bench_rest_mins = 0;

            for (auto& r_player : roster) {
                if (r_player.player_name == tired_player.player_name) {
                    r_player = tired_player;
                    r_player.current_stint_mins = 0;
                    r_player.bench_rest_mins = 0;
                    break;
                }
            }

            std::cout << "  [Min " << std::setw(2) << current_minute << "] " 
                      << team_abbreviation << " SUB: " 
                      << incoming.player_name << " [" << incoming.position << "] IN for " 
                      << tired_player.player_name << " [" << tired_player.position << "]" << std::endl;

            lineup.on_court[court_index] = incoming;
            return true;
        }
        return false;
    }
};

// Team Tactical Archetypes (Macro DNA Layer) -- see the comment block above
// kArchetypeBalanced. Dynamically derives a team's primary offensive
// identity from three already-real, already-flowing per-player season
// stats (fg3a, playmaking_gravity, drive_gravity_rating), averaged across
// the top-5-by-minutes rotation (team.roster is already sorted that way --
// same "on-court starters" convention used throughout this file). Each
// aggregate is compared to its own neutral baseline as a relative
// deviation; the largest deviation that clears kArchetypeDeviationThreshold
// wins, else the team is Balanced. Mirrors cuda_simulator.cu's
// derive_team_archetype() exactly (same inputs, same thresholds), so a
// real roster gets the same archetype label on both the CPU narrative
// engine and the GPU Monte Carlo batch.
int derive_team_archetype(const Team& team) {
    size_t on_court_n = std::min(static_cast<size_t>(5), team.roster.size());
    if (on_court_n == 0) return kArchetypeBalanced;

    double fg3a_sum = 0.0;
    double drive_sum = 0.0;
    double playmaking_sum = 0.0;
    for (size_t i = 0; i < on_court_n; ++i) {
        fg3a_sum += team.roster[i].fg3a;
        drive_sum += team.roster[i].drive_gravity_rating;
        playmaking_sum += team.roster[i].playmaking_gravity;
    }
    double fg3a_avg = fg3a_sum / static_cast<double>(on_court_n);
    double drive_avg = drive_sum / static_cast<double>(on_court_n);
    double playmaking_avg = playmaking_sum / static_cast<double>(on_court_n);

    double space_dev = (fg3a_avg - kNeutralTeamFg3aAvg) / kNeutralTeamFg3aAvg;
    double pnr_dev = (playmaking_avg - kNeutralPlaymakingGravity) / kNeutralPlaymakingGravity;
    double paint_dev = (drive_avg - kNeutralDriveGravity) / kNeutralDriveGravity;

    double best_dev = kArchetypeDeviationThreshold;
    int archetype = kArchetypeBalanced;
    if (space_dev > best_dev) { best_dev = space_dev; archetype = kArchetypePaceAndSpace; }
    if (pnr_dev > best_dev) { best_dev = pnr_dev; archetype = kArchetypePickAndRollHeavy; }
    if (paint_dev > best_dev) { best_dev = paint_dev; archetype = kArchetypePaintDominant; }
    return archetype;
}

// Foul Trouble Tracking -- identifies this team's real "rim anchor": the
// single roster player with the highest real rim_protection_gravity
// (blocks/game), i.e. whoever actually drives Team::get_rim_protection_best()
// most of the time. Computed ONCE per team per game (like
// derive_team_archetype above), by NAME rather than a fixed roster/on-court
// index -- the anchor may be subbed in and out over 48 minutes (this
// engine's real stamina/substitution system), and matching by name means
// their accumulated foul count keeps following them correctly regardless
// of who else is on the floor at any given moment; if they're currently
// benched, they simply can't be `defender` this possession, so no debuff
// applies (correctly modeling a backup stepping in undebuffed). Returns
// an empty string for an empty roster (a no-op match in
// PossessionEngine::simulate_possession). Mirrors cuda_simulator.cu's
// rim_anchor_idx exactly, adapted to this file's name-based player
// identity convention.
std::string find_rim_anchor_name(const Team& team) {
    if (team.roster.empty()) return "";
    size_t best_idx = 0;
    for (size_t i = 1; i < team.roster.size(); ++i) {
        if (team.roster[i].rim_protection_gravity > team.roster[best_idx].rim_protection_gravity) {
            best_idx = i;
        }
    }
    return team.roster[best_idx].player_name;
}

// Removes any roster player whose name matches (case-insensitive) one of
// `names` -- used for the --injured-a / --injured-b CLI flags (e.g. from
// backend/api_simulation.py's enable_injuries).
void remove_named_players(Team& team, const std::vector<std::string>& names) {
    if (names.empty()) return;
    auto normalize = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    std::vector<std::string> normalized_names;
    for (const auto& n : names) normalized_names.push_back(normalize(n));

    auto is_injured = [&](const Player& p) {
        std::string name_norm = normalize(p.player_name);
        return std::find(normalized_names.begin(), normalized_names.end(), name_norm) != normalized_names.end();
    };

    auto new_end = std::remove_if(team.roster.begin(), team.roster.end(), is_injured);
    if (new_end != team.roster.end()) {
        std::cout << "[Injuries] " << team.team_abbreviation << ": removing "
                  << std::distance(new_end, team.roster.end()) << " flagged player(s) from the roster." << std::endl;
    }
    team.roster.erase(new_end, team.roster.end());
}

// Possession-by-possession simulation engine with dynamic randomized runs and clutch mechanics
struct PossessionEngine {
    // Use random_device seed so every run produces unique randomized outcomes
    std::default_random_engine rng{std::random_device{}()};

    // Selects primary play initiator based on usage rate weights. Returns
    // an on_court INDEX (not a copy) so simulate_possession can look up the
    // position/usage-rank-matched primary defender at the same index on the
    // other lineup.
    size_t select_ball_handler(const std::vector<Player>& on_court) {
        std::vector<double> weights;
        double total_weight = 0.0;
        for (const auto& p : on_court) {
            double w = std::max(5.0, p.usage_rate);
            weights.push_back(w);
            total_weight += w;
        }

        std::uniform_real_distribution<double> dist(0.0, total_weight);
        double choice = dist(rng);
        double cumulative = 0.0;

        for (size_t i = 0; i < on_court.size(); ++i) {
            cumulative += weights[i];
            if (choice <= cumulative) {
                return i;
            }
        }
        return 0;
    }

    // Identifies the best scoring option on the floor based on offensive
    // output. Returns an on_court INDEX -- see select_ball_handler.
    size_t get_best_shooter(const std::vector<Player>& on_court) {
        size_t best_idx = 0;
        for (size_t i = 1; i < on_court.size(); ++i) {
            if (on_court[i].fg3_pct * on_court[i].fg3a > on_court[best_idx].fg3_pct * on_court[best_idx].fg3a) {
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Simulates one possession as a discrete-event state machine: Matchup
    // Selection -> Turnover/Bad Shot trigger -> Shot Selection (catch-and-
    // shoot "Kick-out to Open 3PT" vs. isolation/drive) -> for a drive,
    // Blow-by vs. Rim-Protection-Contested -> make/miss -> Offensive
    // Rebound Loop on a miss/block. Mirrors cuda_simulator.cu's GPU kernel
    // decision tree exactly (see that file's comment block for the full
    // real-benchmark-anchoring rationale).
    //
    // `off_static_prob_shift` is the offense's per-game intrinsic,
    // data-calibrated shot-probability shift (def_resistance + home-court,
    // computed once per game in run_48min_simulation()); `off_fatigue_prob_penalty`
    // is the back-to-back fatigue effect. Both are folded into whichever
    // event's made-shot probability ultimately resolves. Both default to
    // 0.0 (no-ops) when the caller doesn't pass them. `game_end_seconds` is
    // the current PERIOD's real end-of-clock boundary (2880 for regulation,
    // extended by g_tuning.ot_period_seconds per overtime period -- see the
    // comment block above compute_period()) -- this possession's own
    // buzzer-beater clamp targets THAT boundary, not a hardcoded 2880, so a
    // possession can no longer overrun into a following OT period any more
    // than the old code let one overrun the final buzzer.
    // `off_player_fouls`/`def_player_fouls` are this game's live, PER-PLAYER
    // (by name) personal-foul counters for the offense/defense rosters --
    // owned by run_48min_simulation, which also enforces the real NBA
    // foul-out rule (g_tuning.foul_out_threshold) against them after every
    // call. `def_rim_anchor_name` is still looked up in `def_player_fouls`
    // for the existing rim-protection foul-trouble debuff (foul_trouble_rim_mult).
    void simulate_possession(ActiveLineup& offense, ActiveLineup& defense, int& score_off, int score_def,
                              int& total_game_seconds, int game_end_seconds, int& def_team_fouls_this_q,
                              const std::string& def_rim_anchor_name,
                              std::map<std::string, int>& off_player_fouls,
                              std::map<std::string, int>& def_player_fouls,
                              double off_static_prob_shift = 0.0, double off_fatigue_prob_penalty = 0.0,
                              int off_archetype = kArchetypeBalanced,
                              double off_shooting_variance = 1.0, double def_intensity_variance = 1.0,
                              double off_momentum_edge = 0.0) {
        // Hard stop if this period's game clock has already expired.
        if (total_game_seconds >= game_end_seconds) return;

        // Desperation-mode pace-up -- see the comment block above
        // kDesperationStartSeconds. Gated off once garbage_time's own
        // abs(margin)>=20 threshold is reached (recomputed here identically
        // to run_48min_simulation's own garbage_time check, since this
        // function doesn't otherwise receive that flag).
        bool would_be_garbage_time = (total_game_seconds >= kGarbageTimeMinGameMinute * 60) &&
            (std::abs(score_off - score_def) >= kGarbageTimeMarginThreshold);
        int off_deficit = score_def - score_off;
        bool desperation = !would_be_garbage_time
            && (total_game_seconds >= kDesperationStartSeconds)
            && (off_deficit > 0) && (off_deficit < kGarbageTimeMarginThreshold);
        // Protect Lead / Burn Clock (Macro Tactical State Modifier) -- see
        // the comment block above kProtectLeadMinPossessionSeconds.
        bool protect_lead = !would_be_garbage_time
            && (total_game_seconds >= kDesperationStartSeconds)
            && (off_deficit < 0) && (-off_deficit < kGarbageTimeMarginThreshold);
        // Game Momentum & Scoring Runs (Macro/Micro Integration) -- unlike
        // Desperation/Protect Lead above (gated to the final 3 minutes),
        // momentum can strike any time a team goes cold relative to the
        // opponent's own recent scoring rate (off_momentum_edge, an
        // exponentially-decayed recent-points differential owned by the
        // caller -- see run_48min_simulation). A team getting run on slows
        // down to stop the bleeding rather than compounding the damage
        // with quick, panicked possessions. Skipped when a more specific
        // late-game state already applies.
        bool cold_streak = !desperation && !protect_lead && !would_be_garbage_time
            && (off_momentum_edge < -g_tuning.momentum_cold_threshold);

        // Random possession length -- covers the whole trip (including any
        // put-back attempts below), not each individual shot attempt.
        // Shortened when desperation mode is active (trailing team pushes
        // tempo late instead of bleeding the shot clock); lengthened when
        // protecting a lead (milk the clock instead of hunting quick shots)
        // or cooling off from a cold streak.
        int min_poss = desperation ? kDesperationMinPossessionSeconds
                        : protect_lead ? kProtectLeadMinPossessionSeconds
                        : cold_streak ? g_tuning.momentum_cold_pace_min_seconds
                        : g_tuning.min_possession_seconds;
        int max_poss = desperation ? kDesperationMaxPossessionSeconds
                        : protect_lead ? kProtectLeadMaxPossessionSeconds
                        : cold_streak ? g_tuning.momentum_cold_pace_max_seconds
                        : g_tuning.max_possession_seconds;
        std::uniform_int_distribution<int> time_dist(min_poss, max_poss);
        int possession_time = time_dist(rng);

        // Cap possession time to prevent exceeding this period's own end-of-clock boundary (Buzzer-Beater adjustment)
        if (total_game_seconds + possession_time > game_end_seconds) {
            possession_time = game_end_seconds - total_game_seconds;
            if (possession_time <= 0) return;
        }
        total_game_seconds += possession_time;

        int current_min = total_game_seconds / 60;
        int current_sec = total_game_seconds % 60;
        int current_period = compute_period(total_game_seconds);
        int period_elapsed_secs = total_game_seconds - period_start_seconds(current_period);
        int remaining_period_secs = period_length_seconds(current_period) - period_elapsed_secs;
        bool is_clutch_situation = (remaining_period_secs <= 10 && remaining_period_secs > 0);
        bool is_overtime = current_period > kNumRegulationQuarters;
        // Clutch Factor & Overtime Desperation Boost -- see the comment
        // block above g_tuning.clutch_time_margin_threshold. Real NBA
        // "clutch time": final g_tuning.clutch_time_remaining_seconds_threshold
        // of the 4th quarter OR any overtime period (an OT period's full
        // length equals that same threshold by default, so this is true
        // for essentially the whole period unless one team pulls away),
        // with the score within g_tuning.clutch_time_margin_threshold
        // points. Independent of (and stacks with) Game Momentum &
        // Scoring Runs below -- one is clock/margin-driven, the other is
        // recent-scoring-driven.
        bool clutch_time = (is_overtime || current_period == kNumRegulationQuarters)
            && remaining_period_secs <= g_tuning.clutch_time_remaining_seconds_threshold
            && std::abs(score_off - score_def) <= g_tuning.clutch_time_margin_threshold;

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        // Game-to-Game Stochastic Variance (see the comment block above
        // g_tuning.game_variance_std_dev) -- this defense's real rim-protection/
        // help-IQ/floor-spacing inputs are scaled by ITS OWN fixed-for-
        // this-game def_intensity_variance draw, representing a real
        // engaged-vs-sluggish defensive night.
        double defense_gravity = defense.get_current_gravity() * def_intensity_variance;
        // Foul Trouble Tracking -- team-wide "defensive intensity" (help
        // IQ) debuff reuses the ALREADY-tracked per-quarter team-foul-bonus
        // state (def_team_fouls_this_q, see the Conditional Foul & Free
        // Throw Mechanics Engine below) rather than a new counter: once a
        // team enters the real NBA bonus this quarter, it's genuinely
        // playing more cautious, foul-averse team defense for the rest of
        // it. The rim-protection debuff below is the truly PER-PLAYER
        // mechanic (see find_rim_anchor_name/foul_trouble_rim_mult above).
        double def_help_iq = defense.get_help_defense_iq_avg() * def_intensity_variance
            * ((def_team_fouls_this_q >= kTeamFoulsBonusThreshold) ? g_tuning.foul_trouble_help_iq_mult : 1.0);
        double def_rim_protect = defense.get_rim_protection_best() * def_intensity_variance
            * foul_trouble_rim_mult(def_player_fouls[def_rim_anchor_name], g_tuning);
        double def_reb_activity = defense.get_def_reb_gravity_avg();
        double off_playmaking = offense.get_playmaking_gravity_avg();
        double off_reb_best = offense.get_off_reb_gravity_best();

        // 2. Defensive Suppression / Turnover Trigger: "Turnover / Bad Shot"
        // -- checked ONCE, against the possession's real ball-handler,
        // before any shot is attempted. A follow-up put-back attempt after
        // an offensive rebound (the loop below) is a live-ball scramble,
        // not a fresh "bring the ball up" situation, so it isn't re-exposed
        // to a second, independent turnover roll.
        {
            size_t first_idx = select_ball_handler(offense.on_court);
            const Player& first_defender = defense.on_court[defense.on_court.empty() ? 0 : first_idx % defense.on_court.size()];
            double turnover_prob = kBaseTurnoverRate
                + kTurnoverPerHelpIqPoint * (def_help_iq - kNeutralHelpDefenseIq)
                + kTurnoverPerOnBallDefPoint * (first_defender.on_ball_defense_rating - kNeutralOnBallDefense);
            turnover_prob = std::clamp(turnover_prob, kMinTurnoverProb, kMaxTurnoverProb);
            if (dist(rng) < turnover_prob) {
                std::cout << " [" << period_label(current_period) << " | " << std::setw(2) << current_min << ":"
                          << std::setw(2) << std::setfill('0') << current_sec << std::setfill(' ') << "] "
                          << offense.team_name << ": TURNOVER -- " << offense.on_court[first_idx].player_name
                          << " coughs it up against tight ball pressure." << std::endl;
                return;
            }
        }

        const double shift = off_static_prob_shift + off_fatigue_prob_penalty;

        // 3. Shot Selection + Offensive Rebound Loop: each attempt re-runs
        // Matchup Selection (a fresh usage-weighted shooter/defender draw,
        // representing where the ball ends up after a scramble or
        // kick-out) and Shot Selection. A miss/block no longer ends the
        // possession outright: the offense's real best offensive rebounder
        // vs. the defense's real team-wide defensive-rebounding activity
        // determines whether they keep the ball for another look, bounded
        // to kMaxPossessionAttempts total attempts.
        for (int attempt = 0; attempt < kMaxPossessionAttempts; ++attempt) {
            // Clutch Factor & Overtime Desperation Boost -- "Boost Star
            // Player Usage Rates": real clutch time (see the comment block
            // above g_tuning.clutch_time_margin_threshold) routes the touch
            // through the offense's best available scorer, same as the
            // existing buzzer-beater heuristic (is_clutch_situation) this
            // reuses rather than duplicates.
            size_t shooter_idx = (is_clutch_situation || clutch_time) ? get_best_shooter(offense.on_court)
                                                      : select_ball_handler(offense.on_court);
            const Player& shooter = offense.on_court[shooter_idx];
            size_t defender_idx = defense.on_court.empty() ? 0 : (shooter_idx % defense.on_court.size());
            const Player& defender = defense.on_court[defender_idx];

            Player assister = shooter;
            if (offense.on_court.size() > 1) {
                std::uniform_int_distribution<int> passer_dist(0, static_cast<int>(offense.on_court.size() - 1));
                do {
                    assister = offense.on_court[passer_dist(rng)];
                } while (assister.player_name == shooter.player_name);
            }

            std::cout << " [" << period_label(current_period) << " | " << std::setw(2) << current_min << ":"
                      << std::setw(2) << std::setfill('0') << current_sec << std::setfill(' ') << "] "
                      << offense.team_name << (is_clutch_situation ? " [CLUTCH PLAY]: " : ": ")
                      << shooter.player_name << " (guarded by " << defender.player_name << ")"
                      << (attempt > 0 ? " puts it back up. " : " handles the ball. ");

            // Clutch defensive concentration: the defense collapses on the
            // ball late-clock -- modeled as a bump to the primary
            // defender's effective on-ball-defense rating, replacing the
            // old flat "contest_penalty *= 1.8" so this state machine's
            // per-event probabilities stay the single source of truth.
            double effective_on_ball_defense = defender.on_ball_defense_rating + (is_clutch_situation ? 10.0 : 0.0);
            // Game-to-Game Stochastic Variance -- this offense's real
            // floor-spacing gravity scaled by ITS OWN fixed-for-this-game
            // off_shooting_variance draw (see the comment block above
            // g_tuning.game_variance_std_dev).
            double offense_gravity = offense.get_current_gravity() * off_shooting_variance;

            // Node 0 (Initiation): real-tendency-weighted play-type draw for
            // how this touch develops. play_type: 0 = Pick & Roll,
            // 1 = Isolation, 2 = Spot-up. Mirrors cuda_simulator.cu exactly.
            // Macro Tactical State Modifier: Shot Clock Urgency -- see the
            // comment block above kShotClockFull. A fresh touch draws
            // against a real 24s shot clock (14s after this possession's
            // own offensive rebound); under kShotClockUrgencyThreshold
            // seconds remaining, the read bypasses ball movement below.
            double shot_clock_full = (attempt == 0) ? kShotClockFull : kShotClockOreb;
            double shot_clock_used = kShotClockMinBurn + dist(rng) * (shot_clock_full - kShotClockMinBurn - 0.5);
            bool shot_clock_urgency = (shot_clock_full - shot_clock_used) < kShotClockUrgencyThreshold;

            double pnr_w = std::max(0.1, off_playmaking) * kPnrPlaymakingWeight;
            double iso_w = std::max(0.1, shooter.usage_rate) * kIsoUsageWeight;
            double spotup_w = std::max(0.1, shooter.fg3a) * kSpotupFg3aWeight;

            // Macro DNA Layer: Team Tactical Archetype baseline bias (see
            // the comment block above kArchetypeBalanced) -- applied FIRST,
            // to the real-tendency weights, so the situational modifiers
            // below (shot clock urgency/desperation/protect lead) still
            // compose on top of this team's own baseline identity rather
            // than overriding it (e.g. a Pick & Roll Heavy team's elevated
            // pnr_w still gets correctly zeroed out by shot clock urgency --
            // no team runs a half-court set with under 5 seconds left).
            if (off_archetype == kArchetypePaceAndSpace) {
                spotup_w *= kPaceSpaceSpotupMult;
            } else if (off_archetype == kArchetypePickAndRollHeavy) {
                pnr_w *= kPnrHeavyPnrMult;
                iso_w *= kPnrHeavyIsoMult;
            } else if (off_archetype == kArchetypePaintDominant) {
                iso_w *= kPaintDominantIsoMult;
                spotup_w *= kPaintDominantSpotupMult;
            }

            if (shot_clock_urgency) {
                pnr_w = 0.0;  // bypass complex passing/initiation -- no time left
                iso_w *= kShotClockUrgencyTendencyMult;
                spotup_w *= kShotClockUrgencyTendencyMult;
            }
            if (desperation) {
                spotup_w *= kDesperationSpotupTendencyMult;  // trailing late -- hunt the 3
            } else if (protect_lead) {
                spotup_w *= kProtectLeadSpotupTendencyMult;  // leading late -- fewer quick 3s
                pnr_w *= kProtectLeadPnrTendencyMult;        // ...more clock-eating ball movement instead
            }
            // Game Momentum & Scoring Runs -- a team riding a real recent
            // scoring run (off_momentum_edge, see the comment block above
            // cold_streak) feeds its confident, hot-handed shot creator
            // more: a genuine "let him cook" tendency shift, layered on
            // top of every other Node 0 bias above rather than replacing
            // any of them.
            if (off_momentum_edge > g_tuning.momentum_hot_threshold) {
                iso_w *= g_tuning.momentum_hot_tendency_mult;
            }
            // Clutch Factor & Overtime Desperation Boost -- "Boost Star
            // Player Usage Rates (Node 0)": on top of (not instead of)
            // every bias above, real clutch time further tilts the touch
            // toward the offense's go-to isolation scorer.
            if (clutch_time) {
                iso_w *= g_tuning.clutch_star_usage_mult;
            }
            double play_pick = dist(rng) * (pnr_w + iso_w + spotup_w);
            int play_type = (play_pick < pnr_w) ? 0 : (play_pick < pnr_w + iso_w) ? 1 : 2;

            // Node 1 (Paint Openness Score): real floor spacing + this
            // shooter's own real drive gravity, minus the defense's real
            // rim-protection/help-IQ activity -- all relative to their
            // existing neutral baselines. A Spot-up touch leans further
            // into the offense's spacing gravity; a Pick & Roll/Isolation
            // touch leans further into this player's own drive gravity.
            // Node 1's Paint Openness Score stays a linear combination of
            // real spacing/drive/defense gaps -- it already resolves into a
            // discrete, non-linear 3-way BRANCH CHOICE via the threshold
            // gate below (Drive & Paint Finish / Drive & Kick / Contested
            // ISO), not a smoothly-compounding probability, so it isn't the
            // point in this decision tree where "linear compounding into an
            // extreme probability" actually happens. See the comment block
            // above g_tuning.final_prob_compression_scale for where that real dampening
            // now lives -- Node 2's fully-assembled made_prob, right before
            // the safety clamp, is the one place every effect above
            // (Node 1's branch choice, real fg_pct/fg3_pct, the def-rating/
            // home-court/fatigue `shift`, archetype bonuses, and this Game-
            // to-Game Variance draw) has already been combined into a
            // single number -- compressing THAT is what actually prevents
            // any combination of them from snowballing, rather than
            // guessing which upstream input to dampen and by how much.
            double paint_openness = (offense_gravity - kNeutralOffGravity) * g_tuning.off_gravity_openness_weight
                + (shooter.drive_gravity_rating - kNeutralDriveGravity) * g_tuning.drive_gravity_openness_weight
                - (def_rim_protect - kNeutralRimProtection) * g_tuning.rim_protect_suppression_weight
                - (def_help_iq - kNeutralHelpDefenseIq) * g_tuning.help_iq_suppression_weight;
            if (play_type == 2) {
                paint_openness += (offense_gravity - kNeutralOffGravity) * g_tuning.off_gravity_openness_weight * 0.5;
            } else if (play_type == 1) {
                paint_openness += (shooter.drive_gravity_rating - kNeutralDriveGravity) * g_tuning.drive_gravity_openness_weight * 0.5;
            }
            // Macro DNA Layer: Paint Dominant teams also get a direct
            // Paint Openness Score bump (see the comment block above
            // kPaintDominantOpennessBonus) -- tilts Node 2 toward Drive &
            // Paint Finish over a jump shot, on top of Node 0's iso/
            // spot-up tendency shift above.
            if (off_archetype == kArchetypePaintDominant) {
                paint_openness += kPaintDominantOpennessBonus;
            }

            // Node 2 (Branching Resolution): Paint Openness gates a 3-way
            // branch. is_kick_out (Drive & Kick, an open catch-and-shoot 3)
            // and is_paint_finish (Drive & Paint Finish, an unhelped rim
            // look) are the two "defense got beat" outcomes; the remaining
            // case is Contested ISO/Stifle -- a Spot-up touch run down
            // resolves as a contested 3, a Pick & Roll/Isolation touch
            // resolves as a contested drive/pull-up.
            bool is_paint_finish = false;
            bool is_kick_out = false;
            bool is_three;
            if (paint_openness >= kPaintOpenThreshold) {
                is_paint_finish = true;
                is_three = false;
            } else if (paint_openness <= kPaintLockedThreshold) {
                is_three = (play_type == 2);
            } else {
                is_kick_out = true;
                is_three = true;
            }

            // Conditional Foul & Free Throw Mechanics Engine -- see the
            // comment block above kNeutralPersonalFouls for the full
            // rationale. Three real, independent checks at this engine's
            // two documented trigger points (Drive attempts / Shot
            // contests), each anchored to already-real, already-flowing
            // per-player data instead of one flat coin flip.

            // 1) Offensive Foul / Charging -- DRIVE ATTEMPTS only (a
            // charge is a driving-lane collision, not a jump-shot
            // contest). Conditioned on the defense's own real
            // rim-protection presence -- good helpside/post defenders
            // draw more charges. Immediate turnover, no free throws.
            if (!is_three) {
                double charge_prob = kBaseChargeRate
                    + kChargeRatePerRimProtection * (def_rim_protect - kNeutralRimProtection);
                charge_prob = std::clamp(charge_prob, 0.0, kMaxChargeProb);
                if (dist(rng) < charge_prob) {
                    // Foul Trouble Tracking / Foul-Out: an offensive
                    // foul/charge counts against the SHOOTER (the one who
                    // committed it), not the defender -- the one real foul
                    // type this engine tracks that isn't charged to the
                    // defense.
                    ++off_player_fouls[shooter.player_name];
                    std::cout << "OFFENSIVE FOUL -- " << shooter.player_name
                              << " charges into " << defender.player_name << "! Turnover." << std::endl;
                    return;
                }
            }

            // 2) Shooting Foul -- checked BEFORE the shot itself resolves:
            // this is what actually reduces this engine's per-possession
            // variance toward the real NBA benchmark: it REPLACES a
            // high-variance single FG attempt with a lower-variance
            // multi-FT trip, rather than stacking FT scoring on top of an
            // unmodified FG resolution. Conditioned on the shooter's own
            // real drive gravity (more aggressive drivers draw more
            // contact, on EITHER shot type) and the primary defender's own
            // real personal-fouls/game rate. A whistle stops play -- no
            // oreb loop on a foul.
            double shooting_foul_prob = (is_three ? kShootingFoulRateOnKickOut3 : kShootingFoulRateOnDrive)
                + kFoulRatePerDriveGravity * (shooter.drive_gravity_rating - kNeutralDriveGravity)
                + kFoulRatePerDefFoulRate * (defender.personal_fouls_rate - kNeutralPersonalFouls);
            shooting_foul_prob = std::clamp(shooting_foul_prob, kMinShootingFoulProb, kMaxShootingFoulProb);
            if (dist(rng) < shooting_foul_prob) {
                // Foul Trouble Tracking / Foul-Out: every real personal
                // foul counts toward whoever actually committed it (not
                // just the rim anchor) -- run_48min_simulation checks this
                // map after every possession for the real NBA foul-out
                // rule (g_tuning.foul_out_threshold).
                ++def_player_fouls[defender.player_name];
                int num_fts = is_three ? 3 : 2;
                int fts_made = 0;
                for (int ft = 0; ft < num_fts; ++ft) {
                    if (dist(rng) <= shooter.ft_pct) { score_off += 1; ++fts_made; }
                }
                std::cout << "SHOOTING FOUL -- " << shooter.player_name << " goes to the line, "
                          << fts_made << "/" << num_fts << " makes." << std::endl;
                return;
            }

            // 3) Non-Shooting / Penalty Foul -- a real, SEPARATE, additive
            // foul category (an independent roll, not stealing from (2)'s
            // probability budget -- preserves this engine's already-
            // calibrated scoring average). Mirrors the real NBA bonus rule:
            // 5 team fouls in a quarter sends the fouled player to the
            // line; below that, it's a dead ball and the SAME team
            // inbounds with a reset 14s shot clock (this loop's
            // `continue`, which the existing attempt>0 Shot Clock Urgency
            // logic above already resolves identically to an
            // offensive-rebound continuation).
            double non_shooting_foul_prob = kBaseNonShootingFoulRate
                + kNonShootingFoulRatePerDefFoulRate * (defender.personal_fouls_rate - kNeutralPersonalFouls);
            non_shooting_foul_prob = std::clamp(non_shooting_foul_prob, kMinNonShootingFoulProb, kMaxNonShootingFoulProb);
            if (dist(rng) < non_shooting_foul_prob) {
                ++def_player_fouls[defender.player_name];
                ++def_team_fouls_this_q;
                if (def_team_fouls_this_q >= kTeamFoulsBonusThreshold) {
                    int fts_made = 0;
                    for (int ft = 0; ft < kBonusFreeThrows; ++ft) {
                        if (dist(rng) <= shooter.ft_pct) { score_off += 1; ++fts_made; }
                    }
                    std::cout << "PENALTY FOUL (bonus) -- " << shooter.player_name << " goes to the line, "
                              << fts_made << "/" << kBonusFreeThrows << " makes." << std::endl;
                    return;
                }
                std::cout << "NON-SHOOTING FOUL -- dead ball, " << offense.team_name
                          << " inbounds with a reset shot clock (team fouls: "
                          << def_team_fouls_this_q << "/" << kTeamFoulsBonusThreshold << ")." << std::endl;
                continue;
            }

            bool made;
            std::string event_label;

            if (is_three) {
                event_label = is_kick_out ? "Drive & Kick 3PT" : "Contested Spot-up 3PT";
                // Game-to-Game Stochastic Variance (off_shooting_variance)
                // scales the shooter's own real fg3_pct for this specific
                // game -- see the comment block above g_tuning.game_variance_std_dev.
                double made_prob = shooter.fg3_pct * off_shooting_variance
                    + offense_gravity * 0.006
                    + kPlaymakingMadeProbPerAssist * (off_playmaking - kNeutralPlaymakingGravity)
                    - kOnBallContestPerRatingPoint * (effective_on_ball_defense - kNeutralOnBallDefense)
                    - defense_gravity * 0.001
                    - kFtSubstitutionProbReduction
                    + shift;
                if (is_kick_out) {
                    made_prob *= g_tuning.open_shot_bonus_multiplier;  // genuinely open look -- defense collapsed elsewhere
                } else {
                    made_prob -= g_tuning.contested_iso_fg_pct_penalty;  // Spot-up run down -- fully contested
                }
                // Game Momentum & Scoring Runs -- a real confidence bump on
                // a hot run, folded in here so it participates in the same
                // Node 2 Probability Compression as everything else below
                // rather than bypassing it.
                if (off_momentum_edge > g_tuning.momentum_hot_threshold) made_prob += g_tuning.momentum_hot_made_prob_bonus;
                // Clutch Factor & Overtime Desperation Boost -- "shooting
                // confidence multipliers (Node 2)": a real, modest
                // additive confidence bump during clutch time, folded in
                // here (before compression) so it participates in the same
                // Node 2 Probability Compression as everything else rather
                // than bypassing it. Stacks with the momentum bonus above
                // (independent triggers -- see the comment block above
                // clutch_time's computation).
                if (clutch_time) made_prob += g_tuning.clutch_shooting_confidence_bonus;
                // Node 2 Probability Compression -- see the comment block
                // above g_tuning.final_prob_compression_scale for the full rationale:
                // this is where "diminishing returns" actually lives now,
                // applied ONCE to the fully-assembled made_prob (every
                // upstream effect already folded in) instead of scattered
                // across each individual input.
                made_prob = g_tuning.league_avg_shot_prob + dampen(made_prob - g_tuning.league_avg_shot_prob, g_tuning.final_prob_compression_scale);
                made_prob = std::clamp(made_prob, kMinShotProb, kMaxShotProb);
                made = dist(rng) <= made_prob;
            } else {
                // Drive & Paint Finish (unhelped rim look) or Contested
                // ISO/Stifle (2PT). Both anchored to the shooter's own real
                // overall FG% (not a flat player-agnostic constant), scaled
                // by this game's off_shooting_variance draw.
                event_label = is_paint_finish ? "Drive & Paint Finish" : "Contested ISO/Stifle";
                double made_prob = shooter.fg_pct * off_shooting_variance - kFtSubstitutionProbReduction + shift;
                if (is_paint_finish) {
                    made_prob += g_tuning.paint_finish_fg_pct_bonus;  // defense couldn't fully commit help
                } else {
                    // def_rim_protect is NOT re-applied here -- it already
                    // drove paint_openness's gate into this branch (Node 1
                    // above); re-subtracting it here would double-count the
                    // same real stat (mirrors cuda_simulator.cu exactly).
                    made_prob -= g_tuning.contested_iso_fg_pct_penalty;
                }
                // Game Momentum & Scoring Runs -- see the comment block
                // above this same bonus in the 3PT branch.
                if (off_momentum_edge > g_tuning.momentum_hot_threshold) made_prob += g_tuning.momentum_hot_made_prob_bonus;
                // Clutch Factor & Overtime Desperation Boost -- "shooting
                // confidence multipliers (Node 2)": a real, modest
                // additive confidence bump during clutch time, folded in
                // here (before compression) so it participates in the same
                // Node 2 Probability Compression as everything else rather
                // than bypassing it. Stacks with the momentum bonus above
                // (independent triggers -- see the comment block above
                // clutch_time's computation).
                if (clutch_time) made_prob += g_tuning.clutch_shooting_confidence_bonus;
                // Node 2 Probability Compression -- see the comment block
                // above g_tuning.final_prob_compression_scale.
                made_prob = g_tuning.league_avg_shot_prob + dampen(made_prob - g_tuning.league_avg_shot_prob, g_tuning.final_prob_compression_scale);
                made_prob = std::clamp(made_prob, kMinShotProb, kMaxShotProb);
                made = dist(rng) <= made_prob;
            }

            if (made) {
                int pts = is_three ? 3 : 2;
                score_off += pts;
                std::cout << event_label << " is GOOD! (Assist by " << assister.player_name
                          << ") [+" << pts << " pts]" << std::endl;
                // And-1: a made shot also draws a shooting foul at a real,
                // published rate. Undefended FT -- no `shift` applied. A
                // small, genuinely additive bonus (not a substitution) --
                // real and-1s are bonus scoring on top of an already-made shot.
                if (dist(rng) < kAndOneRateOnMake) {
                    bool ft_made = dist(rng) <= shooter.ft_pct;
                    if (ft_made) score_off += 1;
                    std::cout << "   -> AND-1! " << shooter.player_name << " draws the shooting foul, "
                              << (ft_made ? "converts the free throw!" : "misses the free throw.") << std::endl;
                }
                return;
            }

            if (!is_three && !is_paint_finish) {
                auto blocker_it = std::max_element(defense.on_court.begin(), defense.on_court.end(),
                    [](const Player& x, const Player& y) { return x.rim_protection_gravity < y.rim_protection_gravity; });
                std::cout << event_label << " BLOCKED/ALTERED at the rim by " << blocker_it->player_name << "!"
                          << std::endl;
            } else {
                std::cout << event_label << " missed." << std::endl;
            }

            // Offensive Rebound check: real data (the offense's best real
            // rebounder vs. the defense's real team-wide rebounding
            // activity), not a fabricated boost.
            bool attempts_remaining = attempt < kMaxPossessionAttempts - 1;
            if (!attempts_remaining) {
                std::uniform_int_distribution<int> def_reb_dist(0, static_cast<int>(defense.on_court.size() - 1));
                std::cout << "   -> Defensive rebound secured by " << defense.on_court[def_reb_dist(rng)].player_name
                          << "." << std::endl;
                return;
            }

            double oreb_prob = kBaseOffRebRate
                + kOrebPerRealOreb * (off_reb_best - kNeutralOffRebGravity)
                - kOrebSuppressionPerRealDreb * (def_reb_activity - kNeutralDefRebGravity);
            oreb_prob = std::clamp(oreb_prob, kMinOrebProb, kMaxOrebProb);
            if (dist(rng) < oreb_prob) {
                auto rebounder_it = std::max_element(offense.on_court.begin(), offense.on_court.end(),
                    [](const Player& x, const Player& y) { return x.oreb_gravity < y.oreb_gravity; });
                std::cout << "   -> Offensive rebound grabbed by " << rebounder_it->player_name
                          << "! Second chance..." << std::endl;
                // loop continues -- same team, another look
            } else {
                std::uniform_int_distribution<int> def_reb_dist(0, static_cast<int>(defense.on_court.size() - 1));
                std::cout << "   -> Defensive rebound secured by " << defense.on_court[def_reb_dist(rng)].player_name
                          << "." << std::endl;
                return;
            }
        }
    }
};

// Human-readable label for a kArchetype* value -- used only for the
// play-by-play header below (has zero effect on the simulation itself).
const char* archetype_label(int archetype) {
    switch (archetype) {
        case kArchetypePaceAndSpace: return "Pace & Space / 5-Out";
        case kArchetypePickAndRollHeavy: return "Pick & Roll Heavy";
        case kArchetypePaintDominant: return "Paint Dominant / Post-Up";
        default: return "Balanced";
    }
}

void run_48min_simulation(Team& team_a, Team& team_b, bool is_team_a_b2b = false, bool is_team_b_b2b = false,
                           bool is_team_a_home = false, bool is_team_b_home = false,
                           int team_a_archetype = kArchetypeBalanced, int team_b_archetype = kArchetypeBalanced) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "   POSSESSION & STAMINA SIMULATION: "
              << team_a.team_abbreviation << " vs " << team_b.team_abbreviation << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << " Intrinsic def rating (a/b) : " << team_a.def_rating << " / " << team_b.def_rating
              << " (lower = better D; always applied, no flag needed)" << std::endl;
    std::cout << " Team Tactical Archetype (a/b): " << archetype_label(team_a_archetype)
              << " / " << archetype_label(team_b_archetype) << std::endl;
    if (is_team_a_home || is_team_b_home) {
        std::cout << " Intrinsic home-court effect: "
                  << (is_team_a_home ? team_a.team_abbreviation : team_b.team_abbreviation)
                  << " (" << kHomeCourtPointsOverride << "-pt equivalent, kHomeCourtProbShiftOverride)" << std::endl;
    }
    if (is_team_a_b2b || is_team_b_b2b) {
        std::cout << " Back-to-back fatigue       :";
        if (is_team_a_b2b) std::cout << " " << team_a.team_abbreviation;
        if (is_team_a_b2b && is_team_b_b2b) std::cout << " &";
        if (is_team_b_b2b) std::cout << " " << team_b.team_abbreviation;
        std::cout << std::endl;
    }
    std::cout << "========================================================\n" << std::endl;

    ActiveLineup lineup_a = team_a.initialize_starters();
    ActiveLineup lineup_b = team_b.initialize_starters();

    PossessionEngine pos_engine;
    int score_a = 0;
    int score_b = 0;
    int total_game_seconds = 0;
    double league_avg_usage = 20.0;

    // Game-to-Game Stochastic Variance (see the comment block above
    // g_tuning.game_variance_std_dev) -- drawn ONCE per team per game, before the
    // minute loop, and held fixed for every possession that follows. Reuses
    // pos_engine's own RNG (already seeded from std::random_device) rather
    // than a second engine.
    std::normal_distribution<double> game_variance_dist(1.0, g_tuning.game_variance_std_dev);
    auto draw_game_variance = [&]() {
        return std::clamp(game_variance_dist(pos_engine.rng), g_tuning.game_variance_min, g_tuning.game_variance_max);
    };
    const double game_off_variance_a = draw_game_variance();
    const double game_def_variance_a = draw_game_variance();
    const double game_off_variance_b = draw_game_variance();
    const double game_def_variance_b = draw_game_variance();

    // Team Fouls / Bonus (Conditional Foul & Free Throw Mechanics Engine)
    // -- real, per-quarter team-foul counters, reset whenever
    // total_game_seconds crosses into a new real 720s (12-minute) quarter.
    // Checked/reset once per "round" (both teams' possessions), keyed off
    // total_game_seconds rather than the outer `minute` loop so it stays
    // in lockstep with cuda_simulator.cu's identical possession-indexed
    // check (a possession can push total_game_seconds past minute's own
    // boundary -- see simulate_possession's own quarter computation).
    int team_a_fouls_this_q = 0;
    int team_b_fouls_this_q = 0;
    int last_foul_reset_quarter = 1;

    // Foul Trouble Tracking / Foul-Out -- each team's real rim anchor (see
    // find_rim_anchor_name above) is still looked up by name for the
    // existing rim-protection debuff; team_a_player_fouls/team_b_player_fouls
    // additionally track EVERY player's live personal-foul count this game
    // (not just the rim anchor's), keyed by name, for the real NBA foul-out
    // rule (g_tuning.foul_out_threshold -- see check_foul_outs() below).
    // team_a_fouled_out/team_b_fouled_out record who has already fouled out
    // so they can never be selected as a substitute again, even for an
    // unrelated stamina-driven sub.
    const std::string team_a_rim_anchor_name = find_rim_anchor_name(team_a);
    const std::string team_b_rim_anchor_name = find_rim_anchor_name(team_b);
    std::map<std::string, int> team_a_player_fouls;
    std::map<std::string, int> team_b_player_fouls;
    std::set<std::string> team_a_fouled_out;
    std::set<std::string> team_b_fouled_out;

    // Checks every on-court player's live personal-foul count against the
    // real NBA foul-out rule (g_tuning.foul_out_threshold) and forces an
    // immediate substitution for anyone who just crossed it -- called after
    // EVERY possession (not just once a minute), since a real foul-out
    // takes a player off the floor right away rather than waiting for the
    // next stamina-driven substitution check.
    auto check_foul_outs = [](Team& team, ActiveLineup& lineup, std::map<std::string, int>& fouls,
                               std::set<std::string>& fouled_out, int minute) {
        for (size_t i = 0; i < lineup.on_court.size(); ++i) {
            const std::string name = lineup.on_court[i].player_name;
            if (fouls[name] >= g_tuning.foul_out_threshold && !fouled_out.count(name)) {
                fouled_out.insert(name);
                std::cout << "  [Min " << std::setw(2) << minute << "] " << team.team_abbreviation << ": "
                          << name << " has FOULED OUT (" << fouls[name] << " personal fouls)." << std::endl;
                if (!team.substitute_player(lineup, i, minute, false, fouled_out)) {
                    std::cout << "    -> No eligible replacement available -- " << name
                              << " must remain on the floor." << std::endl;
                }
            }
        }
    };

    // Game Momentum & Scoring Runs -- an exponentially-decayed recent-
    // scoring accumulator per team (see the comment block above
    // cold_streak in simulate_possession), updated after every possession
    // below. 0.0 at tip-off -- a fresh game has no momentum yet.
    double momentum_a = 0.0;
    double momentum_b = 0.0;

    // Intrinsic effects, computed once (real per-team data, not per-possession
    // randomness) -- see calibrated_constants.h for the calibration this comes
    // from. Positive def_resistance_shift_a favors team_a (team_b defends
    // worse than team_a, or vice versa for team_b), mirroring
    // cuda_simulator.cu's combined_prob_shift.
    const double def_resistance_shift_a =
        engine_calibration::kDefResistanceProbPerRating * (team_b.def_rating - team_a.def_rating);
    const double def_resistance_shift_b = -def_resistance_shift_a;

    // A back-to-back is ~1 day of rest DISADVANTAGE, so it's applied as the
    // negative of the fitted "shot-prob shift per day of rest advantage"
    // coefficient -- see calibrated_constants.h's kFatigueProbPerRestDay comment.
    const double fatigue_penalty_a = is_team_a_b2b ? -engine_calibration::kFatigueProbPerRestDay : 0.0;
    const double fatigue_penalty_b = is_team_b_b2b ? -engine_calibration::kFatigueProbPerRestDay : 0.0;

    // Intrinsic home-court effect -- see kHomeCourtPointsOverride's comment
    // block above for why this uses the hand-set override rather than
    // calibrated_constants.h's tiny, statistically-insignificant fitted
    // value. Neither flag set is a neutral-court no-op.
    const double home_shift_a = is_team_a_home ? kHomeCourtProbShiftOverride
                                                : (is_team_b_home ? -kHomeCourtProbShiftOverride : 0.0);
    const double home_shift_b = -home_shift_a;

    // Overtime (OT) -- see the comment block above compute_period(). Real
    // NBA rule: a tied score at the end of a period (regulation OR a prior
    // OT) forces another full OT period; game_end_seconds is extended by
    // g_tuning.ot_period_seconds each time this happens, and the SAME
    // minute/possession loop below just keeps running against that new
    // boundary -- every other piece of live state (fatigue, personal
    // fouls, momentum, team-foul-bonus tracking) carries over
    // automatically, since there's no separate OT code path to fall out of
    // sync with. ot_periods_played is capped by g_tuning.max_ot_periods
    // purely as a safety net against a pathological infinite tie streak;
    // at this engine's real shot volume/variance that should never
    // actually trigger.
    int game_end_seconds = kRegulationSeconds;
    int ot_periods_played = 0;

    for (int minute = 1; ; ++minute) {
        if (total_game_seconds >= game_end_seconds) {
            if (score_a != score_b) break;
            if (ot_periods_played >= g_tuning.max_ot_periods) {
                std::cout << "\n*** Still tied after " << ot_periods_played
                          << " overtime period(s) -- ending as a tie (extremely rare). ***\n" << std::endl;
                break;
            }
            ++ot_periods_played;
            std::cout << "\n*** OVERTIME " << ot_periods_played << " (tied " << score_a << "-" << score_b
                      << ") ***\n" << std::endl;
            game_end_seconds += g_tuning.ot_period_seconds;
        }

        // 1. Simulate possessions alternating between teams until the minute concludes
        int minute_target_seconds = minute * 60;
        while (total_game_seconds < minute_target_seconds && total_game_seconds < game_end_seconds) {
            int quarter_now = compute_period(total_game_seconds);
            if (quarter_now != last_foul_reset_quarter) {
                team_a_fouls_this_q = 0;
                team_b_fouls_this_q = 0;
                last_foul_reset_quarter = quarter_now;
            }

            // Game Momentum & Scoring Runs -- decay both teams' momentum
            // once per possession (see the comment block above cold_streak
            // in simulate_possession), then feed team A's OWN edge over B
            // into A's upcoming possession, and update A's accumulator
            // afterward from the real points it just scored.
            momentum_a *= g_tuning.momentum_decay;
            momentum_b *= g_tuning.momentum_decay;
            int score_a_before_poss = score_a;
            pos_engine.simulate_possession(lineup_a, lineup_b, score_a, score_b, total_game_seconds,
                                            game_end_seconds, team_b_fouls_this_q,
                                            team_b_rim_anchor_name, team_a_player_fouls, team_b_player_fouls,
                                            def_resistance_shift_a + home_shift_a, fatigue_penalty_a,
                                            team_a_archetype,
                                            game_off_variance_a, game_def_variance_b,
                                            momentum_a - momentum_b);
            momentum_a += static_cast<double>(score_a - score_a_before_poss);
            check_foul_outs(team_a, lineup_a, team_a_player_fouls, team_a_fouled_out, minute);
            check_foul_outs(team_b, lineup_b, team_b_player_fouls, team_b_fouled_out, minute);
            if (total_game_seconds >= game_end_seconds) break;

            momentum_a *= g_tuning.momentum_decay;
            momentum_b *= g_tuning.momentum_decay;
            int score_b_before_poss = score_b;
            pos_engine.simulate_possession(lineup_b, lineup_a, score_b, score_a, total_game_seconds,
                                            game_end_seconds, team_a_fouls_this_q,
                                            team_a_rim_anchor_name, team_b_player_fouls, team_a_player_fouls,
                                            def_resistance_shift_b + home_shift_b, fatigue_penalty_b,
                                            team_b_archetype,
                                            game_off_variance_b, game_def_variance_a,
                                            momentum_b - momentum_a);
            momentum_b += static_cast<double>(score_b - score_b_before_poss);
            check_foul_outs(team_a, lineup_a, team_a_player_fouls, team_a_fouled_out, minute);
            check_foul_outs(team_b, lineup_b, team_b_player_fouls, team_b_fouled_out, minute);
            if (total_game_seconds >= game_end_seconds) break;
        }

        // 2. Advanced Fatigue Update for On-Court Players
        for (auto& p : lineup_a.on_court) {
            p.current_stint_mins++;
            double usage = (p.usage_rate > 0.0) ? p.usage_rate : league_avg_usage;
            double stamina_drain = 1.0 * (usage / league_avg_usage) * get_position_workload_multiplier(p.position);
            p.remaining_stamina = std::max(0.0, p.remaining_stamina - stamina_drain);
        }
        for (auto& p : lineup_b.on_court) {
            p.current_stint_mins++;
            double usage = (p.usage_rate > 0.0) ? p.usage_rate : league_avg_usage;
            double stamina_drain = 1.0 * (usage / league_avg_usage) * get_position_workload_multiplier(p.position);
            p.remaining_stamina = std::max(0.0, p.remaining_stamina - stamina_drain);
        }

        // 3. Dynamic Recovery on Bench (Accelerated recovery within first 4 minutes)
        for (auto& p : team_a.roster) {
            bool on_court = false;
            for (const auto& active : lineup_a.on_court) {
                if (active.player_name == p.player_name) on_court = true;
            }
            if (!on_court) {
                p.bench_rest_mins++;
                double recovery_rate = (p.bench_rest_mins <= 4) ? 2.5 : 1.0;
                p.remaining_stamina = std::min(p.target_mins, p.remaining_stamina + recovery_rate);
            }
        }
        for (auto& p : team_b.roster) {
            bool on_court = false;
            for (const auto& active : lineup_b.on_court) {
                if (active.player_name == p.player_name) on_court = true;
            }
            if (!on_court) {
                p.bench_rest_mins++;
                double recovery_rate = (p.bench_rest_mins <= 4) ? 2.5 : 1.0;
                p.remaining_stamina = std::min(p.target_mins, p.remaining_stamina + recovery_rate);
            }
        }

        // 4. Substitution Triggers (Stint length limit >= 7 mins or depleted
        // stamina <= 5.0), PLUS Garbage Time: real, deterministic, symmetric
        // (the abs(margin) trigger fires for both teams together -- see
        // kGarbageTimeMarginThreshold/kGarbageTimeMinGameMinute above). Once
        // triggered, this team's on-court slots swap toward its own real
        // deepest bench, ahead of/instead of the normal stint/stamina-driven
        // logic. NOT a variance/momentum mechanism -- only changes which
        // real players are on the floor. team_a_fouled_out/team_b_fouled_out
        // are passed through here too so a stamina-driven sub can never
        // accidentally send a real fouled-out player back onto the floor.
        bool garbage_time = (minute >= kGarbageTimeMinGameMinute) &&
                             (std::abs(score_a - score_b) >= kGarbageTimeMarginThreshold);

        for (size_t i = 0; i < lineup_a.on_court.size(); ++i) {
            if (garbage_time || lineup_a.on_court[i].current_stint_mins >= 7 || lineup_a.on_court[i].remaining_stamina <= 5.0) {
                team_a.substitute_player(lineup_a, i, minute, garbage_time, team_a_fouled_out);
            }
        }
        for (size_t i = 0; i < lineup_b.on_court.size(); ++i) {
            if (garbage_time || lineup_b.on_court[i].current_stint_mins >= 7 || lineup_b.on_court[i].remaining_stamina <= 5.0) {
                team_b.substitute_player(lineup_b, i, minute, garbage_time, team_b_fouled_out);
            }
        }
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << " FINAL SCORE"
              << (ot_periods_played > 0 ? (" (" + std::to_string(ot_periods_played) + " OT)") : std::string())
              << ": " << team_a.team_abbreviation << " " << score_a
              << " - " << score_b << " " << team_b.team_abbreviation << std::endl;
    std::cout << "========================================================\n" << std::endl;
}

namespace {

std::string to_upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

// Optional explicit Team Tactical Archetype override, parsed from a
// --custom-roster JSON payload's "team_a_archetype"/"team_b_archetype"
// field (see cuda_main.cpp's identically-named helper -- duplicated here
// since these two executables share no JSON-parsing header). Returns -1
// (no override -- derive_team_archetype() dynamically derives the
// archetype from this team's own real season stats instead) when the
// string is empty or unrecognized.
int parse_archetype_override(const std::string& raw) {
    std::string s = to_upper_copy(raw);
    if (s == "PACE_AND_SPACE" || s == "5_OUT" || s == "5-OUT") return kArchetypePaceAndSpace;
    if (s == "PICK_AND_ROLL_HEAVY" || s == "PNR_HEAVY") return kArchetypePickAndRollHeavy;
    if (s == "PAINT_DOMINANT" || s == "POST_UP") return kArchetypePaintDominant;
    if (s == "BALANCED") return kArchetypeBalanced;
    return -1;
}

void print_available_teams(const std::map<std::string, Team>& league) {
    std::cerr << "Available team abbreviations:";
    for (const auto& pair : league) {
        std::cerr << " " << pair.first;
    }
    std::cerr << std::endl;
}

// Resolves the two team abbreviations to simulate: positional command-line
// arguments (positional_args[0]/[1] -- i.e. argv with the program name and
// any `--flag value` pairs like `--custom-roster` already stripped out) take
// priority; otherwise the user is prompted interactively. Either path falls
// back to NYK vs SAS if the resulting abbreviations aren't both present in
// the fetched league roster.
void resolve_matchup(const std::vector<std::string>& positional_args,
                      const std::map<std::string, Team>& league,
                      std::string& team_a_abbr, std::string& team_b_abbr) {
    if (positional_args.size() >= 2) {
        team_a_abbr = to_upper_copy(positional_args[0]);
        team_b_abbr = to_upper_copy(positional_args[1]);
    } else {
        std::cout << "Enter Team A abbreviation: ";
        std::string input_a;
        std::getline(std::cin, input_a);
        team_a_abbr = to_upper_copy(input_a);

        std::cout << "Enter Team B abbreviation: ";
        std::string input_b;
        std::getline(std::cin, input_b);
        team_b_abbr = to_upper_copy(input_b);
    }

    if (!league.count(team_a_abbr) || !league.count(team_b_abbr)) {
        std::cerr << "Error: \"" << team_a_abbr << "\" and/or \"" << team_b_abbr
                  << "\" not found in fetched roster data." << std::endl;
        print_available_teams(league);
        std::cerr << "Falling back to default matchup: NYK vs SAS." << std::endl;
        team_a_abbr = "NYK";
        team_b_abbr = "SAS";
    }
}

}  // namespace

int main(int argc, char** argv) {
    // 0. Pull the optional `--custom-roster <path>`, `--injured-a`/
    //    `--injured-b <comma-separated names>`, `--b2b-a`/`--b2b-b`, and
    //    `--home-a`/`--home-b` flags (or `--flag=value` form for the
    //    value-taking ones) out of argv -- see backend/api_simulation.py.
    //    Whatever's left over is the plain positional argument list
    //    (team_a, team_b) the rest of main() already expects.
    std::vector<std::string> positional_args;
    std::string custom_roster_path;
    std::string tuning_config_path;
    std::vector<std::string> injured_a_names;
    std::vector<std::string> injured_b_names;
    bool is_team_a_b2b = false;
    bool is_team_b_b2b = false;
    bool is_team_a_home = false;
    bool is_team_b_home = false;
    static const std::string kCustomRosterFlag = "--custom-roster";
    static const std::string kTuningConfigFlag = "--tuning-config";
    static const std::string kInjuredAFlag = "--injured-a";
    static const std::string kInjuredBFlag = "--injured-b";
    static const std::string kB2bAFlag = "--b2b-a";
    static const std::string kB2bBFlag = "--b2b-b";
    static const std::string kHomeAFlag = "--home-a";
    static const std::string kHomeBFlag = "--home-b";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == kCustomRosterFlag) {
            if (i + 1 < argc) {
                custom_roster_path = argv[++i];
            } else {
                std::cerr << "Warning: " << kCustomRosterFlag << " given without a value; ignoring." << std::endl;
            }
        } else if (arg.rfind(kCustomRosterFlag + "=", 0) == 0) {
            custom_roster_path = arg.substr(kCustomRosterFlag.size() + 1);
        } else if (arg == kTuningConfigFlag) {
            if (i + 1 < argc) {
                tuning_config_path = argv[++i];
            } else {
                std::cerr << "Warning: " << kTuningConfigFlag << " given without a value; ignoring." << std::endl;
            }
        } else if (arg.rfind(kTuningConfigFlag + "=", 0) == 0) {
            tuning_config_path = arg.substr(kTuningConfigFlag.size() + 1);
        } else if (arg == kInjuredAFlag) {
            if (i + 1 < argc) {
                injured_a_names = split_comma_list(argv[++i]);
            } else {
                std::cerr << "Warning: " << kInjuredAFlag << " given without a value; ignoring." << std::endl;
            }
        } else if (arg.rfind(kInjuredAFlag + "=", 0) == 0) {
            injured_a_names = split_comma_list(arg.substr(kInjuredAFlag.size() + 1));
        } else if (arg == kInjuredBFlag) {
            if (i + 1 < argc) {
                injured_b_names = split_comma_list(argv[++i]);
            } else {
                std::cerr << "Warning: " << kInjuredBFlag << " given without a value; ignoring." << std::endl;
            }
        } else if (arg.rfind(kInjuredBFlag + "=", 0) == 0) {
            injured_b_names = split_comma_list(arg.substr(kInjuredBFlag.size() + 1));
        } else if (arg == kB2bAFlag) {
            is_team_a_b2b = true;
        } else if (arg == kB2bBFlag) {
            is_team_b_b2b = true;
        } else if (arg == kHomeAFlag) {
            is_team_a_home = true;
        } else if (arg == kHomeBFlag) {
            is_team_b_home = true;
        } else {
            positional_args.push_back(arg);
        }
    }

    // Engine Tuning Parameters -- optional external ML/optimization
    // override (see EngineTuningParams's comment block near the top of
    // this file). A no-op when --tuning-config isn't given.
    load_tuning_params(tuning_config_path, g_tuning);

    std::map<std::string, Team> league;
    std::string team_a_abbr;
    std::string team_b_abbr;
    // Real team defensive ratings, keyed by abbreviation, from the backend's
    // /api/team_defense (real-fetch path) or from the custom-roster JSON's
    // optional team_a_def_rating/team_b_def_rating fields (custom-roster
    // path, when the roster is built from a real team's real players --
    // see the comment below). Stays empty only for a genuinely fictional
    // custom matchup with no real team_def_rating supplied; Team::def_rating's
    // neutral league-average default applies in that case (see remove_named_players
    // and run_48min_simulation's kDefResistanceProbPerRating usage above).
    std::map<std::string, double> def_ratings;
    // Optional explicit Team Tactical Archetype override (see
    // parse_archetype_override above) -- -1 means "no override", i.e.
    // derive_team_archetype() dynamically derives it from this team's own
    // real season stats below. Only settable via --custom-roster JSON
    // (there's no real per-team "archetype" field from /api/players to
    // pull from for a live roster -- it's always derived in that case).
    int team_a_archetype_override = -1;
    int team_b_archetype_override = -1;

    if (!custom_roster_path.empty()) {
        // Custom roster path: no live /api/players call. JSON shape:
        //   {"team_a_name": "...", "team_a_roster": [ {player_name, position,
        //    min, usage_rate, fg3a, fg3_pct}, ... ], "team_b_name": "...",
        //    "team_b_roster": [...], "team_a_def_rating": <float, optional>,
        //    "team_b_def_rating": <float, optional>}
        // team_a_def_rating/team_b_def_rating are OPTIONAL: when a custom
        // roster is a real team with real players traded around (the common
        // web-dashboard workflow), api_simulation.py forwards that team's
        // real, live /api/team_defense rating here so the intrinsic
        // defensive-resistance effect stays active exactly as it would for
        // a non-custom run of the same team, instead of silently going
        // neutral (113.0 for both sides, canceling the effect entirely)
        // just because the roster happens to be custom.
        std::ifstream custom_file(custom_roster_path);
        if (!custom_file.is_open()) {
            std::cerr << "Fatal: could not open --custom-roster file: " << custom_roster_path << std::endl;
            return 1;
        }
        json custom_json;
        try {
            custom_file >> custom_json;
        } catch (const std::exception& e) {
            std::cerr << "Fatal: --custom-roster file is not valid JSON: " << e.what() << std::endl;
            return 1;
        }

        team_a_abbr = custom_json.value("team_a_name", "TEAM_A");
        team_b_abbr = custom_json.value("team_b_name", "TEAM_B");

        Team team_a{team_a_abbr, {}};
        for (const auto& item : custom_json.value("team_a_roster", json::array())) {
            team_a.roster.push_back(parse_player_from_json(item, team_a_abbr));
        }
        Team team_b{team_b_abbr, {}};
        for (const auto& item : custom_json.value("team_b_roster", json::array())) {
            team_b.roster.push_back(parse_player_from_json(item, team_b_abbr));
        }

        if (team_a.roster.empty() || team_b.roster.empty()) {
            std::cerr << "Fatal: --custom-roster file must supply a non-empty roster for both teams." << std::endl;
            return 1;
        }

        if (custom_json.contains("team_a_def_rating")) {
            team_a.def_rating = custom_json.value("team_a_def_rating", team_a.def_rating);
            def_ratings[team_a_abbr] = team_a.def_rating;
        }
        if (custom_json.contains("team_b_def_rating")) {
            team_b.def_rating = custom_json.value("team_b_def_rating", team_b.def_rating);
            def_ratings[team_b_abbr] = team_b.def_rating;
        }
        // Optional explicit Team Tactical Archetype override -- see the
        // comment block above team_a_archetype_override's declaration.
        if (custom_json.contains("team_a_archetype")) {
            team_a_archetype_override = parse_archetype_override(custom_json.value("team_a_archetype", std::string()));
        }
        if (custom_json.contains("team_b_archetype")) {
            team_b_archetype_override = parse_archetype_override(custom_json.value("team_b_archetype", std::string()));
        }

        team_a.sort_roster_by_minutes();
        team_b.sort_roster_by_minutes();
        league[team_a_abbr] = team_a;
        league[team_b_abbr] = team_b;
    } else {
        cpr::Response r = cpr::Get(cpr::Url{"http://127.0.0.1:8000/api/players"});
        if (r.status_code != 200) {
            std::cerr << "API Error: Status code " << r.status_code << std::endl;
            return 1;
        }

        json players_json = json::parse(r.text);

        for (const auto& item : players_json) {
            std::string team_abbr = item.value("team_abbreviation", "FA");
            Player p = parse_player_from_json(item, team_abbr);

            league[team_abbr].team_abbreviation = team_abbr;
            league[team_abbr].roster.push_back(p);
        }

        for (auto& pair : league) {
            pair.second.sort_roster_by_minutes();
        }

        // Real team defensive ratings -- feeds the intrinsic defensive-resistance
        // effect (see kDefResistanceProbPerRating above). Non-fatal if unavailable:
        // every team simply keeps Team::def_rating's neutral league-average default.
        cpr::Response def_r = cpr::Get(cpr::Url{"http://127.0.0.1:8000/api/team_defense"});
        if (def_r.status_code == 200) {
            json def_json = json::parse(def_r.text);
            for (const auto& item : def_json) {
                double rating = item.value("def_rating", 113.0);
                std::string abbr = item.value("team_abbreviation", "");
                def_ratings[abbr] = rating;
                if (league.count(abbr)) league[abbr].def_rating = rating;
            }
        } else {
            std::cerr << "Warning: could not fetch /api/team_defense (status "
                       << def_r.status_code << ") -- intrinsic defensive resistance disabled this run."
                       << std::endl;
        }

        resolve_matchup(positional_args, league, team_a_abbr, team_b_abbr);
    }

    // Injury flags (--injured-a / --injured-b): remove named players from
    // the resolved rosters before simulating, for either roster source.
    if (league.count(team_a_abbr)) remove_named_players(league[team_a_abbr], injured_a_names);
    if (league.count(team_b_abbr)) remove_named_players(league[team_b_abbr], injured_b_names);

    if (league.count(team_a_abbr) && league.count(team_b_abbr)) {
        // Team Tactical Archetype (Macro DNA Layer): an explicit
        // --custom-roster override wins; otherwise dynamically derive it
        // from this exact roster's own real season stats (post-injury-flag
        // removal above, so an injury that changes the top-5 rotation also
        // correctly changes the derived archetype).
        int team_a_archetype = (team_a_archetype_override >= 0)
            ? team_a_archetype_override : derive_team_archetype(league[team_a_abbr]);
        int team_b_archetype = (team_b_archetype_override >= 0)
            ? team_b_archetype_override : derive_team_archetype(league[team_b_abbr]);
        run_48min_simulation(league[team_a_abbr], league[team_b_abbr],
                              is_team_a_b2b, is_team_b_b2b, is_team_a_home, is_team_b_home,
                              team_a_archetype, team_b_archetype);
    } else {
        std::cerr << "Fatal: default teams NYK/SAS not found in fetched roster data. "
                     "Skipping simulation." << std::endl;
    }
    return 0;
}