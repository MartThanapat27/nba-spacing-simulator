#pragma once

#include <string>
#include <vector>

// Host-side entry point into the CUDA Monte Carlo engine. Implemented in
// cuda_simulator.cu. `extern "C"` only affects linkage (no C++ name
// mangling) -- both the definition and every caller are still compiled as
// C++/CUDA, so passing C++ struct types (including std::vector/std::string
// members) by const reference is fine here.
#ifdef __cplusplus
extern "C" {
#endif

// Flattened, Structure-of-Arrays roster for one team's rotation, sized for a
// single cudaMemcpy per field into device memory. Index i across all four
// vectors describes the same player.
struct GPURoster {
    std::vector<float> fg3_pct;         // 3PT field goal percentage
    std::vector<float> fg3a;            // 3PT attempts per game
    std::vector<float> usage_rate;      // offensive usage rate (possession share)
    std::vector<float> position_weight; // positional floor-spacing weight (bigs/wings/guards)

    // Real overall field goal percentage (all shot types blended, already a
    // real empirical rate -- not a season total needing /gp conversion).
    // Anchors the isolation/drive branches' made-shot probability (see
    // simulate_possession in cuda_simulator.cu) instead of a flat,
    // player-agnostic constant.
    std::vector<float> fg_pct;

    // Real free-throw percentage (already a real empirical rate, no /gp
    // conversion needed) -- GPURoster::fg_pct's sibling. Feeds the
    // possession state machine's and-1/shooting-foul free-throw resolution
    // (see simulate_possession in cuda_simulator.cu), which is what
    // actually differentiates real basketball's low-variance FT scoring
    // from its high-variance FG scoring. Neutral default (a real,
    // league-average FT%) applied when a payload omits it.
    std::vector<float> ft_pct;

    // Real free-throw ATTEMPTS per game -- a genuine, real-data proxy for
    // how often this specific player collapses a defense off the dribble
    // (aggressive downhill drivers draw more contact), used as the
    // Decision-Tree Possession Engine's per-player "drive gravity" input
    // (see kDriveGravityWeight in cuda_simulator.cu's Paint Openness Score).
    // Not a fabricated rating -- already real, already-flowing data (same
    // "already real" precedent as ft_pct/fg_pct). Neutral default applied
    // when a payload omits it (kNeutralDriveGravity in cuda_simulator.cu).
    std::vector<float> drive_gravity;

    // Per-player on-ball defense rating (0-100 scale, higher = tougher
    // individual defender; 50.0 is neutral/league-average). Used as the
    // PRIMARY DEFENDER's contest strength in the possession state machine
    // (see simulate_possession in cuda_simulator.cu) -- there's no real,
    // freely-available per-player on-ball defense tracking stat wired into
    // this project yet (that would need nba_api's leaguedashptdefend), so
    // this stays at its neutral default until that data is integrated; see
    // GPURoster::on_ball_defense_rating's construction in
    // cuda_main.cpp::build_gpu_roster for exactly how/whether it's overridden.
    std::vector<float> on_ball_defense_rating;

    // Per-player real personal fouls per game -- the Conditional Foul &
    // Free Throw Mechanics Engine's per-player foul-committing-tendency
    // input when this player is the primary DEFENDER (see
    // simulate_possession in cuda_simulator.cu). A genuine, real-data
    // proxy (same "already real, no fabrication" convention as
    // fg_pct/ft_pct/drive_gravity above), not a fabricated rating.
    // Neutral default (kNeutralPersonalFouls in cuda_simulator.cu) applied
    // when a payload omits it.
    std::vector<float> personal_fouls_rate;

    int num_players = 0;

    // Team-level (not per-player) real defensive rating: points allowed per
    // 100 possessions, season-to-date, via nba_api's leaguedashteamstats
    // (backend's /api/team_defense endpoint). Lower = better defense. Drives
    // the engine's *intrinsic* defensive-resistance effect (see
    // kDefResistanceProbPerRating in cuda_simulator.cu) -- no external flag
    // needed, this is computed automatically every game from real data.
    // Defaults to the league average (i.e. a neutral opponent) when real
    // data isn't available, e.g. for a custom/fantasy roster.
    float def_rating = 113.0f;

    // Team-wide "help defense" aggregate: mean real steals/game across the
    // top-5 (by minutes) rotation -- a genuine, real-data-grounded proxy for
    // defensive activity/instincts (closing passing lanes, digging down for
    // strips) rather than a fabricated number. Raises the possession's
    // turnover probability (see kTurnoverPerHelpIqPoint). Neutral default
    // 1.0 (a roughly league-average rotation steals/game rate).
    float help_defense_iq_avg = 1.0f;

    // Team-wide "rim protection" aggregate: the SINGLE highest real
    // blocks/game figure across the top-5 rotation (e.g. a team's Victor
    // Wembanyama-caliber anchor) -- a genuine, real-data-grounded proxy for
    // shot-blocking gravity, not a fabricated number. Suppresses made-shot
    // probability for any drive/isolation attempt this game, representing
    // helpside rim deterrence even when the anchor isn't the ball-handler's
    // direct primary defender (see kRimSuppressionPerBlockPerGame). Neutral
    // default 0.5 (a non-shot-blocking-specialist baseline).
    float rim_protection_best = 0.5f;

    // Team-wide "passing / playmaking synergy" aggregate: mean real
    // assists/game across the top-5 (by minutes) rotation -- a genuine,
    // real-data-grounded proxy for ball movement, not a fabricated number.
    // Boosts the offense's catch-and-shoot tendency and made-probability
    // (see kPlaymakingTendencyPerAssist / kPlaymakingMadeProbPerAssist), so
    // elite-ball-movement teams aren't valued purely on isolation-style
    // box-score volume (usage/fg3a). Neutral default 4.5 (a roughly
    // league-average rotation assists/game rate).
    float playmaking_gravity_avg = 4.5f;

    // Team-wide "best offensive rebounder" aggregate: the SINGLE highest
    // real offensive-rebounds/game figure across the top-5 (by minutes)
    // rotation -- offensive rebounding is disproportionately driven by one
    // or two elite glass-crashers, mirroring rim_protection_best's "best,
    // not average" convention. Raises the offensive-rebound-loop's success
    // probability (see kOrebPerRealOreb in cuda_simulator.cu). Neutral
    // default 2.0 (a real, non-specialist rotation-big oreb/game baseline).
    float off_reb_gravity_best = 2.0f;

    // Team-wide "defensive rebounding activity" aggregate: mean real
    // defensive-rebounds/game across the top-5 rotation -- a genuine,
    // real-data-grounded proxy for team-wide box-out discipline, not a
    // fabricated number. Suppresses the OPPONENT's offensive-rebound-loop
    // success probability (see kOrebSuppressionPerRealDreb). Neutral
    // default 6.5 (a roughly league-average rotation dreb/game rate).
    float def_reb_gravity_avg = 6.5f;

    // "Bench" variants of the five team-wide aggregates above, computed the
    // same way (mean/max per field) but over the roster's BOTTOM
    // kGarbageTimeBenchSize players (by minutes) instead of the top-5.
    // Used only when garbage time triggers late in a lopsided game (see
    // kGarbageTimeMarginThreshold in cuda_simulator.cu) -- these can't be
    // recomputed from a per-player array slice inside the GPU kernel
    // without new device buffers, so both the starter and bench variant of
    // each scalar are precomputed host-side and passed in; the kernel just
    // picks between them. Defaults mirror the starter defaults (a roster
    // with no real bench data behaves identically whether or not garbage
    // time triggers).
    float help_defense_iq_avg_bench = 1.0f;
    float rim_protection_best_bench = 0.5f;
    float playmaking_gravity_avg_bench = 4.5f;
    float off_reb_gravity_best_bench = 2.0f;
    float def_reb_gravity_avg_bench = 6.5f;

    // Team Tactical Archetype (Macro DNA Layer): a categorical bias on this
    // team's Node 0 play-initiation tendencies and Node 2 paint-attack lean
    // (see the kArchetype* constants and derive_team_archetype() in
    // cuda_simulator.cu -- this header has no shared enum with that file,
    // so the plain int values are duplicated in both places): 0=Balanced
    // (no strong lean), 1=Pace & Space / 5-Out, 2=Pick & Roll Heavy,
    // 3=Paint Dominant / Post-Up. When has_archetype_override is false (the
    // default), run_cuda_monte_carlo dynamically derives this team's
    // archetype from its own real per-game 3PT-attempt volume, assist
    // rate, and free-throw-attempt (drive) rate -- already-populated real
    // fields on this same struct -- rather than a fabricated label.
    // has_archetype_override is set true only when a --custom-roster JSON
    // payload explicitly supplies "team_a_archetype"/"team_b_archetype"
    // (see cuda_main.cpp::build_gpu_roster).
    int archetype = 0;
    bool has_archetype_override = false;

    // Foul Trouble Tracking (Dynamic State & Momentum Mechanics): the index
    // (within this roster's uploaded rotation) of this team's real "rim
    // anchor" -- whoever has the single highest real rim_protection_gravity
    // (blocks/game), i.e. whoever actually drives rim_protection_best above.
    // Computed once host-side in cuda_main.cpp::build_gpu_roster (a plain
    // argmax over the same per-player data already used to compute
    // rim_protection_best, same "already real, no fabrication" convention).
    // -1 for an empty roster (a no-op match in simulate_possession).
    // Mirrors main.cpp's find_rim_anchor_name() exactly, adapted to this
    // engine's array-index (rather than name-based) player identity
    // convention -- see EngineTuningParams::foul_trouble_threshold below
    // for how a live per-game foul count against this specific index
    // debuffs rim_protection_best for the rest of that game.
    int rim_anchor_idx = -1;
};

// Engine Tuning Parameters (ML/Optimization-Driven Calibration) -- groups
// this engine's real tunable decision-tree weights (Node 1 Paint Openness
// Score, Node 2 shot-quality spread + final probability compression, and
// the Macro mechanics: Game-to-Game Stochastic Variance, Foul Trouble,
// Momentum/Scoring Runs) into ONE struct instead of scattered standalone
// __constant__/constexpr values, so an external ML/optimization script
// (e.g. fitting these weights against real historical match results as a
// loss function) can hand a whole tuned parameter set to a single GPU
// batch via GPUMatchupInput::tuning, with no source changes required.
// Passed BY VALUE into monte_carlo_matchup_kernel (a small, plain-old-data
// struct -- CUDA copies kernel-parameter structs into each thread's own
// parameter space automatically, no explicit device-memory upload needed).
// Default-constructed to this engine's own already-verified GPU
// calibration -- see cuda_simulator.cu's module comments for why three of
// these fields (paint_finish_fg_pct_bonus/open_shot_bonus_multiplier/
// contested_iso_fg_pct_penalty) DELIBERATELY default to different values
// than main.cpp's identically-named EngineTuningParams fields (the GPU
// kernel's floor-spacing gravity has no fatigue/substitution model, so it
// runs systematically higher than main.cpp's fatigue-aware version).
struct EngineTuningParams {
    // Node 1 -- Paint Openness Score weights.
    float off_gravity_openness_weight = 0.22f;
    float drive_gravity_openness_weight = 0.07f;
    float rim_protect_suppression_weight = 0.09f;
    float help_iq_suppression_weight = 0.06f;

    // Node 2 -- shot-quality spread between branches.
    float paint_finish_fg_pct_bonus = 0.01f;
    float open_shot_bonus_multiplier = 0.97f;
    float contested_iso_fg_pct_penalty = 0.05f;

    // Node 2 -- final probability compression (diminishing returns).
    // Retuned from 0.16f to 0.12f -- see the Cumulative Probability Bias
    // investigation comment block in cuda_simulator.cu (above kDefResistanceProbPerRating's
    // usage) for the measured root cause and why this is the honest lever,
    // not a re-fit of the calibrated coefficient itself. Mirrors main.cpp exactly.
    float league_avg_shot_prob = 0.42f;
    float final_prob_compression_scale = 0.12f;

    // Game-to-Game Stochastic Variance. std_dev retuned from 0.06f to
    // 0.09f -- still within the "commonly documented ~8-10% relative" real
    // range this was always anchored to. Mirrors main.cpp exactly.
    float game_variance_std_dev = 0.09f;
    float game_variance_min = 0.80f;
    float game_variance_max = 1.20f;

    // Foul Trouble.
    int foul_trouble_threshold = 4;
    int foul_trouble_severe_threshold = 5;
    float foul_trouble_rim_protect_mult = 0.75f;
    float foul_trouble_rim_protect_severe_mult = 0.55f;
    float foul_trouble_help_iq_mult = 0.85f;

    // Momentum / Scoring Runs.
    float momentum_decay = 0.5f;
    float momentum_hot_threshold = 4.0f;
    float momentum_cold_threshold = 4.0f;
    float momentum_hot_made_prob_bonus = 0.015f;
    float momentum_hot_tendency_mult = 1.08f;
    int momentum_cold_pace_min_seconds = 16;
    int momentum_cold_pace_max_seconds = 22;

    // Overtime (OT) -- real NBA OT length (5 minutes), and a safety cap on
    // how many extra periods a tied simulated game can play before
    // monte_carlo_matchup_kernel gives up and reports a tie. Mirrors
    // main.cpp's identically-named fields exactly.
    int ot_period_seconds = 300;
    int max_ot_periods = 6;

    // Foul Trouble Tracking's severe-threshold sibling: the real NBA
    // foul-out rule, applied to each team's real rim anchor (the ONE
    // individually-tracked player identity this GPU engine models -- see
    // the comment block above GPURoster::rim_anchor_idx). Deliberately
    // NOT a full per-player foul-out system on this engine -- see the
    // comment block above monte_carlo_matchup_kernel's foul-out handling
    // for why that documented CPU/GPU asymmetry is the right call here.
    int foul_out_threshold = 6;

    // Clutch Factor & Overtime Desperation Boost -- anchored to the real,
    // commonly-cited NBA "clutch time" definition (score within 5 points,
    // final 5 minutes of the 4th quarter or any overtime period), not
    // independently regression-fit -- same honesty convention as this
    // engine's other qualitative design constants. Mirrors main.cpp's
    // identically-named fields exactly.
    float clutch_time_remaining_seconds_threshold = 300.0f;
    float clutch_time_margin_threshold = 5.0f;
    float clutch_star_usage_mult = 1.15f;
    float clutch_shooting_confidence_bonus = 0.02f;

    // Head-to-Head Tactical Counter-Strategies -- the NEUTRAL-state
    // possession-length range, now tunable (was plain constexpr) so an
    // external pair-specific pace override (see compute_h2h_tactics.py)
    // can retune this SPECIFIC matchup's pace without touching the
    // engine's own default calibration. Mirrors main.cpp's identically-
    // named fields exactly. Desperation/Protect Lead/cold-streak keep
    // their own dedicated, already-tuned ranges, unaffected by this.
    int min_possession_seconds = 11;
    int max_possession_seconds = 18;
};

// Inputs for one GPU Monte Carlo batch: the exact Team A vs Team B matchup
// chosen by the user, flattened into device-ready rosters.
struct GPUMatchupInput {
    std::string team_a_name;
    GPURoster team_a;
    std::string team_b_name;
    GPURoster team_b;

    // Optional external calibration signal from the project's ML margin
    // predictor (train_ml_model.py): the model's expected Team A - Team B
    // point margin for this exact matchup, e.g. +2.7 means Team A is
    // favored by ~2.7 points. Zero (the default) reproduces the engine's
    // original unbiased possession model exactly. See ml_margin_bias usage
    // in cuda_simulator.cu for how this is translated into a per-possession
    // shot-probability shift.
    double ml_margin_bias = 0.0;
    bool has_ml_margin_bias = false;

    // Optional "Big Match Hot Hands / Star Momentum" multiplier: whether and
    // how much to boost each team's highest-usage ("star") player's 3PT
    // success probability and shot-selection weight, for marquee/high-stakes
    // matchups. The caller (e.g. backtest_model.py, using train_ml_model.py's
    // big_match_indicator) decides *whether* a matchup qualifies and supplies
    // the multiplier accordingly; the kernel itself just applies whatever
    // value it's given. 1.0 (the default) is a no-op -- identical to the
    // unboosted model. See hot_hand_boost usage in cuda_simulator.cu.
    double hot_hand_boost = 1.0;
    bool has_hot_hand_boost = false;

    // Which side (if either) is the home team for this game. Drives the
    // engine's *intrinsic*, data-calibrated home-court effect
    // (engine_calibration::kHomeCourtProbShift in calibrated_constants.h,
    // fit from real game margins -- see calibrate_engine.py). Both false is
    // a neutral-court no-op; at most one should be true.
    bool is_team_a_home = false;
    bool is_team_b_home = false;

    // Optional schedule-fatigue penalty: whether team_a/team_b is playing on
    // a back-to-back (0 days rest). When set, that team's shooting
    // probability and effective floor-spacing gravity both take a hit for
    // the whole game, sized by the engine's intrinsic, data-calibrated
    // fatigue coefficient (engine_calibration::kFatigueProbPerRestDay in
    // calibrated_constants.h) -- tired legs shoot and move worse. false (the
    // default) for both is a no-op.
    bool is_team_a_b2b = false;
    bool is_team_b_b2b = false;

    // Engine Tuning Parameters (ML/Optimization-Driven Calibration) --
    // default-constructed to this engine's own already-verified GPU
    // calibration; see EngineTuningParams's comment block above. Populated
    // from an optional --tuning-config JSON file in cuda_main.cpp.
    EngineTuningParams tuning;
};

// Runs a fully parallel, GPU-resident Monte Carlo simulation of 100,000
// independent 48-minute games between team_a and team_b. Each CUDA thread
// simulates one entire game (a possession loop for both offenses, using
// curand for shooting variance and roster-derived spacing boosts) and writes
// its final score for each team; no per-thread branch divergence from the
// player-count loop bounds (uniform across all threads) and no device-side
// I/O. Results are copied back and aggregated/reported on the host.
void run_cuda_monte_carlo(const GPUMatchupInput& matchup);

#ifdef __cplusplus
}
#endif
