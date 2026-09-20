# NBA Matchup & Spacing Simulator

A high-performance basketball simulation engine that models player matchups, evaluates lineup spacing indices using continuous mathematical gravity formulas, and runs possession-by-possession game simulations with physiological fatigue, a real-data-driven decision tree, and in-game dynamic state (foul trouble, scoring runs, and per-game shooting variance).

The C++ simulation engine ships two executables from a single CMake project:
* **`simulator`** — a CPU narrative simulator that plays out a single game possession-by-possession and prints a full play-by-play log.
* **`cuda_simulator`** — the full pipeline executable: it fetches rosters from the API, runs the CPU narrative/batch engine for a chosen Team A vs Team B matchup (with an optional interactive [roster customization stage](#interactive-roster-customization--player-importtrade-pipeline) beforehand), then launches a fully parallel GPU Monte Carlo simulation of **100,000 complete 48-minute games** for that exact matchup (see [GPU Parallel Monte Carlo Engine](#gpu-parallel-monte-carlo-engine-cuda) below).

Three root-level Python scripts add a hybrid ML + Monte Carlo layer on top: `fetch_real_nba_data.py` pulls real games from `nba_api`, `train_ml_model.py` fits an XGBoost expected-margin model on them, and `backtest_model.py` drives `cuda_simulator.exe` (with and without that ML calibration) to measure real-world predictive accuracy on held-out games (see the "Hybrid ML + Monte Carlo Pipeline" section below).

**This is a Hybrid Margin & Probability Model, not a moneyline/win-probability classifier.** The ML component (`train_ml_model.py`) never predicts win/loss or a probability directly -- it's an `XGBRegressor` trained to predict a single continuous quantity, the expected point margin (`team_a`'s score minus `team_b`'s), the same target a Vegas spread models. That margin is converted into a small per-possession shot-probability shift (`--ml-margin`, see [GPU Parallel Monte Carlo Engine](#gpu-parallel-monte-carlo-engine-cuda) below) and handed to the possession-by-possession Monte Carlo engine, which then *simulates* 100,000 complete games under that bias. Win probability is therefore an emergent statistic of the simulated score distribution, not a model output in its own right -- there is no separate win/loss classifier anywhere in this pipeline, and the two accuracy metrics reported throughout this README (margin-sign accuracy and full-pipeline win/loss accuracy -- see [Hybrid ML + Monte Carlo Pipeline](#hybrid-ml--monte-carlo-pipeline-fetch_real_nba_datapy-train_ml_modelpy-backtest_modelpy) below) measure two different, non-interchangeable things for exactly that reason.

## Architecture & Tech Stack
This project separates data processing from the core simulation engine to ensure performance and flexibility:
* **Database:** PostgreSQL (via Docker)
* **Data Engineering & EDA:** Python, Pandas, Streamlit
* **Backend API:** FastAPI, SQLAlchemy, Uvicorn
* **CPU Simulation Engine:** C++17 (cpr, nlohmann/json, STL `<random>`), CMake, MSVC
* **GPU Simulation Engine:** CUDA 13.4 (nvcc, cuRAND), built via the same CMake project behind an `ENABLE_CUDA` flag

## Data Pipeline
1. **Raw Data Ingestion:** Fetches up-to-date player game logs (67 columns) via `nba_api` and stores them in the `player_stats_raw` table (ELT approach).
2. **Exploratory Data Analysis (EDA):** Streamlit dashboard validates statistical skewness, volume-efficiency relationships, and feature distributions.
3. **API Layer:** FastAPI serves transformed, position-verified roster and statistical payloads via `/api/players`. Both `simulator` and `cuda_simulator` pull their roster data from this endpoint at startup, so the backend must be running first.

## Mathematical Foundations
# 1. Continuous Floor Spacing Index (Defensive Gravity)
Defensive gravity naturally exhibits diminishing returns. Instead of arbitrary threshold tiers, the engine uses a Box-Cox Transformation ($\lambda \approx 0.2671$) derived empirically from NBA tracking data:

$$\text{Spacing\_Index} = \text{Adjusted\_FG3\_PCT} \times (\text{FG3A} + 1)^{0.2671}$$
- Empirical Volume Scaling: Accurately models how initial 3-point volume radically pulls perimeter defenders away from help positions, while extreme volume yields logarithmic returns.
- Non-Shooter Boundary: The $+1$ offset mathematically dampens non-shooting bigs down to zero gravity without undefined behavior.

# 2. Shot Probability: Assemble, then Compress
Every field goal attempt assembles a made-probability from every real signal active for that shot (shooter's own real FG%/3PT%, floor-spacing gravity, defensive resistance, archetype/momentum/foul-trouble adjustments, the calibrated def-rating/home-court/fatigue shift), then passes the fully-assembled number through a single non-linear compression — `dampen(gap, scale) = scale · tanh(gap / scale)` — centered on a league-average probability, before the final safety clamp:

$$P(\text{score}) = \text{clamp}\Big(P_{\text{league avg}} + \text{dampen}(P_{\text{raw}} - P_{\text{league avg}},\ \text{scale}),\ 0.20,\ 0.75\Big)$$

For an ordinary shot the raw assembled probability sits close to league-average, where `tanh` is indistinguishable from the identity function, so this changes nothing for typical possessions. When several real effects stack into an extreme raw probability, the compression saturates instead of letting the shot's outcome become a near-certainty — see [Node 2 Probability Compression](#node-2-probability-compression-diminishing-returns) below for why this replaced a set of per-input dampening attempts.

## Possession Decision Tree (Shared by CPU and GPU)

`cpp_engine/main.cpp` (CPU narrative engine, `PossessionEngine::simulate_possession`) and `cpp_engine/cuda_simulator.cu` (GPU kernel, the device `simulate_possession` function) implement the **exact same possession-by-possession decision tree**, so a matchup run through either engine reflects the same underlying tactical model — not two different simulators that happen to agree on paper. Every possession resolves through three explicit nodes plus a layer of situational and dynamic-state modifiers:

### Node 0 — Initiation
A real-tendency-weighted categorical draw decides how this touch develops: **Pick & Roll** (weighted by the offense's real team assists/game), **Isolation** (weighted by the shooter's own real usage rate), or **Spot-up** (weighted by the shooter's own real 3PT-attempt volume). This is where [Team Tactical Archetypes](#team-tactical-archetypes-macro-dna), the [Macro Tactical State Modifiers](#macro-tactical-state-modifiers), and [Game Momentum](#game-momentum--scoring-runs) apply their tendency biases, all composing on top of the same real-data baseline weights rather than overriding them.

### Node 1 — Paint Openness Score
A linear combination of real floor-spacing gravity, the shooter's own real drive gravity (FTA/game — how often they draw contact), and the defense's real rim-protection/help-IQ activity (each debuffed live by [Foul Trouble](#foul-trouble-tracking) when applicable), all relative to real neutral baselines. This score gates a discrete 3-way branch, not a smooth probability, so it doesn't need its own dampening.

### Node 2 — Branching Resolution
Paint Openness routes the possession into one of three outcomes — **Drive & Paint Finish** (defense got beat inside), **Drive & Kick** (an open catch-and-shoot 3), or **Contested ISO/Stifle** (defense held its ground) — each with its own real-data-anchored made-probability formula, before the [Probability Compression](#node-2-probability-compression-diminishing-returns) and final safety clamp.

### Conditional Foul & Free Throw Mechanics
Three real, independent checks replace a single flat foul-rate coin flip: **Offensive Foul/Charging** (drive attempts only, conditioned on the defense's real rim-protection presence), **Shooting Foul** (conditioned on the shooter's real drive gravity and the primary defender's real personal-fouls/game rate — 2 or 3 FTs, resolved via the shooter's real FT%, no game-clock advance), and **Non-Shooting/Penalty Foul** (a separate, additive check; below the real NBA 5-team-foul bonus threshold it's a dead ball with a reset 14s shot clock and the same team keeps the possession, at/above it the fouled player shoots bonus FTs).

### Team Tactical Archetypes (Macro DNA)
Each team is dynamically classified — **5-Out / Pace & Space**, **Pick & Roll Heavy**, **Paint Dominant / Post-Up**, or **Balanced** — from its own real season stats (3PT-attempt volume, team assist rate, drive/FTA rate), each compared to a real neutral baseline; the strongest deviation above a 10% threshold wins, otherwise the team is genuinely Balanced. An explicit override is also accepted via `--custom-roster` JSON (`team_a_archetype`/`team_b_archetype`). The resulting archetype biases Node 0's tendency weights and, for Paint Dominant, gives a direct Node 1 openness bump toward driving.

### Macro Tactical State Modifiers
Real in-game context reshapes tendencies and pace without touching per-player inputs:
* **Shot Clock Urgency** — an explicit, per-attempt real 24s shot clock (14s after an offensive rebound); under 5 seconds remaining, the read bypasses Pick & Roll and favors the ball-handler's own ISO/3PT game.
* **Desperation / Chase Mode** — a team trailing by a modest, still-catchable margin in the final 3 minutes shortens its own possessions and hunts 3s harder.
* **Protect Lead / Burn Clock** — the symmetric case: a team nursing a real, competitive late lead lengthens its possessions toward the full shot clock and leans on ball movement over quick 3PT attempts.

### CPU-Only: Physiological Stamina & Substitution System
`main.cpp`'s CPU narrative engine additionally models a full 48-minute fatigue/rotation cycle that the GPU batch engine deliberately does not (a performance tradeoff — see the GPU section below): workload drain scaled by real `usage_rate` relative to league average, positional workload multipliers ($1.25\times$ Bigs, $1.10\times$ Wings) for interior paint contact, accelerated bench recovery in a player's first 4 minutes of rest, and automatic like-for-like (`GUARD`/`WING`/`BIG`) substitutions once a stint exceeds 7 minutes or stamina drops to $\le 5.0$. This is also why the two engines' floor-spacing gravity inputs deliberately diverge in magnitude — see [Engine Tuning Parameters](#engine-tuning-parameters-mloptimization-driven-calibration) below.

## Dynamic State & Momentum Mechanics

Three mechanics give a single simulated game its own in-the-moment texture, on top of the two teams' fixed season-average inputs:

### Foul Trouble Tracking
Each team's real "rim anchor" — whoever has the single highest real blocks/game on the roster — gets a live, whole-game foul counter (not reset at quarter breaks, unlike team fouls). Cross 4 fouls and their contribution to the team's rim-protection aggregate is reduced 25%; cross 5 and it's reduced 45% — a real, individually-tracked defender playing more cautiously to avoid fouling out, creating a genuine, exploitable weakness for the offense to attack. Separately, a team's real per-quarter foul-bonus state (already tracked for the Conditional Foul mechanics above) also mildly debuffs team-wide help-defense activity once a team is in the penalty.

### Game Momentum & Scoring Runs
An exponentially-decayed per-team scoring accumulator (roughly a 2-possession memory) drives a real confidence/pace response: a team riding a hot stretch gets a modest boost to its isolation tendency and made-probability; a team getting run on slows its own pace to stop the bleeding rather than compounding the damage with rushed possessions. The memory window is deliberately short — an earlier, longer-memory version measurably became a proxy for "which team is already better" (the stronger team scores more, builds momentum, and gets rewarded further), reintroducing exactly the compounding overconfidence the rest of this section works to avoid; shortening the window to a genuine short-burst signal removed that feedback loop (verified: a controlled matchup's favorite win rate returned to its pre-momentum baseline once corrected).

### Game-to-Game Stochastic Variance
A real single-game shooting/defensive-intensity swing — a team can have a cold night or an engaged defensive one — is drawn ONCE per team per simulated game (not re-sampled every possession, which the law of large numbers would just wash out over ~100 shots) and held fixed for that whole game. Anchored to a real, commonly-documented ~6% relative team-level game-to-game efficiency swing, clamped to a plausible single-game range.

## Node 2 Probability Compression (Diminishing Returns)

Modeling ~100 possessions/game as independent per-shot draws is a well-known source of overconfidence: even a small, genuinely real per-shot probability edge compounds through the Central Limit Theorem into an aggregate win probability far more extreme than real single-game NBA variance ever shows. An early fix attempted to dampen each individual Node 1/2 input (spacing gravity, rim protection, playmaking, …) separately with its own `tanh`-based scale — this worked for isolated large gaps, but didn't address combinations of moderate effects stacking up, and interacted unpredictably with Game-to-Game Stochastic Variance (a Jensen's-inequality-style bias: dampening a stat *before* multiplying it by a symmetric per-game variance draw shifts its expected value, not just its variance).

The fix that stuck compresses the *fully-assembled* made-probability instead — every upstream effect already folded in — right before the safety clamp:

```
made_prob = league_avg_shot_prob + dampen(made_prob - league_avg_shot_prob, final_prob_compression_scale)
```

This is the one place in the decision tree where every real signal for that shot has already been combined into a single number, so compressing *that* directly prevents any *combination* of effects from snowballing, without having to guess which individual upstream input to distrust. Both engines' `dampen()` and the two tunable constants (`league_avg_shot_prob`, `final_prob_compression_scale`) live on [`EngineTuningParams`](#engine-tuning-parameters-mloptimization-driven-calibration) below.

## GPU Parallel Monte Carlo Engine (CUDA)

`cpp_engine/cuda_simulator.cu` runs the *entire* matchup Monte Carlo batch on the GPU — one CUDA thread per simulated game, running the same [Possession Decision Tree](#possession-decision-tree-shared-by-cpu-and-gpu) as the CPU engine, not one thread per stat reduction:

* **`GPURoster` / `GPUMatchupInput` (`cpp_engine/cuda_simulator.cuh`):** Structure-of-Arrays layout for exactly the two rosters being simulated (Team A and Team B), flattening each rotation player's real `fg3_pct`, `fg3a`, `fg_pct`, `ft_pct`, `drive_gravity`, `usage_rate`, `on_ball_defense_rating`, `personal_fouls_rate`, and positional `position_weight` into contiguous float arrays for a single `cudaMemcpy` per field, plus each team's precomputed aggregates (`rim_protection_best`, `help_defense_iq_avg`, `rim_anchor_idx` for [Foul Trouble](#foul-trouble-tracking), archetype, …) and starter/bench variants for Garbage Time.
* **Shared Game Clock:** both offenses draw from ONE shared 2,880-second (48-minute) clock and strictly alternate real possessions, rather than each being guaranteed a fixed possession count — the resulting possession count per team (targeting the real NBA-average ~99-100/team) is an emergent property of the shared clock, tying the two teams' scoring opportunities together as one real resource.
* **`monte_carlo_matchup_kernel`:** A `__global__` kernel that assigns **one full simulated 48-minute game per CUDA thread** (`idx = blockIdx.x * blockDim.x + threadIdx.x`, 100,000 threads per batch), each seeded with an independent `curandState` via `curand_init`. Per-thread state persists for that whole simulated game: [Game-to-Game Stochastic Variance](#game-to-game-stochastic-variance) draws, each team's [Foul Trouble](#foul-trouble-tracking) rim-anchor foul counter, each team's [Momentum](#game-momentum--scoring-runs) accumulator, and per-quarter team-foul counters. Floor-spacing gravity is computed deterministically from the on-court starters (the same Box-Cox `fg3_pct * (fg3a + 1)^0.2671` transform as the CPU engine, with no per-possession noise, so the two engines' expected shot-quality distribution stays identical) — all entirely on-device, with no console I/O and no cross-thread communication.
* **Host aggregation (`run_cuda_monte_carlo`):** Copies all 100,000 threads' final scores back in one batch and computes win probabilities, average scores, point differential, and standard deviation — then prints a summary block in the same style as the CPU batch report. Completes all 100,000 full games in **~85-140 ms** (varies by run/GPU; the development RTX 4070 Ti typically lands around 90-115 ms), with average scores holding in the realistic **~105-115 points/team** range.
* **Intrinsic, data-calibrated effects (always on / flag-triggered, but never hand-picked):** every coefficient below is read at compile time from `cpp_engine/calibrated_constants.h`, auto-generated by `calibrate_engine.py` (see "Native-Engine Calibration" below) — nothing here is a guessed constant.
  * **Defensive resistance (always on, no flag needed):** `GPURoster::def_rating` (real, from the backend's `/api/team_defense`) is fetched automatically, and `engine_calibration::kDefResistanceProbPerRating` shifts every possession's shot probability based on the actual defensive-rating differential between the two rosters.
  * **Home-court (`--home-a`/`--home-b`, no value):** applies a modern, hand-set **2.25-point** home-court equivalent (`kHomeCourtProbShiftOverride`) rather than `calibrated_constants.h`'s own tiny, statistically-insignificant fitted value — deliberately superseded because a modern regular-season home-court edge runs closer to 2-2.5 points than either the fitted ~0.8-point figure or the older ~3-point "textbook" one. Kept numerically in sync with the backend's own `kHomeCourtPointsPrior` (see the Market-Prior Safety Brake below) so the raw engine and the market-prior blend agree on this one real-world number.
  * **Schedule fatigue (`--b2b-a`/`--b2b-b`, no value):** applies `-engine_calibration::kFatigueProbPerRestDay` (a back-to-back is ~1 day of rest *disadvantage*) to whichever side is flagged.
* **Optional *external* override flags:** `--ml-margin <pts>` (ML-predicted point margin) and `--hot-hand-boost <mult>` (star usage/shooting multiplier for marquee games) — see the "Hybrid ML + Monte Carlo Pipeline" section below. Both default to neutral/off, and (since the ML feature set no longer includes defense/rest/home-court — see below) no longer double-count against the intrinsic effects above.
* **`--tuning-config <path.json>`:** loads an [`EngineTuningParams`](#engine-tuning-parameters-mloptimization-driven-calibration) override — see that section below.
* **Build target:** `cuda_simulator`, only built when the project is configured with `-DENABLE_CUDA=ON` (see [Build Instructions](#build-instructions) below). It links the same `nlohmann_json` and `cpr` dependencies as `simulator` and is compiled with `-O3` in Release/RelWithDebInfo/MinSizeRel configs (Debug uses unoptimized `/Od` to stay compatible with MSVC's `/RTC1` runtime checks).

## Engine Tuning Parameters (ML/Optimization-Driven Calibration)

Both engines group their real tunable decision-tree weights — Node 1's Paint Openness Score weights, Node 2's shot-quality spread and [Probability Compression](#node-2-probability-compression-diminishing-returns) constants, and the three [Dynamic State](#dynamic-state--momentum-mechanics) mechanics' own magnitudes — into one plain, mutable `EngineTuningParams` struct (`cpp_engine/main.cpp` for the CPU engine, `cpp_engine/cuda_simulator.cuh`/`.cu` for the GPU engine — mirrored field-for-field, `double` vs. `float` per each file's own convention) instead of scattered standalone constants. Both structs default-construct to this project's own already-verified calibration.

**`--tuning-config <path.json>`** (accepted by both `simulator.exe` and `cuda_simulator.exe`) loads a JSON file that overrides any subset of these fields — anything the file omits keeps its default:

```json
{
  "off_gravity_openness_weight": 0.25,
  "final_prob_compression_scale": 0.14,
  "momentum_hot_tendency_mult": 1.10,
  "foul_trouble_rim_protect_mult": 0.70
}
```

This is the hook for an external Python/ML optimization script — e.g. fitting these weights against real historical match results (the same `historical_games.csv` / `backtest_model.py` pipeline this project already uses for accuracy scoring) as a loss function, writing out a tuned parameter set, and handing it straight back to either engine with no source changes required. A malformed or unreadable `--tuning-config` file is a fatal error (not a silent partial apply), so an automated tuning loop gets an honest failure instead of quietly scoring against defaults it didn't ask for.

Three fields — `paint_finish_fg_pct_bonus`, `open_shot_bonus_multiplier`, `contested_iso_fg_pct_penalty` — are the one deliberate exception to the "identical defaults" rule: the GPU kernel's floor-spacing gravity has no per-possession fatigue/substitution model (see the CPU engine's own stamina system below), so it runs systematically higher than the CPU engine's fatigue-aware version, and these three constants are independently tuned in each engine's own struct defaults so both land in the real ~105-115 pts/team target on their own terms.

### Native-Engine Calibration (`calibrate_engine.py`)

Both `cuda_simulator.cu`'s kernel and `main.cpp`'s CPU possession loop bake in **intrinsic** effects calibrated from real data, rather than depending on the external `--ml-margin` override for baseline accuracy. `calibrate_engine.py` fits a genuinely interpretable model — plain OLS regression via `numpy` normal equations, not XGBoost gain-importances repurposed as coefficients (gain-importance tells you how much a *tree split* reduced loss, not "points of margin per rest day") — on real historical margins against real rest-days and real defensive-rating differentials (reusing `train_ml_model.py`'s exact feature pipeline):

```
 Term                         Coefficient    Std. Error    t-stat   Significant?
 intercept (home-court)           +0.8048        1.3490      0.60   NO -- weak/no evidence
 rest_advantage                   +1.5459        1.6438      0.94   NO -- weak/no evidence
 net_def_edge                     +2.2470        0.2944      7.63   yes (|t|>2)
```
(120 real 2024-25 games, R² = 0.334)

**Home-court via the intercept, not a regressor:** `fetch_real_nba_data.py` always puts the home team in the `team_a` slot, so a would-be `is_team_a_home` regressor would be constant (100% true) across every row — zero within-sample variation, unestimable. The regression's intercept (the expected margin when every other regressor is zero) *is* that constant home-team effect, so it's read off directly as the home-court coefficient instead.

Each coefficient is converted into the kernel's probability-shift units via the same points-to-probability derivation `--ml-margin` already used (`kMlMarginToProbShift`), then **automatically exported** every time the script runs into `cpp_engine/calibrated_constants.h` — a generated (`DO NOT EDIT BY HAND`) header, `#include`d by both `cuda_simulator.cu` and `main.cpp`, so the GPU batch and CPU narrative engines always share one calibration byte-for-byte:

```
namespace engine_calibration {
constexpr double kDefResistanceProbPerRating = 0.00488481;
constexpr double kFatigueProbPerRestDay      = 0.00336068;
constexpr double kHomeCourtProbShift         = 0.00174950;
}
```

* **`kDefResistanceProbPerRating` (defensive resistance)** — real, statistically significant (t=7.63), baked in with confidence. Always active: `GPURoster::def_rating`/`Team::def_rating` is fetched automatically from `/api/team_defense`, so a matchup's defensive quality shapes shot probability with zero external flags.
* **`kFatigueProbPerRestDay` (back-to-back fatigue, via `--b2b-a`/`--b2b-b`)** — a *rest-advantage* coefficient (positive = more rest is better); since a back-to-back is ~1 day of rest *disadvantage*, the engine applies it as `-kFatigueProbPerRestDay` when a team is flagged. Statistically insignificant at n=120 (fatigue is famously hard to detect even in full-season NBA data) — included at its fitted, honest near-zero magnitude rather than an invented one.
* **`kHomeCourtProbShift` (home-court, via `--home-a`/`--home-b`)** — likewise statistically insignificant at this sample size, included honestly rather than hard-coded to the textbook "+2.5 to 3.0 points." This fitted value is kept in the generated header for transparency and reproducibility, but the engine's *active* home-court effect is the hand-set `kHomeCourtProbShiftOverride` (**+2.25 points**, applied in Node 2 — see "Node 2 Probability Compression" below), chosen because n=120 isn't enough to trust a regression-fit home-court term over the well-established real-world NBA average.
* **Star concentration / "hot hand"** — showed no measurable intrinsic effect in earlier calibration runs, so it stays out of the intrinsic set entirely. `--hot-hand-boost` remains available as an explicit, opt-in hypothesis to test (see "Big Match Hot Hands" below) rather than a default the evidence doesn't support.

**Double-counting, resolved:** the external ML margin model (`train_ml_model.py`) previously included its own `def_rating`/`def_matchup_edge`/`rest_advantage`/`home_indicator` features — the *same* real-world signal the engine now applies intrinsically. Those four features have been **removed** from the model's feature set (14 → 10 computed, 9 actually fed to the model — see "Hybrid ML + Monte Carlo Pipeline" below for the full current count and why it's not simply "10"); the underlying `def_rating`/`rest_days` data is still computed in `train_ml_model.py`'s `load_training_table()` (as plain, non-`feat_`-prefixed columns) purely so `calibrate_engine.py` can reuse it for its own regression, but it no longer reaches the XGBoost feature vector.

**Verified result, at small sample size.** An earlier, small-sample check (24 real held-out games, before both the feature-count and swap-antisymmetry fixes described in "Hybrid ML + Monte Carlo Pipeline" below) found the baseline pass at 70.8% win/loss accuracy and the hybrid pass at 75.0% — those specific numbers are **superseded** and should not be quoted going forward; see "Benchmark Results: Baseline vs. Hybrid" further down for the current, much larger (245- and 84-game) verified figures and why they moved. The qualitative finding that motivated recording this in the first place still holds at the larger sample size too: the hybrid pass's Brier score and regular-season accuracy both still improve over baseline, meaning the ML layer is adding real, non-overlapping signal rather than re-applying what the engine already accounts for intrinsically. Re-run `python calibrate_engine.py` after fetching more games (`fetch_real_nba_data.py` with no `--max-games`) for a sturdier calibration; n=120 is still a small sample for the OLS fit above, and the generated header says so explicitly for every coefficient.

## Interactive Roster Customization & Player Import/Trade Pipeline

Before either engine runs, `cuda_simulator.exe` offers an optional customization stage (`customize_team_roster` in `cpp_engine/cuda_main.cpp`) for the two selected teams:

* **Add or Import Player:** search the *entire* fetched league by name (e.g. `"LaMelo Ball"`) and, on confirmation, transfer that player — stats, position, and all — from their current team onto the roster being edited (a genuine trade: they're removed from the source team, not duplicated). Falls back to a manual entry form (name, position, 3PT%, 3PT attempts, usage rate, minutes) if the search comes up empty or a custom/fictional player is wanted instead.
* **Remove a player** from a roster (e.g. to model an injury or a trade out).
* **Edit an existing player's stats** (3PT shooting, usage rate, minutes) in place, with blank input keeping the current value.

Edits are made directly on the in-memory roster shared by both engines, so a trade applied here automatically flows into **both** the CPU narrative/batch simulation and the GPU matchup kernel for that run — there's no separate data path to re-sync.

## Complete Pipeline Overview

End to end, the project now supports this workflow:

1. **FastAPI & PostgreSQL backend** — ingested player stats served from Postgres via `/api/players`.
2. **C++ CPU narrative simulation & batch Monte Carlo** — `simulator.exe` / `cuda_simulator.exe` play a single detailed possession-by-possession game, or run a silent 10,000-game CPU batch, for a chosen matchup.
3. **CUDA GPU massive matchup Monte Carlo** — the same `cuda_simulator.exe` run hands that matchup to the GPU for a 100,000-game parallel batch (`monte_carlo_matchup_kernel`), reporting win probabilities, average scores, and point differential in ~85-140 ms (typically 90-115 ms on an RTX 4070 Ti).
4. **Interactive Roster Customization & Player Import/Trade** — before either simulation runs, add/import, remove, or edit players on either roster so both engines evaluate the exact hypothetical lineup you want.
5. **Native-Engine Calibration** — `calibrate_engine.py` fits a plain OLS regression on real historical margins vs. real rest-days and defensive-rating differentials, and auto-generates `cpp_engine/calibrated_constants.h` — the single source of truth both C++ engines `#include` for their intrinsic defensive-resistance, fatigue, and home-court effects (see "Native-Engine Calibration" above).
6. **Hybrid ML + Monte Carlo calibration** — `fetch_real_nba_data.py` pulls real completed games, schedules, and rest days via `nba_api`; `train_ml_model.py` fits an XGBoost *regression* model (predicting a continuous point margin, not a win/loss class) on 9 model features — roster-derived spacing/shooting/usage, real per-game star-availability (from that game's actual box score), and the "Big Match Hot Hands / Star Momentum" proxy — to predict each matchup's expected point margin, then hands that margin to `cuda_simulator.exe` via `--ml-margin` and `--hot-hand-boost` (the two remaining *external* overrides, layered on top of the intrinsic calibration from step 5).
7. **Automated Model Backtesting** — `backtest_model.py` drives the compiled executable, with and without the external ML layer, against a held-out (never-trained-on) slice of real games to score both the pure-engine and hybrid model's real-world Win/Loss accuracy, Brier score, and point-differential error, side by side.
8. **REST API Wrapper (`POST /api/simulate`, `POST /api/simulate-ml`)** — `backend/api_simulation.py` bridges both compiled engines to any web/game client: pick `mode` (`gpu` statistical batch or `cpu` narrated single game), `roster_type` (`real`, fetched live, or `custom`, a submitted fantasy roster), and toggle fatigue/big-match/injuries/home-court — same engines, same flags, over plain JSON. `/api/simulate-ml` additionally runs step 6's trained margin model live for the requested matchup and returns it alongside the real simulated result, with display-only (never double-counted) contextual adjustments for fatigue/big-match/home-court — see "Hybrid ML Endpoint" below.

## Prerequisites & Environment

Tested on:
* **OS:** Windows 11
* **IDE / Compiler:** Visual Studio 2022 (MSVC toolset, "Desktop development with C++" workload)
* **CMake:** 4.4.3 (project supports CMake 3.18 through 4.x)
* **NVIDIA CUDA Toolkit:** 13.4, installed **with** the Visual Studio Integration component (required for `ENABLE_CUDA=ON` — this installs `CUDA 13.4.props/.targets` into VS2022's MSBuild `BuildCustomizations` folder)
* **Python:** 3.11 (backend, EDA dashboard, data ingestion)
* **Docker:** for the PostgreSQL database

GPU builds also require an NVIDIA GPU with a driver compatible with CUDA 13.4. `CMAKE_CUDA_ARCHITECTURES` defaults to `native`, i.e. CMake compiles for whatever GPU is detected on the build machine.

> **Note on `CUDA_PATH`:** if you install the CUDA Toolkit while a terminal or Visual Studio is already open, that shell won't see the `CUDA_PATH` environment variable the installer just set — open a fresh terminal before configuring with `ENABLE_CUDA=ON`. The CMake script also resolves the toolkit path independently and bakes it into the generated Visual Studio project, so a stale `CUDA_PATH` will not break a build that has already been configured once.

## Getting Started

### 1. Database Setup
```bash
docker compose up -d
```

### 2. Ingest Data & Run API
```bash
# Ingest stats
backend/venv/Scripts/python.exe backend/load_player_stats_raw.py

# Run FastAPI backend
backend/venv/Scripts/python.exe -m uvicorn backend.main:app --reload
```
Leave this running — both simulator executables fetch player data from `http://127.0.0.1:8000/api/players` at startup and will fail immediately if the API isn't reachable.

### 3. Build the C++ Engine

#### Build Instructions

CPU-only build (default — `ENABLE_CUDA` is `OFF` unless set):
```bash
cd cpp_engine
cmake -B build
cmake --build build --config Debug
```
This configures and builds only the `simulator` target.

GPU-enabled build (also builds `cuda_simulator`):
```bash
cd cpp_engine
cmake -B build -DENABLE_CUDA=ON
cmake --build build --config Debug
```
Use `--config Release` instead of `Debug` for an optimized GPU build (enables `-O3` on the CUDA kernel).

`ENABLE_CUDA` is a normal CMake cache variable, so once a build directory has been configured with it `ON`, subsequent `cmake --build build` calls will keep building both targets — no need to repeat the flag. To switch it back off, reconfigure with `cmake -B build -DENABLE_CUDA=OFF` or delete and regenerate the `build/` directory.

### 4. Running the Executables

With the FastAPI backend still running from step 2:

```bash
# CPU narrative simulator - plays and prints one full game
./build/Debug/simulator.exe

# GPU parallel Monte Carlo simulator (only present if built with ENABLE_CUDA=ON)
./build/Debug/cuda_simulator.exe
```

Swap `Debug` for `Release` (or whichever `--config` you built with) to match your build output directory.

`cuda_simulator.exe` also accepts optional CLI arguments for non-interactive/scripted runs:

```bash
./build/Release/cuda_simulator.exe <TEAM_A> <TEAM_B> [sim_mode]
# e.g.
./build/Release/cuda_simulator.exe MIN OKC 2
```
* `<TEAM_A>` / `<TEAM_B>` — team abbreviations (e.g. `MIN`, `OKC`, `NYK`); prompted for interactively if omitted, and falls back to `NYK` vs `SAS` if either abbreviation isn't found in the fetched roster.
* `[sim_mode]` — `1` for a detailed single-game play-by-play, `2` for a silent 10,000-game CPU batch; prompted for interactively if omitted.
* Regardless of arguments, the run always pauses once for **"Customize rosters before simulating? [y/N]"** — answer `y` to add/import, remove, or edit players (see [Interactive Roster Customization](#interactive-roster-customization--player-importtrade-pipeline) above) or just press Enter/EOF to proceed with the rosters as fetched.
* The GPU's 100,000-game matchup batch always runs afterward, using whatever rosters (customized or not) were used for the CPU stage.

## Hybrid ML + Monte Carlo Pipeline (`fetch_real_nba_data.py`, `train_ml_model.py`, `backtest_model.py`)

Three root-level Python scripts turn the GPU engine from a standalone Monte Carlo simulator into a data-calibrated hybrid predictor, and validate the result against real games:

1. **`fetch_real_nba_data.py`** pulls real, completed NBA games from `nba_api` (`leaguegamefinder.LeagueGameFinder`) and writes them to `historical_games.csv` — one row per game (home team as `team_a`, away as `team_b`), with `team_a`, `team_b`, `actual_winner`, `actual_score_a`, `actual_score_b`, `game_date`, `game_id`, and each team's `_rest_days` (computed from that team's full-season schedule, so even the earliest game kept after a `--max-games` slice still has an accurate prior-game date to diff against).
2. **`train_ml_model.py`** fits `xgb.XGBRegressor` on **9 model features** to predict a matchup's expected point margin (a regression target, not a class label): the original 7 roster-derived team-strength stats (3PT floor-spacing gravity, shooting volume/efficiency, usage concentration, a net-rating proxy), a real per-game **star-availability flag** (that game's actual `nba_api` box score — see below), and one of the two "Big Match Hot Hands" interaction proxies (see that subsection below). **Defensive rating, rest-days advantage, and home-court are deliberately *not* ML features** — `cpp_engine`'s engine now applies all three intrinsically (see "Native-Engine Calibration" above), so including them here too would double-count the same real-world signal; `load_training_table()` still computes the underlying `def_rating`/`rest_days` data as plain columns purely so `calibrate_engine.py` can reuse them for its own separate regression.

   **Swap-symmetry (`MODEL_FEATURE_NAMES` vs. the 10th, diagnostic-only feature):** a matchup's predicted margin must satisfy `predict_margin(A, B) == -predict_margin(B, A)` -- swapping which team is "Team A" and which is "Team B" (independent of which one is actually home, a separate `home_team` field) should exactly invert the prediction, the same way flipping a Vegas spread's sign flips who's favored. A tenth candidate feature, `big_match_indicator = min(team_a, team_b)`'s roster-strength proxy, is symmetric under that swap (`min()` doesn't care about argument order) — feeding it to XGBoost as a raw input broke that guarantee, since `historical_games.csv` records `team_a` as *always* the real home team, so the model could (and empirically did) pick up incidental correlation with that fixed labeling rather than genuine, swap-invariant team-quality signal. It's still computed for every row (kept as a diagnostic-only column, and as the multiplier inside the properly-antisymmetric `star_usage_concentration` interaction term below, which *is* a real model input), but it's excluded from `MODEL_FEATURE_NAMES`, the actual 9-column set XGBoost trains and predicts on. `predict_margin()`/`predict_margin_with_context()` additionally symmetrize their own output by construction (`(predict(A,B) - predict(B,A)) / 2`) rather than trusting the trained model to have learned an odd function on its own -- gradient-boosted trees have no such guarantee even when every input is antisymmetric, since nothing in ordinary training enforces it. `predict_win_probability()` is symmetrized the same way, so `P(A wins) + P(B wins) == 1.0` exactly under a swap too, not just approximately.

   Validated with a **chronological train/test split**: the model trains only on the earliest games and is scored only on the most recent, held-out ones, so the reported accuracy reflects predicting *forward* in time, the way it would actually be used.
3. **`backtest_model.py`** drives `cuda_simulator.exe` twice per held-out game — a **baseline** pass (`--home-a` for team_a, `--b2b-a`/`--b2b-b` for any team on 0 rest days -- the engine's intrinsic, data-calibrated effects, applied in *both* passes, not hybrid-only) and a **hybrid** pass that layers the two remaining *external* overrides on top (`--ml-margin`, and for qualifying games `--hot-hand-boost` — see [GPU Parallel Monte Carlo Engine](#gpu-parallel-monte-carlo-engine-cuda) above for how each biases the kernel's possession probabilities) — and reports, for both:

   * **Win/Loss accuracy** — did the team with the higher predicted win probability actually win?
   * **Brier score** — `(P(actual winner) − 1)²`, averaged across games (0 = fully confident in the correct winner, 1 = fully confident in the wrong one, 0.25 ≈ coin-flip).
   * **Point differential MAE** — mean absolute error between predicted and actual score margin.

   Only the ML model's held-out test games are ever backtested in hybrid mode — games used to train the ML component are excluded, since scoring the hybrid engine on games it already memorized would be leakage, not validation.

#### Running the full pipeline

```bash
# From the project root, with your virtual environment active:
.venv\Scripts\activate          # Windows PowerShell/cmd
# or: source .venv/Scripts/activate   # Git Bash
pip install -r requirements.txt   # numpy, pandas, requests, scikit-learn, xgboost, joblib

# 1. Fetch real games (writes historical_games.csv). Requires internet access
#    to the NBA stats API; no local backend/DB needed for this step.
python fetch_real_nba_data.py                       # full season, e.g. 2024-25
python fetch_real_nba_data.py --max-games 150        # or just the most recent N games

# 2. (Optional) train + evaluate the ML margin model on its own first.
#    Requires the FastAPI backend running (step 2 under Getting Started) --
#    features are computed from whatever roster /api/players currently serves.
python train_ml_model.py

# 3. Run the full baseline-vs-hybrid backtest (trains the ML model itself if
#    you skipped step 2). Requires the FastAPI backend + a built cuda_simulator.
python backtest_model.py
```

Useful flags (all three scripts accept `--csv`/`--out` paths; see `--help` on any of them for the full list):

```bash
# Bigger/smaller chronological holdout (default: last 20% of games)
python backtest_model.py --test-fraction 0.3

# Force leave-one-out CV instead of a chronological split (e.g. for a small/dateless CSV)
python backtest_model.py --split-mode loocv

# Only run the plain Monte Carlo baseline (skips the ML step entirely)
python backtest_model.py --mode baseline

# Cap how many held-out games actually get backtested (each one launches the
# executable twice in --mode both, so this bounds wall-clock time)
python backtest_model.py --limit 10
```

#### What the output looks like

`train_ml_model.py` first prints the chronological split, per-game holdout predictions, and gain-based feature importances (which stats — spacing, shooting volume, usage — actually drive the model, out of the 9 real `MODEL_FEATURE_NAMES` inputs, see "Swap-symmetry" above). `backtest_model.py` then prints a baseline pass, a hybrid pass, and a side-by-side comparison, all restricted to the same held-out games -- this is real output from the current deployed model, the full 245-game regular-season holdout:

```
========================================================
 HYBRID (ML + MONTE CARLO) vs BASELINE (MONTE CARLO ONLY)
 -- both scored on the same held-out (never-trained-on) games --
========================================================
 Metric                          Baseline      Hybrid       Delta
 Win/Loss accuracy                  64.9%       66.5%       +1.6%
 Average Brier score               0.2217      0.2087     -0.0130
 Point differential MAE            13.01       12.63       -0.38
--------------------------------------------------------
 (accuracy higher is better; Brier score and MAE lower is better)

 Hybrid win/loss accuracy vs 60% threshold: CROSSED (66.5%)
========================================================
```

`ml_model/hybrid_pipeline_benchmark.json` is the canonical, machine-readable source of truth for these numbers — regenerated by re-running `backtest_model.py` and `evaluate_playoff_holdout.py`, never hand-edited. Two genuinely different metrics live side by side in it, and conflating them is the single easiest way to misrepresent this pipeline's accuracy:

* **Margin-sign accuracy** (`ml_only_holdout_directional_accuracy`, also called `directional_accuracy` in `holdout_predictions.json`'s `eval_report`) — does the ML model's predicted margin have the same *sign* as the real margin? This is the ML component evaluated **in complete isolation**, no Monte Carlo engine involved at all. Across every configuration measured so far on the 245-game chronological regular-season holdout, this has ranged **~65.7%–66.9%**.
* **Full hybrid pipeline win/loss accuracy** (`hybrid_accuracy` under `regular_season_holdout`/`playoff_holdout`) — did the *simulated* game (engine + `--ml-margin` bias + `--hot-hand-boost` for flagged matchups) predict the actual winner? This is the number that matters end to end, and it's *not* simply "better" than the margin-sign number just because the engine adds its own real, independently-calibrated defense/rest/home-court effects on top (see "Native-Engine Calibration" above) — see the current numbers below.

#### Benchmark Results: Baseline vs. Hybrid (current deployed model)

| Holdout | n | Baseline (pure engine) | Hybrid (+ ML margin) | Delta |
|---|---:|---:|---:|---:|
| Regular season (245 held-out games, 2025-03-13 to 2025-04-13) | 245 | 64.9% | **66.5%** | **+1.6 pp** |
| Playoffs (84 real 2024-25 playoff games, never trained on) | 84 | 58.3% | **54.8%** | **-3.6 pp** |

*Baseline* is `cuda_simulator.exe` with only its intrinsic, data-calibrated effects active (defense/rest/home-court — no `--ml-margin`, `enable_shrinkage` off). *Hybrid* layers `train_ml_model.py`'s external margin prediction on top via `--ml-margin` (plus `--hot-hand-boost` for the 17/245 games with a real, programmatically-classified high-leverage flag — see "Big Match Hot Hands" below). These are 245- and 84-game samples respectively, not the 24-game window this README previously reported from — an order of magnitude more data, and the current honest picture.

**On real playoff games specifically, the hybrid layer currently scores *below* the pure engine baseline (54.8% vs. 58.3%).** This is reported plainly rather than hidden: `analyze_high_leverage_variance()` (a Welch's t-test, Bonferroni-corrected across all 10 computed features) found no statistically significant structural difference between high-leverage and ordinary games in this dataset, so there was never strong evidence the ML layer's season-aggregate signal would transfer especially well to playoff intensity — this result is consistent with that finding, not a surprise contradicting it. Don't read the hybrid layer as *always* helping; read it as helping on the much larger, more representative regular-season sample while adding no proven playoff-specific edge.

#### Why the numbers moved: the swap-antisymmetry fix

An earlier configuration of this same "clean, statistically-justified" pipeline (uniform training weight, Platt-calibrated probabilities — everything described above except one detail) measured **68.2% regular-season / 57.1% playoff** hybrid accuracy, and configurations before that reached as high as **68.6%** (`ml_model/hybrid_pipeline_benchmark.json`'s `prior_configurations` array keeps the full history, from 67.3% up through 68.6%, for comparison). All of those higher numbers came from a model that still included `big_match_indicator` as a raw, swap-*symmetric* feature (see "Swap-symmetry" above) — meaning `predict_margin(A, B)` was **not** guaranteed to equal `-predict_margin(B, A)`, a real, provable correctness bug: swapping which team a live API caller labels "Team A" vs. "Team B" would not reliably invert the predicted margin and win probability the way it must, since `historical_games.csv`'s home-team-is-always-team_a convention let that one feature ride along with an artifact of the data's labeling rather than genuine team-quality signal.

Removing it and enforcing exact swap-antisymmetry (`(predict(A,B) - predict(B,A)) / 2`, see "Swap-symmetry" above) cost **1.7 percentage points regular-season, 3.7 playoff** — a real, honestly-disclosed accuracy trade against the higher historical figures, made deliberately in favor of correctness rather than chasing the larger number. This is the intended reading of "~67.3%–68.6%" anywhere it appears in this project's history: it describes the *range of measured hybrid accuracy across the several configurations this pipeline has gone through*, not a single current, static claim — the currently deployed model measures 66.5%/54.8%, and that's the number to trust going forward.

**Architectural takeaway.** Both before and after this fix, the hybrid pass's win/loss accuracy improves over the pure-engine baseline on the regular-season holdout (+3.3pp pre-fix, +1.6pp post-fix) — the external ML margin model and the raw engine are still adding non-overlapping signal there, not fighting over the same variance. That decoupling (real defense/rest/home-court calibrated intrinsically via `calibrate_engine.py`, with `train_ml_model.py`'s feature set deliberately excluding those same signals — see "Double-counting, resolved" above) is what keeps the hybrid layer's regular-season contribution positive and stable across every recalibration this pipeline has gone through, even when the specific number moves.

#### Advanced analytics components

Four components beyond the original spacing/shooting/usage features. Three (fatigue, home-court, defense) are now **intrinsic engine calibrations** (see "Native-Engine Calibration" above), not ML features — only star availability remains an XGBoost input:

* **Schedule density / fatigue** — REAL data: any team on 0 rest days for a given game gets `--b2b-a`/`--b2b-b` passed to both `cuda_simulator.exe` and `simulator.exe`, applying `engine_calibration::kFatigueProbPerRestDay` (from `calibrated_constants.h`, fit by `calibrate_engine.py`) for that whole game. Not an ML feature — see "Native-Engine Calibration" above for why (double-counting).
* **Home-court advantage** — both engines take `--home-a`/`--home-b` (no value). The *active* effect is the hand-set `kHomeCourtProbShiftOverride` (**+2.25 points**, matching `backend/api_simulation.py`'s own `kHomeCourtPointsPrior`); `calibrate_engine.py`'s fitted `engine_calibration::kHomeCourtProbShift` is generated and kept for transparency but is not used at n=120 — see "Native-Engine Calibration" above. Not an ML feature, for the same double-counting reason as fatigue.
* **Defensive ratings & stylistic matchups** — REAL data: each team's season `def_rating` (points allowed per 100 possessions, via `nba_api`'s `leaguedashteamstats`) drives `engine_calibration::kDefResistanceProbPerRating` intrinsically in both engines, with no external flag needed at all (see `/api/team_defense`). Not an ML feature, for the same double-counting reason.
* **Injury / lineup impact (star availability)** — REAL, per-game data: for every historical game, `train_ml_model.py` checks that game's actual `nba_api` box score (`boxscoretraditionalv3`) to see whether each team's current top-usage player is recorded with any minutes. When a team's star is flagged absent for a specific game, that team's spacing/usage features for *that row only* are recomputed with the star excluded from the rotation (`team_features_excluding()`) — the "dynamic downgrade" — and an explicit `star_out_diff` feature is also fed to XGBoost. This one **does** stay an ML feature — there's no equivalent intrinsic engine mechanism for per-game injuries (that's what `--injured-a`/`--injured-b` is for, a separate, explicit roster edit). Box-score lookups are cached to `ml_model/star_availability_cache.json` after the first run. This only affects historical/backtest rows; ad hoc `--predict TEAM_A TEAM_B` calls have no specific game to check and assume full health.

#### "Big Match Hot Hands / Star Momentum" hypothesis

One more feature tests whether pre-game team-strength *context* — not just raw per-team stats — predicts margins better: `star_usage_concentration`, the star-usage gap between the two teams' best players, amplified in genuine "clash of contenders" matchups via a multiplier, `big_match_indicator = min(team_a, team_b)`'s roster-strength proxy. `big_match_indicator` itself is **not** a raw XGBoost input (see "Swap-symmetry" above — it's symmetric under a team_a/team_b swap and broke the model's required antisymmetry when fed in directly); it survives only as that multiplier, which — as an antisymmetric term (`star_usage_concentration`'s own diff) times a symmetric one — is itself correctly antisymmetric and a real, legitimate model input.

Which games get flagged for the separate `--hot-hand-boost 1.05` engine override is now a **real, programmatic classification** (`compute_high_leverage_flags()`), not the old `big_match_indicator`-at-median heuristic this README previously described: a game's actual NBA `game_id` season-type prefix (genuine playoff/Finals rounds), live end-of-season conference standings (real top-4-in-conference contenders), and the immediately preceding season's actual Conference Finals/Finals participants (rematch detection) — plus a small fixed-date table for the In-Season Tournament's knockout round. On the 245-game regular-season holdout this flags a genuinely small, real slice (17/245 games), replacing an earlier crude proxy that flagged over half the holdout (129/245) without any statistical basis. `analyze_high_leverage_variance()` (a Welch's t-test, Bonferroni-corrected across all 10 computed features, `ml_model/high_leverage_variance_report.json`) found no significant structural difference between these flagged games and ordinary ones on this dataset — which is exactly why `DEFAULT_HIGH_LEVERAGE_WEIGHT_MULTIPLIER` is `1.0` (uniform training weight, a genuine no-op) rather than an arbitrary upweighting: a discarded earlier `2.0x` value, chosen without this check, empirically made the full hybrid pipeline *worse* (see `ml_model/hybrid_pipeline_benchmark.json`'s `prior_configurations`). Treat `star_usage_concentration`'s feature importance as "does this proxy carry weight on this dataset," not confirmation of an actual in-game hot-hand effect. See `train_ml_model.py`'s module docstring for the full explanation.

A game whose team abbreviation isn't found (triggering the executable's NYK/SAS fallback), or that errors/times out, is reported inline and excluded from the summary metrics rather than aborting the whole run.

## REST API Wrapper (`POST /api/simulate`, `POST /api/simulate-ml`)

`backend/api_simulation.py` puts both compiled engines behind JSON endpoints, so a web app or game client never has to shell out or know about CLI flags directly. It's mounted into the existing FastAPI app in `backend/main.py`, so it's already live wherever `/api/players` is (`http://127.0.0.1:8000` by default). `/api/simulate` is the general-purpose engine wrapper described below; `/api/simulate-ml` layers the [Hybrid ML + Monte Carlo Pipeline](#hybrid-ml--monte-carlo-pipeline-fetch_real_nba_datapy-train_ml_modelpy-backtest_modelpy)'s trained margin model on top of it for live matchups — see its own subsection further down.

#### Two engine modes

* **`mode: "gpu"`** — runs `cuda_simulator.exe` in batch mode (always 100,000 threads) and returns aggregate stats: win probabilities, average scores, point differential and its std dev. `cuda_main.cpp` no longer runs a legacy, standalone 10,000-game CPU-only batch ahead of the real GPU kernel for this mode -- that pass's own output was never parsed by `parse_gpu_output()` (which only scans for the GPU-specific result block), so it was pure wasted work on every single API call; it's now skipped whenever invoked via the backend's `--sim-mode 2`, leaving only the real ~85-140 ms GPU batch (see [GPU Parallel Monte Carlo Engine](#gpu-parallel-monte-carlo-engine-cuda) above) plus request/subprocess overhead.
* **`mode: "cpu"`** — runs `simulator.exe` for one narrated, possession-by-possession game and returns the full play-by-play text plus the final score. Ideal for a "Game 7" cinematic viewing experience.

#### Market-Prior Safety Brake ("Prevent Overconfident Simulation")

The raw engine's own [Node 2 Probability Compression](#node-2-probability-compression-diminishing-returns) and [Game-to-Game Stochastic Variance](#game-to-game-stochastic-variance) meaningfully reduced blowout overconfidence, but a possession-level Monte Carlo model still treats ~100 shots/game as close to independent, which real single-game NBA variance never fully matches. `backend/api_simulation.py`'s `enable_shrinkage` toggle (`gpu` mode + `roster_type: "real"` only — it needs live real Pace/OFF_RATING/DEF_RATING to build a prior from) is a second, independent guardrail sitting *outside* the C++/CUDA engine entirely:

1. **`compute_market_prior()`** builds a structural (not raw-NET_RATING-diff) prior from each team's real, live Pace + OFF_RATING/DEF_RATING (`nba_api`'s `leaguedashteamstats`): average pace, each team's projected scoring rate tempered by the opponent's real defense, a real home-court margin (`kHomeCourtPointsPrior = 2.25`, kept numerically in sync with the C++ engine's own `kHomeCourtPointsOverride`), and a win probability via the normal CDF using `kRealMarginStdDev = 12.0` — the real single-game NBA margin variance the possession-level engine's own iid-ish model doesn't fully reproduce.
2. **`compute_dynamic_shrinkage_weight()`** decides how much of the final answer comes from the raw simulation vs. this prior — scaled by how statistically divergent the *structural prior's own* projected margin is (an exogenous signal, not the simulation's own confidence, which would be a self-referential trigger): a genuinely competitive matchup still leans on the simulation (`shrinkage_weight` request field sets this ceiling, default 0.45), a matchup with a real, large efficiency gap is pulled harder toward the prior (down to a fixed floor of 0.15).
3. **`apply_bayesian_shrinkage()`** is a plain, honestly-labeled convex blend (`weight · sim_prob + (1 - weight) · prior_prob`) — explicitly *not* a literal Bayesian posterior update over the 100,000 simulated games' win/loss counts (that sample size would make the prior's influence vanish, defeating the point; those games are correlated draws from one model, not independent real observations).

The response always exposes `raw_win_probability_a/b` alongside the blended `win_probability_a/b` (and `market_prior_probability_a/b`/`market_prior_expected_margin_a` when the brake fired), so a caller can audit the raw engine against the blended answer side by side. `enable_shrinkage: false` (the default) returns the pure, unblended raw simulation output — exactly what the underlying decision tree produces on its own.

#### Two roster sources

* **`roster_type: "real"`** — `team_a`/`team_b` are real team abbreviations (e.g. `"MIN"`, `"OKC"`); the executable fetches current rosters from `/api/players` itself, exactly as it always has.
* **`roster_type: "custom"`** — supply `custom_roster_a`/`custom_roster_b`, each a `{team_name, players: [...]}` payload (player fields: `player_name`, `position`, `fg3_pct`, `fg3a`, `usage_rate`, `min`). The API writes this to a server-generated temporary JSON file and passes it via the executables' `--custom-roster` flag (added to both `cuda_main.cpp` and `main.cpp` for this) -- no backend/database roster is touched, and the temp file is deleted once the run completes. `team_a`/`team_b` in the request body are ignored in this mode (the team names come from the roster payloads).

#### Advanced toggles

| Field | Effect | Notes |
|---|---|---|
| `enable_fatigue` + `team_a_rest_days`/`team_b_rest_days` | `--b2b-a`/`--b2b-b` for whichever team has 0 rest days | **both modes** -- intrinsic, data-calibrated effect |
| `home_team` (`"a"` / `"b"` / omitted) | `--home-a` / `--home-b` | **both modes** -- intrinsic, data-calibrated effect |
| `enable_injuries` + `injured_players_a`/`injured_players_b` | `--injured-a`/`--injured-b <names>` -- removes those players from the roster before simulating | **both modes** (real or custom roster) |
| `enable_big_match` + `hot_hand_boost` | `--hot-hand-boost <mult>` | gpu mode only -- external override |
| `ml_margin` (optional, beyond the task's 4 toggles) | `--ml-margin <pts>` -- pass a pre-computed margin (e.g. from `train_ml_model.py --predict`) | gpu mode only -- external override |
| `enable_shrinkage` + `shrinkage_weight` | Blends the raw win probability toward the [Market-Prior Safety Brake](#market-prior-safety-brake-prevent-overconfident-simulation) | gpu mode + `roster_type: "real"` only |

A toggle that doesn't apply to the chosen `mode` (e.g. `enable_big_match` with `mode="cpu"`) is never silently dropped -- it's skipped and reported in the response's `warnings` list. `enable_fatigue`, `home_team`, and `enable_injuries` now work in **both** modes: `simulator.exe`'s CPU narrative engine bakes in the same `calibrated_constants.h` coefficients as the GPU kernel (see "Native-Engine Calibration" above), and removing a named player from a roster is a plain roster edit either way. Only `enable_big_match`/`hot_hand_boost` and `ml_margin` remain gpu-only -- `simulator.exe` doesn't run `monte_carlo_matchup_kernel`, so it has no star-usage-boost or external-margin mechanism to apply them to.

#### Example requests

```bash
# GPU statistical batch, real rosters, full advanced-analytics stack
curl -X POST http://127.0.0.1:8000/api/simulate -H "Content-Type: application/json" -d '{
  "mode": "gpu", "roster_type": "real", "team_a": "MIN", "team_b": "OKC",
  "enable_fatigue": true, "team_b_rest_days": 0,
  "enable_big_match": true,
  "home_team": "a"
}'

# CPU narrated single game, a custom fantasy roster, with a star ruled out
curl -X POST http://127.0.0.1:8000/api/simulate -H "Content-Type: application/json" -d '{
  "mode": "cpu", "roster_type": "custom", "team_a": "-", "team_b": "-",
  "custom_roster_a": {"team_name": "Dream Team", "players": [
    {"player_name": "Star Guy", "position": "SG", "fg3_pct": 0.45, "fg3a": 9, "usage_rate": 32, "min": 36},
    {"player_name": "Big Guy", "position": "C", "fg3_pct": 0.10, "fg3a": 0.5, "usage_rate": 18, "min": 30}
  ]},
  "custom_roster_b": {"team_name": "Underdogs", "players": [
    {"player_name": "Role Player", "position": "PF", "fg3_pct": 0.32, "fg3a": 3, "usage_rate": 20, "min": 28}
  ]},
  "enable_injuries": true, "injured_players_a": ["Star Guy"]
}'
```

The response is always a flat JSON object: `status`, `mode`, `engine`, `resolved_team_a`/`resolved_team_b`, `parameters_applied` (exactly which toggles took effect), `warnings`, `players_removed`, `elapsed_ms`, the exact `command` argv that ran, mode-specific fields (`win_probability_a/b`, `average_score_a/b`, `point_differential*` for gpu; `final_score_a/b`, `winner`, `play_by_play` for cpu), and `raw_stdout` for full transparency/debugging.

#### Hybrid ML Endpoint (`POST /api/simulate-ml`)

A second endpoint, sharing the same FastAPI app, wraps the [Hybrid ML + Monte Carlo Pipeline](#hybrid-ml--monte-carlo-pipeline-fetch_real_nba_datapy-train_ml_modelpy-backtest_modelpy) for live matchups instead of historical backtesting: it computes `train_ml_model.py`'s expected-margin prediction for `team_a` vs. `team_b`, feeds it to `cuda_simulator.exe` via `--ml-margin` (always `mode="gpu"`, `roster_type="real"` — the trained model's features are keyed to real NBA abbreviations, with no equivalent for a custom fantasy roster), and returns **every** `/api/simulate` field (`base_response.model_dump()` — the real, simulated `win_probability_a/b`, average scores, point differential, etc.) alongside a set of `ml_*` fields describing the ML layer's own contribution:

| Field | Meaning |
|---|---|
| `ml_predicted_margin_a_minus_b` | The margin actually **displayed** — see "Contextual display adjustments" below |
| `ml_base_predicted_margin_a_minus_b` | The unadjusted margin actually sent to `--ml-margin` (what the simulation used) |
| `ml_context_adjustment_applied` | Whether the displayed margin differs from the base one this call |
| `ml_win_probability_a`/`_b` | The ML margin alone, run through the fitted Platt/Isotonic calibrator (`predict_win_probability()`) — **not** the same number as the top-level `win_probability_a`, which is the full simulated engine result |
| `ml_calibration_method` | `"platt"` or `"isotonic"` — whichever the training run's calibration comparison chose |
| `ml_confidence_score` | `abs(ml_win_probability_a - 0.5) * 2` — 0 at a toss-up, 1 at full confidence |
| `ml_only_holdout_accuracy`/`_n_games` | The static, pre-computed margin-sign accuracy from `holdout_predictions.json` (see "Benchmark Results" above) — **not** recomputed per request |
| `hybrid_pipeline_benchmark_accuracy`/`_n_games`/`_baseline_accuracy` | The static regular-season hybrid vs. baseline figures from `hybrid_pipeline_benchmark.json` |
| `hybrid_pipeline_playoff_accuracy`/`_n_games`/`_baseline_accuracy` | The static playoff-holdout figures from the same file |

The `ml_only_holdout_*`/`hybrid_pipeline_*` fields are **reference context, not live measurements** — they're read from `ml_model/`'s JSON artifacts once per server process (cached in-process) and describe the deployed *model's* validated track record, not anything computed for this specific request. Re-running `train_ml_model.py`/`backtest_model.py` while the server is up won't be reflected until it's restarted.

**Swap symmetry**, live: two calls with `team_a`/`team_b` and `home_team` fully swapped (identical rosters/injuries/rest-days otherwise) produce an exactly mirrored `ml_predicted_margin_a_minus_b` and complementary `ml_win_probability_a` (`P(A) + P(B) == 1.0`) — see "Swap-symmetry" above for why this is a guarantee, not a coincidence.

**Contextual display adjustments, without double-counting.** `apply_contextual_ml_adjustment()` can shift `ml_predicted_margin_a_minus_b` away from `ml_base_predicted_margin_a_minus_b` for three toggles, so the Hybrid ML panel visibly responds instead of looking static, while `--ml-margin` (what's actually simulated) always uses the unadjusted base value:

* **`enable_big_match`/`hot_hand_boost`** — a genuine re-inference through the trained model with both teams' `top_player_usage` scaled by `hot_hand_boost`, not an invented constant.
* **`enable_fatigue`** — the same real, calibrated `kRestAdvantageMarginPerDay` shift `calibrate_engine.py` fits, applied only when exactly one side is on a back-to-back (if both or neither are, the engine's own per-side penalty cancels out, so no adjustment is shown either).
* **`home_team`** — the same real `kHomeCourtPointsPrior` (2.25 pts) used by the Market-Prior Safety Brake below, signed toward whichever side is home.

All three are deliberately **display-only**: `cuda_simulator.exe` already applies hot-hand/fatigue/home-court intrinsically on its own simulation (see "GPU Parallel Monte Carlo Engine" and "Native-Engine Calibration" above), so re-adding any of them into `--ml-margin` would double-count the same real-world effect — precisely the failure mode `train_ml_model.py`'s feature set was already designed to avoid (see "Double-counting, resolved" above). This function's only job is making the *display* consistent with what's actually being simulated, never changing what's simulated.

```bash
curl -X POST http://127.0.0.1:8000/api/simulate-ml -H "Content-Type: application/json" -d '{
  "team_a": "BOS", "team_b": "MIA", "home_team": "a",
  "enable_fatigue": true, "team_b_rest_days": 0
}'
```

#### Safety notes

Every subprocess call passes a Python list to `subprocess.run` (`shell=True` is never used), so there is no shell-injection surface regardless of what a caller submits. Team/player names are additionally restricted to a safe character set by Pydantic validators. `roster_type="custom"` never lets a caller supply a filesystem path -- the temp file's path is always generated server-side. The executable path itself is always resolved from a fixed candidate list under `cpp_engine/build/`, never from the request. Every call has an enforced `timeout_seconds` (default 90s, capped at 300s) and returns a clean `504` on expiry rather than hanging the request.

See `backend/api_simulation.py`'s module docstring for the full design rationale, including why `enable_injuries` here is a direct, immediate roster edit rather than the offline `train_ml_model.py` box-score-driven auto-detection (which takes ~90 seconds and isn't suitable for a synchronous HTTP request).
