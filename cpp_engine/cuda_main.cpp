#define NOMINMAX

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include "cuda_simulator.cuh"

using json = nlohmann::json;

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
    // isolation/drive branches (see cuda_simulator.cu) instead of a flat,
    // player-agnostic constant. Defaults to a real league-average baseline
    // when a payload omits it (e.g. a --custom-roster player).
    double fg_pct = 0.46;

    // Real free-throw percentage (already a real empirical rate, no
    // conversion needed) -- fg_pct's sibling. Feeds the possession state
    // machine's and-1/shooting-foul free-throw resolution (see
    // cuda_simulator.cu). Defaults to a real league-average baseline.
    double ft_pct = 0.77;

    // Feeds the GPU possession state machine's defensive/playmaking/
    // rebounding mechanics (see cuda_simulator.cu). rim_protection_gravity/
    // help_defense_iq/playmaking_gravity/oreb_gravity/dreb_gravity are
    // genuine real-data proxies (real per-game blocks/steals/assists/
    // rebounds, computed in parse_player_from_json) unless a payload
    // explicitly overrides them; on_ball_defense_rating has no real
    // per-player tracking-stat source wired into this project yet, so it
    // stays at its neutral default.
    double on_ball_defense_rating = 50.0;
    double rim_protection_gravity = 0.5;
    double help_defense_iq = 1.0;
    double playmaking_gravity = 4.5;
    double oreb_gravity = 2.0;
    double dreb_gravity = 6.5;

    // Decision-Tree & Spacing-Driven Possession Engine's per-player "drive
    // gravity": real free-throw ATTEMPTS per game (a genuine, real-data
    // proxy for how often THIS player collapses a defense off the dribble
    // -- aggressive downhill drivers draw more contact), not a fabricated
    // rating. Feeds the Paint Openness Score (see cuda_simulator.cu).
    // Neutral default when a payload has no explicit override and no "fta".
    double drive_gravity_rating = 3.0;

    // Conditional Foul & Free Throw Mechanics Engine's per-player
    // foul-committing-tendency input when this player is the primary
    // DEFENDER (see cuda_simulator.cu): real personal fouls per game, a
    // genuine real-data proxy, not a fabricated rating. Neutral default
    // when a payload has no explicit override and no "pf".
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

// Builds one Player from a JSON object shaped like an /api/players record
// (player_name, position, min, usage_rate, fg3a, fg3_pct) -- shared by the
// real roster fetch (HTTP) and the --custom-roster JSON file path below, so
// both sources produce identically-initialized Players. `team_abbr` is
// applied to the player regardless of what (if anything) the JSON itself
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

    // Possession state machine's defensive/playmaking attributes (see
    // cuda_simulator.cu). An explicit override key always wins (future
    // real tracking-data integration); otherwise rim_protection_gravity/
    // help_defense_iq/playmaking_gravity derive from this player's real,
    // already-available season blk/stl/ast totals (divided by the same
    // games_played used above) rather than a fabricated number.
    // on_ball_defense_rating has no real per-player proxy available yet,
    // so it always takes the neutral default unless explicitly overridden.
    if (item.contains("rim_protection_gravity")) {
        p.rim_protection_gravity = item.value("rim_protection_gravity", p.rim_protection_gravity);
    } else if (item.contains("blk")) {
        // Real season blocks total (same season-cumulative convention as
        // "min"/"fg3a" above) -- 0.0 blk/game is a genuine real data point
        // for a non-shot-blocker, distinct from "no data at all" (a
        // --custom-roster payload with neither key), which keeps the
        // struct's neutral default instead.
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
        // Real season free-throw-attempts total (same season-cumulative
        // convention as "min"/"fg3a"/"blk" above).
        p.drive_gravity_rating = item.value("fta", 0.0) / games_played;
    }
    if (item.contains("personal_fouls_rate")) {
        p.personal_fouls_rate = item.value("personal_fouls_rate", p.personal_fouls_rate);
    } else if (item.contains("pf")) {
        // Real season personal-fouls total (same season-cumulative
        // convention as "min"/"fg3a"/"blk"/"fta" above).
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
};

struct Team {
    std::string team_abbreviation;
    std::vector<Player> roster;

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

    bool substitute_player(ActiveLineup& lineup, size_t court_index, int current_minute, bool verbose = true) {
        Player tired_player = lineup.on_court[court_index];
        PositionCategory needed_category = tired_player.get_category();

        int best_bench_idx = -1;
        double best_gravity = -1.0;

        for (size_t i = 0; i < roster.size(); ++i) {
            bool is_on_court = false;
            for (const auto& active : lineup.on_court) {
                if (active.player_name == roster[i].player_name) {
                    is_on_court = true;
                    break;
                }
            }

            if (!is_on_court && roster[i].remaining_stamina > 5.0 && roster[i].bench_rest_mins >= 2) {
                if (roster[i].get_category() == needed_category) {
                    double player_gravity = roster[i].spacing_index * get_position_weight(roster[i].position);
                    if (player_gravity > best_gravity) {
                        best_gravity = player_gravity;
                        best_bench_idx = static_cast<int>(i);
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

            if (verbose) {
                std::cout << "  [Min " << std::setw(2) << current_minute << "] "
                          << team_abbreviation << " SUB: "
                          << incoming.player_name << " [" << incoming.position << "] IN for "
                          << tired_player.player_name << " [" << tired_player.position << "]" << std::endl;
            }

            lineup.on_court[court_index] = incoming;
            return true;
        }
        return false;
    }
};

// Removes any roster player whose name matches (case-insensitive) one of
// `names` -- the non-interactive equivalent of the interactive roster
// customization stage's [2] Remove a player, used for the --injured-a /
// --injured-b CLI flags (e.g. from backend/api_simulation.py's enable_injuries).
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

    // Selects primary play initiator based on usage rate weights
    Player select_ball_handler(const std::vector<Player>& on_court) {
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
                return on_court[i];
            }
        }
        return on_court[0];
    }

    // Identifies the best scoring option on the floor based on offensive output
    Player get_best_shooter(const std::vector<Player>& on_court) {
        auto best_it = std::max_element(on_court.begin(), on_court.end(), [](const Player& a, const Player& b) {
            return (a.fg3_pct * a.fg3a) < (b.fg3_pct * b.fg3a);
        });
        return *best_it;
    }

    void simulate_possession(ActiveLineup& offense, ActiveLineup& defense, int& score_off, int& total_game_seconds,
                              bool verbose = true) {
        // Hard stop if game duration has already reached regulation limit (48 minutes = 2880 seconds)
        if (total_game_seconds >= 2880) return;

        // Random possession length between 14 and 22 seconds
        std::uniform_int_distribution<int> time_dist(14, 22);
        int possession_time = time_dist(rng);

        // Cap possession time to prevent exceeding the exact 48-minute mark (Buzzer-Beater adjustment)
        if (total_game_seconds + possession_time > 2880) {
            possession_time = 2880 - total_game_seconds;
            if (possession_time <= 0) return;
        }
        total_game_seconds += possession_time;

        int current_min = total_game_seconds / 60;
        int current_sec = total_game_seconds % 60;
        int quarter_elapsed_secs = total_game_seconds % 720;
        int remaining_quarter_secs = 720 - quarter_elapsed_secs;

        Player shooter;
        bool is_clutch_situation = (remaining_quarter_secs <= 10 && remaining_quarter_secs > 0);

        // In clutch/buzzer-beater situations, design the play for the team's best shooter
        if (is_clutch_situation) {
            shooter = get_best_shooter(offense.on_court);
        } else {
            shooter = select_ball_handler(offense.on_court);
        }

        // Select passer/assister from remaining on-court teammates
        Player assister = shooter;
        if (offense.on_court.size() > 1) {
            std::uniform_int_distribution<int> passer_dist(0, static_cast<int>(offense.on_court.size() - 1));
            do {
                assister = offense.on_court[passer_dist(rng)];
            } while (assister.player_name == shooter.player_name);
        }

        double offense_gravity = offense.get_current_gravity();
        double base_fg = (shooter.fg3_pct > 0.0) ? (0.32 + shooter.fg3_pct * 0.25) : 0.45;
        double spacing_boost = offense_gravity * 0.012;

        // Apply clutch defensive concentration and pressure multiplier if operating late-clock
        double contest_penalty = 0.08; // Base defensive contest penalty
        if (is_clutch_situation) {
            double clutch_pressure_factor = 1.8; // Defense collapses heavily on the star player
            contest_penalty *= clutch_pressure_factor;
        }

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double roll = dist(rng);

        // Clutch Success Probability Formula: (Base + Spacing) - (Contest * Pressure)
        double final_prob = std::clamp((base_fg + spacing_boost) - contest_penalty, 0.15, 0.70);

        // Determine if shot attempt is a 3-pointer or 2-pointer
        bool is_three = (shooter.fg3a >= 3.0 && dist(rng) > 0.4);

        int current_quarter = std::min(4, (current_min / 12) + 1);
        if (verbose) {
            std::cout << " [Q " << current_quarter << " | " << std::setw(2) << current_min << ":"
                      << std::setw(2) << std::setfill('0') << current_sec << std::setfill(' ') << "] "
                      << offense.team_name << (is_clutch_situation ? " [CLUTCH PLAY]: " : ": ")
                      << shooter.player_name << " handles the ball. ";
        }

        if (roll <= final_prob) {
            // Made field goal
            int pts = is_three ? 3 : 2;
            score_off += pts;
            if (verbose) {
                std::cout << (is_three ? "3-Pointer" : "2-Pointer") << " JUMPER is GOOD! (Assist by "
                          << assister.player_name << ") [+" << pts << " pts]" << std::endl;
            }
        }
        else if (roll > final_prob && roll <= final_prob + 0.08) {
            // Blocked shot event
            if (verbose) {
                Player blocker = defense.on_court[0];
                std::cout << "Shot BLOCKED at the rim by " << blocker.player_name << "!" << std::endl;
                std::cout << "   -> Defensive rebound secured by " << blocker.player_name << "." << std::endl;
            }
        }
        else {
            // Missed field goal and rebound contest
            if (verbose) {
                std::cout << (is_three ? "3-Pointer" : "2-Pointer") << " missed." << std::endl;
            }

            // Rebound resolution (~22% offensive rebound rate) -- always rolled,
            // regardless of verbosity, so batch-mode RNG draws stay identical to
            // the detailed single-game path.
            if (dist(rng) < 0.22) {
                std::uniform_int_distribution<int> off_reb_dist(0, static_cast<int>(offense.on_court.size() - 1));
                Player o_rebounds = offense.on_court[off_reb_dist(rng)];
                if (verbose) {
                    std::cout << "   -> Offensive rebound grabbed by " << o_rebounds.player_name << "!" << std::endl;
                }
            } else {
                std::uniform_int_distribution<int> def_reb_dist(0, static_cast<int>(defense.on_court.size() - 1));
                Player d_rebounds = defense.on_court[def_reb_dist(rng)];
                if (verbose) {
                    std::cout << "   -> Defensive rebound secured by " << d_rebounds.player_name << "." << std::endl;
                }
            }
        }
    }
};

// Final score from a completed game, used both to print the single-game
// result and to aggregate statistics across a batch of silent runs.
struct GameResult {
    int score_a = 0;
    int score_b = 0;
};

GameResult run_48min_simulation(Team& team_a, Team& team_b, bool verbose = true) {
    if (verbose) {
        std::cout << "\n========================================================" << std::endl;
        std::cout << "   POSSESSION & STAMINA SIMULATION: "
                  << team_a.team_abbreviation << " vs " << team_b.team_abbreviation << std::endl;
        std::cout << "========================================================\n" << std::endl;
    }

    ActiveLineup lineup_a = team_a.initialize_starters();
    ActiveLineup lineup_b = team_b.initialize_starters();

    PossessionEngine pos_engine;
    int score_a = 0;
    int score_b = 0;
    int total_game_seconds = 0;
    double league_avg_usage = 20.0;

    for (int minute = 1; minute <= 48; ++minute) {
        // 1. Simulate possessions alternating between teams until the minute concludes
        int minute_target_seconds = minute * 60;
        while (total_game_seconds < minute_target_seconds && total_game_seconds < 2880) {
            pos_engine.simulate_possession(lineup_a, lineup_b, score_a, total_game_seconds, verbose);
            if (total_game_seconds >= 2880) break;

            pos_engine.simulate_possession(lineup_b, lineup_a, score_b, total_game_seconds, verbose);
            if (total_game_seconds >= 2880) break;
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

        // 4. Substitution Triggers (Stint length limit >= 7 mins or depleted stamina <= 5.0)
        for (size_t i = 0; i < lineup_a.on_court.size(); ++i) {
            if (lineup_a.on_court[i].current_stint_mins >= 7 || lineup_a.on_court[i].remaining_stamina <= 5.0) {
                team_a.substitute_player(lineup_a, i, minute, verbose);
            }
        }
        for (size_t i = 0; i < lineup_b.on_court.size(); ++i) {
            if (lineup_b.on_court[i].current_stint_mins >= 7 || lineup_b.on_court[i].remaining_stamina <= 5.0) {
                team_b.substitute_player(lineup_b, i, minute, verbose);
            }
        }
    }

    if (verbose) {
        std::cout << "\n========================================================" << std::endl;
        std::cout << " FINAL SCORE: " << team_a.team_abbreviation << " " << score_a
                  << " - " << score_b << " " << team_b.team_abbreviation << std::endl;
        std::cout << "========================================================\n" << std::endl;
    }

    return GameResult{score_a, score_b};
}

namespace {

// Flattens a team's rotation (already sorted by minutes, most-used first)
// into the SoA layout the GPU kernel consumes. Capped to a realistic
// rotation size so garbage-time bench players don't dilute the GPU's
// usage-weighted shot distribution. `def_rating` (real, from
// /api/team_defense, or the neutral league-average default when unavailable
// -- e.g. a custom/fantasy roster) drives the kernel's intrinsic defensive-
// resistance effect; see GPURoster::def_rating and
// kDefResistanceProbPerRating in cuda_simulator.cu.
GPURoster build_gpu_roster(const Team& team, float def_rating, int archetype_override = -1,
                            size_t max_rotation_players = 9) {
    GPURoster roster;
    size_t count = std::min(max_rotation_players, team.roster.size());

    roster.fg3_pct.reserve(count);
    roster.fg3a.reserve(count);
    roster.fg_pct.reserve(count);
    roster.ft_pct.reserve(count);
    roster.drive_gravity.reserve(count);
    roster.usage_rate.reserve(count);
    roster.position_weight.reserve(count);
    roster.on_ball_defense_rating.reserve(count);
    roster.personal_fouls_rate.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const Player& p = team.roster[i];
        roster.fg3_pct.push_back(static_cast<float>(p.fg3_pct));
        roster.fg3a.push_back(static_cast<float>(p.fg3a));
        roster.fg_pct.push_back(static_cast<float>(p.fg_pct));
        roster.ft_pct.push_back(static_cast<float>(p.ft_pct));
        roster.drive_gravity.push_back(static_cast<float>(p.drive_gravity_rating));
        roster.usage_rate.push_back(static_cast<float>(p.usage_rate));
        roster.position_weight.push_back(static_cast<float>(get_position_weight(p.position)));
        roster.on_ball_defense_rating.push_back(static_cast<float>(p.on_ball_defense_rating));
        roster.personal_fouls_rate.push_back(static_cast<float>(p.personal_fouls_rate));
    }

    // Foul Trouble Tracking: this team's real "rim anchor" -- the single
    // uploaded-rotation player with the highest real rim_protection_gravity
    // (blocks/game), i.e. whoever actually drives rim_protection_best below.
    // Searched across the WHOLE uploaded rotation (not just the top-5 "on
    // court" slice below), since a team's real shot-blocking anchor can be
    // a bench big. See GPURoster::rim_anchor_idx's comment block in
    // cuda_simulator.cuh for how simulate_possession uses this.
    if (count > 0) {
        size_t best_idx = 0;
        for (size_t i = 1; i < count; ++i) {
            if (team.roster[i].rim_protection_gravity > team.roster[best_idx].rim_protection_gravity) {
                best_idx = i;
            }
        }
        roster.rim_anchor_idx = static_cast<int>(best_idx);
    }

    // Team-wide defensive/playmaking/rebounding aggregates for the GPU
    // possession state machine (see cuda_simulator.cu): mean real
    // help_defense_iq (steals/game), max real rim_protection_gravity
    // (blocks/game), mean real playmaking_gravity (assists/game), max real
    // oreb_gravity (offensive rebounds/game -- "best glass-crasher", same
    // convention as rim protection), and mean real dreb_gravity (defensive
    // rebounds/game) across the top-5 (by minutes -- team.roster is
    // already sorted that way) rotation, i.e. the same "on-court starters"
    // convention used for floor-spacing gravity.
    constexpr size_t kOnCourtCount = 5;
    size_t on_court_n = std::min(kOnCourtCount, count);
    if (on_court_n > 0) {
        double help_iq_sum = 0.0;
        double rim_protect_max = 0.0;
        double playmaking_sum = 0.0;
        double oreb_max = 0.0;
        double dreb_sum = 0.0;
        for (size_t i = 0; i < on_court_n; ++i) {
            help_iq_sum += team.roster[i].help_defense_iq;
            rim_protect_max = std::max(rim_protect_max, team.roster[i].rim_protection_gravity);
            playmaking_sum += team.roster[i].playmaking_gravity;
            oreb_max = std::max(oreb_max, team.roster[i].oreb_gravity);
            dreb_sum += team.roster[i].dreb_gravity;
        }
        roster.help_defense_iq_avg = static_cast<float>(help_iq_sum / static_cast<double>(on_court_n));
        roster.rim_protection_best = static_cast<float>(rim_protect_max);
        roster.playmaking_gravity_avg = static_cast<float>(playmaking_sum / static_cast<double>(on_court_n));
        roster.off_reb_gravity_best = static_cast<float>(oreb_max);
        roster.def_reb_gravity_avg = static_cast<float>(dreb_sum / static_cast<double>(on_court_n));
    }

    // "Bench" variants of the same five aggregates, over the bottom
    // kGarbageTimeBenchSize players of the UPLOADED rotation (i.e. the
    // lowest-minutes players within the already-capped `count`-sized GPU
    // array, not deep-bench players beyond it -- simulate_possession's
    // garbage-time index reversal only ever draws from this same uploaded
    // array). kGarbageTimeBenchSize must match cuda_simulator.cu's
    // constant of the same name.
    constexpr size_t kGarbageTimeBenchSize = 3;
    size_t bench_n = std::min(kGarbageTimeBenchSize, count);
    if (bench_n > 0) {
        double help_iq_sum = 0.0;
        double rim_protect_max = 0.0;
        double playmaking_sum = 0.0;
        double oreb_max = 0.0;
        double dreb_sum = 0.0;
        for (size_t i = count - bench_n; i < count; ++i) {
            help_iq_sum += team.roster[i].help_defense_iq;
            rim_protect_max = std::max(rim_protect_max, team.roster[i].rim_protection_gravity);
            playmaking_sum += team.roster[i].playmaking_gravity;
            oreb_max = std::max(oreb_max, team.roster[i].oreb_gravity);
            dreb_sum += team.roster[i].dreb_gravity;
        }
        roster.help_defense_iq_avg_bench = static_cast<float>(help_iq_sum / static_cast<double>(bench_n));
        roster.rim_protection_best_bench = static_cast<float>(rim_protect_max);
        roster.playmaking_gravity_avg_bench = static_cast<float>(playmaking_sum / static_cast<double>(bench_n));
        roster.off_reb_gravity_best_bench = static_cast<float>(oreb_max);
        roster.def_reb_gravity_avg_bench = static_cast<float>(dreb_sum / static_cast<double>(bench_n));
    }

    roster.num_players = static_cast<int>(count);
    roster.def_rating = def_rating;
    // Team Tactical Archetype (Macro DNA Layer): an explicit
    // --custom-roster override (see the comment above GPURoster::archetype
    // in cuda_simulator.cuh). archetype_override < 0 (the default -- no
    // override) leaves GPURoster::archetype/has_archetype_override at
    // their defaults, so run_cuda_monte_carlo dynamically derives the
    // archetype from this same roster's real stats instead.
    if (archetype_override >= 0) {
        roster.archetype = archetype_override;
        roster.has_archetype_override = true;
    }
    return roster;
}

// Engine Tuning Parameters (ML/Optimization-Driven Calibration) -- loads an
// optional `--tuning-config <path>` JSON file into `params`: any field
// present in the file overrides that field's default (see
// EngineTuningParams's comment block in cuda_simulator.cuh); fields the
// file omits are left untouched. Silently a no-op (keeps all defaults) if
// `path` is empty. A malformed/unreadable file is a fatal error (same
// convention as --custom-roster below) rather than a silent partial apply,
// so an external ML script gets an honest failure instead of quietly
// running against defaults it didn't ask for. Mirrors main.cpp's
// load_tuning_params() exactly (float fields here vs. double there).
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

std::string to_upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Optional explicit Team Tactical Archetype override, parsed from a
// --custom-roster JSON payload's "team_a_archetype"/"team_b_archetype"
// field (see cuda_simulator.cuh's GPURoster::archetype comment for the
// canonical enum this maps to -- duplicated here as plain ints since this
// file has no shared header with cuda_simulator.cu for it). Returns -1 (no
// override -- GPURoster::has_archetype_override stays false, and
// run_cuda_monte_carlo dynamically derives the archetype from this team's
// own real season stats instead) when the string is empty or unrecognized.
int parse_archetype_override(const std::string& raw) {
    std::string s = to_upper_copy(raw);
    if (s == "PACE_AND_SPACE" || s == "5_OUT" || s == "5-OUT") return 1;
    if (s == "PICK_AND_ROLL_HEAVY" || s == "PNR_HEAVY") return 2;
    if (s == "PAINT_DOMINANT" || s == "POST_UP") return 3;
    if (s == "BALANCED") return 0;
    return -1;
}

void print_available_teams(const std::map<std::string, Team>& league) {
    std::cerr << "Available team abbreviations:";
    for (const auto& pair : league) {
        std::cerr << " " << pair.first;
    }
    std::cerr << std::endl;
}

// ---------------------------------------------------------------------------
// Roster Customization Pipeline Stage
//
// Runs interactively between team selection and simulation. Edits are made
// directly on the `Team` objects living in the `league` map, so both the CPU
// narrative/batch engine (`run_48min_simulation` / `run_batch_simulations`)
// and the GPU matchup kernel (via `build_gpu_roster`, called after this
// stage in main()) automatically pick up whatever the user changed here --
// there is no separate data path to keep in sync.
// ---------------------------------------------------------------------------

std::string format_stat(double value, int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

// Reads one line as a double; an empty line (including piped/EOF input)
// keeps `default_value` so every prompt in this stage is skippable.
double prompt_double(const std::string& prompt, double default_value) {
    std::cout << prompt;
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) return default_value;
    try {
        return std::stod(input);
    } catch (...) {
        std::cout << "    (invalid number, keeping " << format_stat(default_value) << ")\n";
        return default_value;
    }
}

void print_roster(const Team& team) {
    std::cout << "\n  " << team.team_abbreviation << " roster:\n";
    std::cout << "  ---------------------------------------------------------------\n";
    std::cout << "   #   Player                 Pos   3PT%    3PA   USG%    MIN\n";
    for (size_t i = 0; i < team.roster.size(); ++i) {
        const Player& p = team.roster[i];
        std::cout << "  [" << std::setw(2) << i << "] "
                  << std::left << std::setw(22) << p.player_name << std::right
                  << std::setw(4) << p.position
                  << std::setw(8) << std::fixed << std::setprecision(3) << p.fg3_pct
                  << std::setw(7) << std::setprecision(1) << p.fg3a
                  << std::setw(7) << std::setprecision(1) << p.usage_rate
                  << std::setw(7) << std::setprecision(1) << p.target_mins << "\n";
    }
}

// [1b] Manual fallback: build a brand-new Player from scratch (a custom
// rookie/fictional player, used when the search comes up empty or the user
// explicitly asks to skip the import search).
void add_custom_player_manual(Team& team) {
    Player p;

    std::cout << "\n    New player name: ";
    std::getline(std::cin, p.player_name);
    if (p.player_name.empty()) {
        std::cout << "    Cancelled -- a player name is required.\n";
        return;
    }

    std::cout << "    Position (PG/SG/SF/PF/C) [SG]: ";
    std::getline(std::cin, p.position);
    if (p.position.empty()) p.position = "SG";

    p.fg3_pct = prompt_double("    3PT field goal % (0.0-1.0) [0.350]: ", 0.350);
    p.fg3a = prompt_double("    3PT attempts per game [3.0]: ", 3.0);
    p.usage_rate = prompt_double("    Usage rate (%) [20.0]: ", 20.0);
    p.target_mins = prompt_double("    Target minutes per game [20.0]: ", 20.0);

    p.team_abbreviation = team.team_abbreviation;
    p.remaining_stamina = p.target_mins;
    p.current_stint_mins = 0;
    p.bench_rest_mins = 10;
    p.spacing_index = calculate_spacing(p.fg3_pct, p.fg3a);

    team.roster.push_back(p);
    std::cout << "    Added " << p.player_name << " [" << p.position << "] to "
              << team.team_abbreviation << ".\n";
}

// One search hit: which team a matching player currently sits on, and their
// index within that team's roster vector.
struct PlayerLocation {
    std::string team_abbr;
    size_t index;
};

// Case-insensitive substring search for `query` across every roster in the
// fetched league (all teams, not just A/B), so a player can be pulled from
// any team, not only the two currently being simulated.
std::vector<PlayerLocation> find_players_by_name(const std::map<std::string, Team>& league,
                                                  const std::string& query) {
    std::string query_lower = to_lower_copy(query);
    std::vector<PlayerLocation> matches;

    for (const auto& pair : league) {
        const Team& t = pair.second;
        for (size_t i = 0; i < t.roster.size(); ++i) {
            if (to_lower_copy(t.roster[i].player_name).find(query_lower) != std::string::npos) {
                matches.push_back({t.team_abbreviation, i});
            }
        }
    }
    return matches;
}

// [1a] Search the whole league for an existing player by name and, on
// confirmation, transfer them onto `target_team`'s roster -- a genuine
// trade (removed from their original team), not a duplicate. Returns false
// (with no roster change) on a cancelled/empty/no-match search, so the
// caller can fall back to manual entry.
bool try_import_player(std::map<std::string, Team>& league, Team& target_team) {
    std::cout << "\n    Search player name (any team, e.g. \"LaMelo Ball\"): ";
    std::string query;
    std::getline(std::cin, query);
    if (query.empty()) {
        std::cout << "    Cancelled -- no search term entered.\n";
        return false;
    }

    std::vector<PlayerLocation> matches = find_players_by_name(league, query);
    if (matches.empty()) {
        std::cout << "    No player matching \"" << query << "\" found in the fetched league data.\n";
        return false;
    }

    size_t chosen = 0;
    if (matches.size() > 1) {
        std::cout << "    Multiple matches found:\n";
        for (size_t i = 0; i < matches.size(); ++i) {
            const Player& candidate = league.at(matches[i].team_abbr).roster[matches[i].index];
            std::cout << "      [" << i << "] " << candidate.player_name
                      << " (" << matches[i].team_abbr << ", " << candidate.position << ")\n";
        }
        std::cout << "    Select match # (default 0): ";
        std::string sel;
        std::getline(std::cin, sel);
        int sel_idx = sel.empty() ? 0 : std::atoi(sel.c_str());
        if (sel_idx < 0 || static_cast<size_t>(sel_idx) >= matches.size()) {
            std::cout << "    Invalid selection -- cancelled.\n";
            return false;
        }
        chosen = static_cast<size_t>(sel_idx);
    }

    const PlayerLocation loc = matches[chosen];
    if (loc.team_abbr == target_team.team_abbreviation) {
        std::cout << "    " << league.at(loc.team_abbr).roster[loc.index].player_name
                  << " is already on " << target_team.team_abbreviation << "'s roster.\n";
        return false;
    }

    Player found = league.at(loc.team_abbr).roster[loc.index];  // copy: source entry is erased below
    std::cout << "\n    Found: " << found.player_name << " [" << found.position << "] -- currently " << loc.team_abbr << "\n"
              << "      3PT%: " << format_stat(found.fg3_pct) << "   3PA: " << format_stat(found.fg3a, 1)
              << "   USG%: " << format_stat(found.usage_rate, 1) << "   MIN: " << format_stat(found.target_mins, 1) << "\n";
    std::cout << "    Transfer " << found.player_name << " from " << loc.team_abbr << " to "
              << target_team.team_abbreviation << "? [Y/n]: ";
    std::string confirm;
    std::getline(std::cin, confirm);
    bool confirmed = confirm.empty() || confirm[0] == 'y' || confirm[0] == 'Y';
    if (!confirmed) {
        std::cout << "    Cancelled.\n";
        return false;
    }

    Team& source_team = league.at(loc.team_abbr);
    source_team.roster.erase(source_team.roster.begin() + loc.index);

    found.team_abbreviation = target_team.team_abbreviation;
    target_team.roster.push_back(found);

    std::cout << "    Transferred " << found.player_name << " [" << found.position << "] to "
              << target_team.team_abbreviation << ".\n";
    return true;
}

// [1] Add or Import Player: tries the league-wide name search/import first
// (1a) and only drops to the manual custom-player form (1b) if the user
// asks for it outright, or the search comes up empty/cancelled.
void add_or_import_player(std::map<std::string, Team>& league, Team& team) {
    std::cout << "\n    [1a] Import existing player from another team (search by name)\n"
                 "    [1b] Create a custom player manually\n"
                 "    Enter choice (default 1a): ";
    std::string sub_choice;
    std::getline(std::cin, sub_choice);

    std::string sub_choice_lower = to_lower_copy(sub_choice);
    bool wants_manual = !sub_choice_lower.empty() &&
                         (sub_choice_lower[0] == '2' || sub_choice_lower.find('b') != std::string::npos);

    if (!wants_manual) {
        if (try_import_player(league, team)) return;
        std::cout << "    Switching to manual entry...\n";
    }
    add_custom_player_manual(team);
}

// [2] Remove/drop a player from `team`'s roster (e.g. an injury or trade out).
void remove_player(Team& team) {
    if (team.roster.empty()) {
        std::cout << "    Roster is empty -- nothing to remove.\n";
        return;
    }

    std::cout << "\n    Enter player # to remove (0-" << (team.roster.size() - 1) << "): ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) {
        std::cout << "    Cancelled.\n";
        return;
    }

    int idx = std::atoi(input.c_str());
    if (idx < 0 || static_cast<size_t>(idx) >= team.roster.size()) {
        std::cout << "    Invalid selection -- no player removed.\n";
        return;
    }

    std::cout << "    Removed " << team.roster[idx].player_name << " from "
              << team.team_abbreviation << ".\n";
    team.roster.erase(team.roster.begin() + idx);
}

// [3] Edit an existing player's stats (3PT shooting, usage, minutes, etc.).
// Blank input at any field keeps that player's current value.
void edit_player(Team& team) {
    if (team.roster.empty()) {
        std::cout << "    Roster is empty -- nothing to edit.\n";
        return;
    }

    std::cout << "\n    Enter player # to edit (0-" << (team.roster.size() - 1) << "): ";
    std::string input;
    std::getline(std::cin, input);
    if (input.empty()) {
        std::cout << "    Cancelled.\n";
        return;
    }

    int idx = std::atoi(input.c_str());
    if (idx < 0 || static_cast<size_t>(idx) >= team.roster.size()) {
        std::cout << "    Invalid selection -- no player edited.\n";
        return;
    }

    Player& p = team.roster[idx];
    std::cout << "    Editing " << p.player_name << " (blank keeps the current value):\n";

    p.fg3_pct = prompt_double("      3PT field goal % [" + format_stat(p.fg3_pct) + "]: ", p.fg3_pct);
    p.fg3a = prompt_double("      3PT attempts/gm [" + format_stat(p.fg3a, 1) + "]: ", p.fg3a);
    p.usage_rate = prompt_double("      Usage rate (%) [" + format_stat(p.usage_rate, 1) + "]: ", p.usage_rate);
    p.target_mins = prompt_double("      Target minutes [" + format_stat(p.target_mins, 1) + "]: ", p.target_mins);

    p.remaining_stamina = p.target_mins;
    p.spacing_index = calculate_spacing(p.fg3_pct, p.fg3a);

    std::cout << "    Updated " << p.player_name << ".\n";
}

// Interactive per-team customization loop. Keeps showing the roster + menu
// until the user picks [0] (or gives empty/invalid input, so piped/non-TTY
// runs fall through safely), then re-sorts by minutes so the edited roster
// feeds `initialize_starters()` and `build_gpu_roster()`'s rotation cap in
// the same most-used-first order as freshly-fetched data.
//
// Takes the whole `league` (not just `team_abbr`'s Team) because [1] Add or
// Import Player can pull a player from any other team's roster.
void customize_team_roster(std::map<std::string, Team>& league, const std::string& team_abbr) {
    Team& team = league.at(team_abbr);

    while (true) {
        print_roster(team);
        std::cout << "\n  Customize " << team.team_abbreviation << " roster:\n"
                     "    [0] Proceed with current roster as-is\n"
                     "    [1] Add or Import Player (search another team, or create custom)\n"
                     "    [2] Remove a player\n"
                     "    [3] Edit an existing player's stats\n"
                     "  Enter choice (default 0): ";
        std::string choice_input;
        std::getline(std::cin, choice_input);
        int choice = choice_input.empty() ? 0 : std::atoi(choice_input.c_str());

        if (choice == 1) {
            add_or_import_player(league, team);
        } else if (choice == 2) {
            remove_player(team);
        } else if (choice == 3) {
            edit_player(team);
        } else {
            break;
        }
    }
    team.sort_roster_by_minutes();
}

// Resolves the two team abbreviations to simulate: positional command-line
// arguments (positional_args[0]/[1] -- i.e. argv with the program name and
// any `--flag value` pairs like `--ml-margin` already stripped out) take
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

// Resets a roster to the same fresh state used right after parsing the API
// response (full stamina, no stint/rest history). Every trial in a Monte
// Carlo batch must start from this identical baseline -- otherwise stamina
// and substitution history would carry over between iterations and each
// simulated game would no longer be an independent, identically-distributed
// sample.
void reset_team_for_simulation(Team& team) {
    for (auto& p : team.roster) {
        p.remaining_stamina = p.target_mins;
        p.current_stint_mins = 0;
        p.bench_rest_mins = 10;
    }
}

struct BatchSimulationStats {
    int num_simulations = 0;
    int team_a_wins = 0;
    int team_b_wins = 0;
    int ties = 0;
    double avg_score_a = 0.0;
    double avg_score_b = 0.0;
    double avg_margin = 0.0;   // team_a_score - team_b_score, signed
    double margin_stddev = 0.0;
};

// Runs `num_simulations` silent 48-minute games and aggregates win/loss and
// scoring statistics. Teams are taken by value so the caller's `league`
// rosters are never mutated by the per-trial stamina reset/drain below.
BatchSimulationStats run_batch_simulations(Team team_a, Team team_b, int num_simulations) {
    BatchSimulationStats stats;
    stats.num_simulations = num_simulations;

    std::vector<int> margins;
    margins.reserve(num_simulations);

    double sum_a = 0.0;
    double sum_b = 0.0;

    for (int i = 0; i < num_simulations; ++i) {
        reset_team_for_simulation(team_a);
        reset_team_for_simulation(team_b);

        GameResult result = run_48min_simulation(team_a, team_b, /*verbose=*/false);

        sum_a += result.score_a;
        sum_b += result.score_b;
        margins.push_back(result.score_a - result.score_b);

        if (result.score_a > result.score_b) {
            ++stats.team_a_wins;
        } else if (result.score_b > result.score_a) {
            ++stats.team_b_wins;
        } else {
            ++stats.ties;
        }
    }

    stats.avg_score_a = sum_a / num_simulations;
    stats.avg_score_b = sum_b / num_simulations;

    double sum_margin = 0.0;
    for (int m : margins) sum_margin += m;
    stats.avg_margin = sum_margin / num_simulations;

    double sq_diff_sum = 0.0;
    for (int m : margins) {
        double diff = m - stats.avg_margin;
        sq_diff_sum += diff * diff;
    }
    stats.margin_stddev = std::sqrt(sq_diff_sum / num_simulations);

    return stats;
}

void print_batch_summary(const std::string& team_a_abbr, const std::string& team_b_abbr,
                          const BatchSimulationStats& stats, double elapsed_ms) {
    double win_pct_a = 100.0 * stats.team_a_wins / stats.num_simulations;
    double win_pct_b = 100.0 * stats.team_b_wins / stats.num_simulations;
    double tie_pct = 100.0 * stats.ties / stats.num_simulations;
    double games_per_sec = stats.num_simulations / (elapsed_ms / 1000.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n========================================================" << std::endl;
    std::cout << " MONTE CARLO BATCH RESULTS: " << team_a_abbr << " vs " << team_b_abbr << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << " Simulations run          : " << stats.num_simulations << std::endl;
    std::cout << " Elapsed time              : " << elapsed_ms << " ms (~" << games_per_sec << " games/sec)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << " " << team_a_abbr << " win probability        : " << win_pct_a << "% (" << stats.team_a_wins << " wins)" << std::endl;
    std::cout << " " << team_b_abbr << " win probability        : " << win_pct_b << "% (" << stats.team_b_wins << " wins)" << std::endl;
    if (stats.ties > 0) {
        std::cout << " Ties                       : " << tie_pct << "% (" << stats.ties << ")" << std::endl;
    }
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << " " << team_a_abbr << " average score          : " << stats.avg_score_a << std::endl;
    std::cout << " " << team_b_abbr << " average score          : " << stats.avg_score_b << std::endl;
    std::cout << " Avg point differential (" << team_a_abbr << "-" << team_b_abbr << ") : " << stats.avg_margin << std::endl;
    std::cout << " Point differential std dev : " << stats.margin_stddev << std::endl;
    std::cout << "========================================================\n" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    // 0. Pull the optional advanced-analytics flags out of argv (or
    //    `--flag=value` form for the value-taking ones) -- see
    //    train_ml_model.py / backtest_model.py, and cpp_engine/cuda_simulator.cu
    //    for what each does to the GPU kernel's possession probabilities:
    //      --ml-margin <pts>          ML-predicted Team A minus Team B point margin
    //      --hot-hand-boost <mult>    "Big Match Hot Hands" star usage/shooting multiplier
    //      --home-a / --home-b        flag (no value): that team has home court this game
    //                                 (applies the data-calibrated
    //                                 engine_calibration::kHomeCourtProbShift from
    //                                 calibrated_constants.h -- see cuda_simulator.cu)
    //      --b2b-a / --b2b-b          flag (no value): that team is on 0 days rest
    //      --custom-roster <path>     JSON file with team_a_name/team_a_roster/
    //                                 team_b_name/team_b_roster; when given, this
    //                                 completely replaces the HTTP-fetched league
    //                                 (no backend call at all) with the two rosters
    //                                 in the file, for api_simulation.py's "custom"
    //                                 roster_type. See the JSON shape example in
    //                                 backend/api_simulation.py.
    //      --injured-a / --injured-b <comma-separated names>
    //                                 removes those (case-insensitive, exact-match)
    //                                 named players from that team's roster before
    //                                 simulating -- the non-interactive equivalent
    //                                 of the roster customization stage's [2] Remove
    //                                 a player, for api_simulation.py's enable_injuries.
    //    Whatever's left over is the plain positional argument list (team_a,
    //    team_b, sim_mode) that the rest of main() already expects, so it
    //    works identically regardless of which flags are present or where
    //    they appear.
    std::vector<std::string> positional_args;
    double ml_margin_bias = 0.0;
    bool has_ml_margin_bias = false;
    double hot_hand_boost = 1.0;
    bool has_hot_hand_boost = false;
    bool is_team_a_home = false;
    bool is_team_b_home = false;
    bool is_team_a_b2b = false;
    bool is_team_b_b2b = false;
    std::string custom_roster_path;
    std::string tuning_config_path;
    std::vector<std::string> injured_a_names;
    std::vector<std::string> injured_b_names;
    static const std::string kMlMarginFlag = "--ml-margin";
    static const std::string kHotHandFlag = "--hot-hand-boost";
    static const std::string kHomeAFlag = "--home-a";
    static const std::string kHomeBFlag = "--home-b";
    static const std::string kB2bAFlag = "--b2b-a";
    static const std::string kB2bBFlag = "--b2b-b";
    static const std::string kCustomRosterFlag = "--custom-roster";
    static const std::string kTuningConfigFlag = "--tuning-config";
    static const std::string kInjuredAFlag = "--injured-a";
    static const std::string kInjuredBFlag = "--injured-b";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == kMlMarginFlag) {
            if (i + 1 < argc) {
                ml_margin_bias = std::atof(argv[++i]);
                has_ml_margin_bias = true;
            } else {
                std::cerr << "Warning: " << kMlMarginFlag << " given without a value; ignoring." << std::endl;
            }
        } else if (arg.rfind(kMlMarginFlag + "=", 0) == 0) {
            ml_margin_bias = std::atof(arg.c_str() + kMlMarginFlag.size() + 1);
            has_ml_margin_bias = true;
        } else if (arg == kHotHandFlag) {
            if (i + 1 < argc) {
                hot_hand_boost = std::atof(argv[++i]);
                has_hot_hand_boost = true;
            } else {
                std::cerr << "Warning: " << kHotHandFlag << " given without a value; ignoring." << std::endl;
            }
        } else if (arg.rfind(kHotHandFlag + "=", 0) == 0) {
            hot_hand_boost = std::atof(arg.c_str() + kHotHandFlag.size() + 1);
            has_hot_hand_boost = true;
        } else if (arg == kHomeAFlag) {
            is_team_a_home = true;
        } else if (arg == kHomeBFlag) {
            is_team_b_home = true;
        } else if (arg == kB2bAFlag) {
            is_team_a_b2b = true;
        } else if (arg == kB2bBFlag) {
            is_team_b_b2b = true;
        } else if (arg == kCustomRosterFlag) {
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
        } else {
            positional_args.push_back(arg);
        }
    }

    std::map<std::string, Team> league;
    std::string team_a_abbr;
    std::string team_b_abbr;
    // Real team defensive ratings (points allowed/100 possessions), keyed by
    // abbreviation, from the backend's /api/team_defense (see backend/main.py).
    // Stays empty for --custom-roster runs (no real team to look up), in
    // which case build_gpu_roster()/GPURoster::def_rating's neutral
    // league-average default applies -- the intrinsic defensive-resistance
    // term becomes a no-op for a fantasy matchup, which is the honest
    // behavior (there is no real defense to calibrate against).
    std::map<std::string, float> def_ratings;
    // Optional explicit Team Tactical Archetype override (see
    // parse_archetype_override above) -- -1 means "no override", i.e.
    // build_gpu_roster()/run_cuda_monte_carlo dynamically derive it from
    // this team's own real season stats below. Only settable via
    // --custom-roster JSON (there's no real per-team "archetype" field
    // from /api/players to pull from for a live roster -- it's always
    // derived in that case).
    int team_a_archetype_override = -1;
    int team_b_archetype_override = -1;

    if (!custom_roster_path.empty()) {
        // 1-4 (custom roster path). No live /api/players call: both rosters
        // come entirely from the JSON file, shaped as
        //   {"team_a_name": "...", "team_a_roster": [ {player_name, position,
        //    min, usage_rate, fg3a, fg3_pct}, ... ], "team_b_name": "...",
        //    "team_b_roster": [...], "team_a_def_rating": <float, optional>,
        //    "team_b_def_rating": <float, optional>}
        // -- exactly what api_simulation.py writes for roster_type="custom".
        // team_a_def_rating/team_b_def_rating are OPTIONAL: when a custom
        // roster is built from a real team's real players (the common "trade
        // a real player" workflow), api_simulation.py forwards that team's
        // real, live /api/team_defense rating here so the intrinsic
        // defensive-resistance effect (kDefResistanceProbPerRating) stays
        // active -- exactly as it would for a non-custom run of the same
        // team -- instead of silently going neutral just because the roster
        // happens to be custom. A genuinely fictional/fantasy team (no real
        // team behind it) simply omits the field and gets the same neutral
        // GPURoster::def_rating default as always.
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
            def_ratings[team_a_abbr] = custom_json.value("team_a_def_rating", GPURoster{}.def_rating);
        }
        if (custom_json.contains("team_b_def_rating")) {
            def_ratings[team_b_abbr] = custom_json.value("team_b_def_rating", GPURoster{}.def_rating);
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
        // 1. Fetch player data from the FastAPI backend -- once, feeding both
        //    the CPU narrative simulation and the GPU Monte Carlo simulation
        //    below.
        cpr::Response r = cpr::Get(cpr::Url{"http://127.0.0.1:8000/api/players"});
        if (r.status_code != 200) {
            std::cerr << "API Error: Status code " << r.status_code << std::endl;
            return 1;
        }

        // 1b. Fetch real team defensive ratings -- feeds the GPU/CPU engines'
        //     intrinsic defensive-resistance effect (see
        //     kDefResistanceProbPerRating in cuda_simulator.cu). Non-fatal if
        //     unavailable: every team simply keeps GPURoster::def_rating's
        //     neutral league-average default, so the simulation still runs,
        //     just without this particular calibrated effect.
        cpr::Response def_r = cpr::Get(cpr::Url{"http://127.0.0.1:8000/api/team_defense"});
        if (def_r.status_code == 200) {
            json def_json = json::parse(def_r.text);
            for (const auto& item : def_json) {
                def_ratings[item.value("team_abbreviation", "")] = item.value("def_rating", 113.0f);
            }
        } else {
            std::cerr << "Warning: could not fetch /api/team_defense (status "
                       << def_r.status_code << ") -- intrinsic defensive resistance disabled this run."
                       << std::endl;
        }

        json players_json = json::parse(r.text);

        // 2. Parse the JSON into team rosters, shared by the CPU narrative
        //    engine and (once flattened per-team below) the GPU matchup kernel.
        for (const auto& item : players_json) {
            std::string team_abbr = item.value("team_abbreviation", "FA");
            Player p = parse_player_from_json(item, team_abbr);

            league[team_abbr].team_abbreviation = team_abbr;
            league[team_abbr].roster.push_back(p);
        }

        for (auto& pair : league) {
            pair.second.sort_roster_by_minutes();
        }

        // 4. Resolve which two teams to simulate (CLI args, prompt, or fallback).
        resolve_matchup(positional_args, league, team_a_abbr, team_b_abbr);
    }

    // 4a. Injury flags (--injured-a / --injured-b): remove named players from
    //     the resolved rosters before anything else runs, for either roster source.
    if (league.count(team_a_abbr)) remove_named_players(league[team_a_abbr], injured_a_names);
    if (league.count(team_b_abbr)) remove_named_players(league[team_b_abbr], injured_b_names);

    // 4b. Roster Customization Pipeline Stage -- runs after team selection
    //     but before either simulator so a trade/injury/hot-hand edit here
    //     flows into both the CPU and GPU engines below unchanged.
    if (league.count(team_a_abbr) && league.count(team_b_abbr)) {
        std::cout << "\nCustomize rosters before simulating (add/remove/edit players)? [y/N]: ";
        std::string customize_input;
        std::getline(std::cin, customize_input);
        bool do_customize = !customize_input.empty() &&
                             (customize_input[0] == 'y' || customize_input[0] == 'Y');

        if (do_customize) {
            std::cout << "\n========================================================" << std::endl;
            std::cout << " ROSTER CUSTOMIZATION: " << team_a_abbr << " vs " << team_b_abbr << std::endl;
            std::cout << "========================================================" << std::endl;

            std::cout << "\n-- " << team_a_abbr << " --" << std::endl;
            customize_team_roster(league, team_a_abbr);

            std::cout << "\n-- " << team_b_abbr << " --" << std::endl;
            customize_team_roster(league, team_b_abbr);
        }
    }

    // 5. Resolve simulation mode: an optional 3rd positional CLI arg ("1" or
    //    "2"), or an interactive prompt if that's absent/invalid.
    int sim_mode = (positional_args.size() >= 3) ? std::atoi(positional_args[2].c_str()) : 0;
    if (sim_mode != 1 && sim_mode != 2) {
        std::cout << "\nSelect simulation mode:\n"
                     "  [1] Detailed single-game simulation (full play-by-play)\n"
                     "  [2] Batch Monte Carlo simulation (10,000 silent CPU games)\n"
                     "Enter choice (default 1): ";
        std::string mode_input;
        std::getline(std::cin, mode_input);
        sim_mode = (!mode_input.empty() && mode_input[0] == '2') ? 2 : 1;
    }

    // 6. Run the CPU narrative simulation in the chosen mode.
    //
    // sim_mode == 2 (the backend's --sim-mode 2, used for every /api/simulate
    // and /api/simulate-ml GPU-mode request) used to also run a legacy,
    // standalone 10,000-game CPU-only batch here (run_batch_simulations /
    // print_batch_summary) before step 7's real GPU Monte Carlo batch below.
    // That legacy batch's "MONTE CARLO BATCH RESULTS" output was never
    // parsed by backend/api_simulation.py's parse_gpu_output() (it only
    // scans for the separate GPU_BLOCK_MARKER block step 7 prints) -- so it
    // was pure wasted CPU work on every single API call. Skipped here; the
    // sim_mode == 1 interactive single-game path (run_48min_simulation) is
    // unaffected.
    if (league.count(team_a_abbr) && league.count(team_b_abbr)) {
        if (sim_mode != 2) {
            run_48min_simulation(league[team_a_abbr], league[team_b_abbr]);
        }
    } else {
        std::cerr << "Fatal: default teams NYK/SAS not found in fetched roster data. "
                     "Skipping CPU simulation." << std::endl;
    }

    // 7. Run the full parallel match-up Monte Carlo simulation on the GPU for
    //    the exact same matchup (team_a_abbr vs team_b_abbr) chosen above,
    //    rather than generic/league-wide data.
    if (league.count(team_a_abbr) && league.count(team_b_abbr)) {
        GPUMatchupInput matchup;
        matchup.team_a_name = team_a_abbr;
        matchup.team_a = build_gpu_roster(league[team_a_abbr],
            def_ratings.count(team_a_abbr) ? def_ratings[team_a_abbr] : GPURoster{}.def_rating,
            team_a_archetype_override);
        matchup.team_b_name = team_b_abbr;
        matchup.team_b = build_gpu_roster(league[team_b_abbr],
            def_ratings.count(team_b_abbr) ? def_ratings[team_b_abbr] : GPURoster{}.def_rating,
            team_b_archetype_override);
        matchup.ml_margin_bias = ml_margin_bias;
        matchup.has_ml_margin_bias = has_ml_margin_bias;
        matchup.hot_hand_boost = hot_hand_boost;
        matchup.has_hot_hand_boost = has_hot_hand_boost;
        matchup.is_team_a_home = is_team_a_home;
        matchup.is_team_b_home = is_team_b_home;
        matchup.is_team_a_b2b = is_team_a_b2b;
        matchup.is_team_b_b2b = is_team_b_b2b;
        // Engine Tuning Parameters -- optional external ML/optimization
        // override (see EngineTuningParams's comment block in
        // cuda_simulator.cuh). A no-op (defaults apply) when
        // --tuning-config isn't given.
        load_tuning_params(tuning_config_path, matchup.tuning);

        run_cuda_monte_carlo(matchup);
    } else {
        std::cerr << "Fatal: \"" << team_a_abbr << "\" and/or \"" << team_b_abbr
                   << "\" not found in fetched roster data. Skipping GPU simulation." << std::endl;
    }

    return 0;
}
