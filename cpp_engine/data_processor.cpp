#include "data_processor.hpp"

#include <cmath>
#include <iostream>

namespace {
// Mirrors calculate_spacing() in cuda_main.cpp / main.cpp: the empirical
// Box-Cox floor-spacing transform (lambda ~= 0.2671) applied per player.
float box_cox_spacing_gravity(float fg3_pct, float fg3a) {
    return fg3_pct * std::pow(fg3a + 1.0f, 0.2671f);
}
} // namespace

FlattenedPlayerData flatten_player_data(const nlohmann::json& json_data) {
    FlattenedPlayerData data;

    const nlohmann::json* players_array = nullptr;
    if (json_data.is_array()) {
        players_array = &json_data;
    } else if (json_data.contains("players") && json_data["players"].is_array()) {
        // Handles APIs that wrap the roster in an outer object, e.g. {"players": [...]}.
        players_array = &json_data["players"];
    } else {
        std::cerr << "[DataProcessor] Error: player data is not an array.\n";
        return data;
    }

    data.num_players = static_cast<int>(players_array->size());

    data.player_ids.reserve(data.num_players);
    data.x_coords.reserve(data.num_players);
    data.y_coords.reserve(data.num_players);
    data.three_pt_gravity.reserve(data.num_players);
    data.speed_ratings.reserve(data.num_players);

    for (const auto& player : *players_array) {
        float fg3_pct = player.value("fg3_pct", 0.0f);
        float fg3a = player.value("fg3a", 0.0f);

        data.player_ids.push_back(player.value("player_id", 0));
        // The API does not yet expose on-court tracking coordinates or a
        // speed metric, so these default to neutral placeholders. Keeping
        // them in the SoA layout means GPU kernels can already consume the
        // full struct without further changes once that data lands.
        data.x_coords.push_back(player.value("x", 0.0f));
        data.y_coords.push_back(player.value("y", 0.0f));
        data.three_pt_gravity.push_back(box_cox_spacing_gravity(fg3_pct, fg3a));
        data.speed_ratings.push_back(player.value("speed", 1.0f));
    }

    std::cout << "[DataProcessor] Flattened " << data.num_players
               << " players into SoA layout for GPU transfer.\n";
    return data;
}
