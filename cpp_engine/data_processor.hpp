#pragma once

#include <vector>
#include <nlohmann/json.hpp>

// Structure-of-Arrays layout for player data, sized for coalesced GPU memory
// access once transferred into CUDA device buffers.
struct FlattenedPlayerData {
    std::vector<int> player_ids;
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    std::vector<float> three_pt_gravity;
    std::vector<float> speed_ratings;
    int num_players = 0;
};

// Parses the raw player JSON returned by the /api/players endpoint into the
// SoA layout above, ready for a single cudaMemcpy per field.
FlattenedPlayerData flatten_player_data(const nlohmann::json& json_data);
