-- Unified Schema for NBA Matchup & Spacing Simulator
-- Supports teams, player metadata/positions, and raw box-score stats for C++ / FastAPI integration.

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    abbreviation CHAR(3) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    team_id INTEGER REFERENCES teams(team_id),
    position VARCHAR(5),
    height_inches INT,
    weight_lbs INT,
    three_point_pct NUMERIC(4, 3),
    usage_rate NUMERIC(4, 1),
    spacing_impact_score NUMERIC(5, 2),
    perimeter_defense_score NUMERIC(5, 2)
);

CREATE TABLE IF NOT EXISTS player_stats_raw (
    player_id INTEGER PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    team_id BIGINT,
    team_abbreviation CHAR(3),
    age NUMERIC(4, 1),
    gp INT,
    min NUMERIC(7, 2),
    fgm NUMERIC(6, 1),
    fga NUMERIC(6, 1),
    fg_pct NUMERIC(5, 3),
    fg3m NUMERIC(6, 1),
    fg3a NUMERIC(6, 1),
    fg3_pct NUMERIC(5, 3),
    ftm NUMERIC(6, 1),
    fta NUMERIC(6, 1),
    ft_pct NUMERIC(5, 3),
    oreb NUMERIC(6, 1),
    dreb NUMERIC(6, 1),
    reb NUMERIC(6, 1),
    ast NUMERIC(6, 1),
    tov NUMERIC(6, 1),
    stl NUMERIC(6, 1),
    blk NUMERIC(6, 1),
    pf NUMERIC(6, 1),
    pts NUMERIC(7, 1),
    plus_minus NUMERIC(7, 1)
);

CREATE INDEX IF NOT EXISTS idx_players_team_id ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_raw_team_id ON player_stats_raw(team_id);