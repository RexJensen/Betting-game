import pandas as pd

from features.pregame_features_pipeline import build_pregame_features, validate_point_in_time_integrity


def test_build_and_validate_point_in_time():
    games = pd.DataFrame([
        {
            "game_id": "g1",
            "game_datetime": "2025-07-10T23:00:00Z",
            "home_team_id": "NYY",
            "away_team_id": "BOS",
            "home_starting_pitcher_id": "p1",
            "away_starting_pitcher_id": "p2",
        },
        {
            "game_id": "g0",
            "game_datetime": "2025-07-09T23:00:00Z",
            "home_team_id": "NYY",
            "away_team_id": "BOS",
            "home_starting_pitcher_id": "p1",
            "away_starting_pitcher_id": "p2",
        },
    ])
    team_games = pd.DataFrame([
        {"game_id": "g0", "team_id": "NYY", "game_datetime": "2025-07-09T23:00:00Z", "runs_scored": 5, "runs_allowed": 2, "hits": 8, "walks": 3, "strikeouts": 9, "innings_pitched": 9, "batters_faced": 34},
        {"game_id": "g0", "team_id": "BOS", "game_datetime": "2025-07-09T23:00:00Z", "runs_scored": 2, "runs_allowed": 5, "hits": 6, "walks": 2, "strikeouts": 7, "innings_pitched": 8, "batters_faced": 35},
        {"game_id": "g1", "team_id": "NYY", "game_datetime": "2025-07-10T23:00:00Z", "runs_scored": 4, "runs_allowed": 3, "hits": 7, "walks": 4, "strikeouts": 8, "innings_pitched": 9, "batters_faced": 36},
        {"game_id": "g1", "team_id": "BOS", "game_datetime": "2025-07-10T23:00:00Z", "runs_scored": 3, "runs_allowed": 4, "hits": 5, "walks": 3, "strikeouts": 6, "innings_pitched": 9, "batters_faced": 34},
    ])
    pitcher_games = pd.DataFrame([
        {"pitcher_id": "p1", "game_datetime": "2025-07-09T23:00:00Z", "earned_runs": 2, "walks": 1, "strikeouts": 7, "innings_pitched": 6, "hits_allowed": 5, "fip": 3.2, "bb": 1, "k": 7},
        {"pitcher_id": "p1", "game_datetime": "2025-07-10T23:00:00Z", "earned_runs": 1, "walks": 2, "strikeouts": 8, "innings_pitched": 7, "hits_allowed": 4, "fip": 2.9, "bb": 2, "k": 8},
        {"pitcher_id": "p2", "game_datetime": "2025-07-09T23:00:00Z", "earned_runs": 3, "walks": 2, "strikeouts": 6, "innings_pitched": 5, "hits_allowed": 7, "fip": 4.1, "bb": 2, "k": 6},
        {"pitcher_id": "p2", "game_datetime": "2025-07-10T23:00:00Z", "earned_runs": 2, "walks": 1, "strikeouts": 9, "innings_pitched": 6, "hits_allowed": 4, "fip": 3.5, "bb": 1, "k": 9},
    ])
    bullpen = pd.DataFrame([
        {"team_id": "NYY", "appearance_datetime": "2025-07-09T02:00:00Z", "innings": 3.0, "is_high_leverage": 1},
        {"team_id": "BOS", "appearance_datetime": "2025-07-09T03:00:00Z", "innings": 4.0, "is_high_leverage": 1},
    ])
    elo = pd.DataFrame([
        {"team_id": "NYY", "rating_ts": "2025-07-09T22:00:00Z", "elo": 1520},
        {"team_id": "BOS", "rating_ts": "2025-07-09T22:00:00Z", "elo": 1490},
    ])

    features = build_pregame_features(games, team_games, pitcher_games, bullpen, elo)

    assert set(["game_id", "side", "as_of_ts"]).issubset(features.columns)
    assert len(features[features["game_id"] == "g1"]) == 2
    validate_point_in_time_integrity(features, games)
