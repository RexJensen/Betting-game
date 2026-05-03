from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

OUTPUT_PATH = Path("data/warehouse/pregame_features.parquet")


ROLLING_WINDOWS = (7, 14, 30)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"Unsupported input format: {path}")


def _to_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Found invalid timestamps")
    return ts


def _team_view(games: pd.DataFrame) -> pd.DataFrame:
    home = games[["game_id", "game_datetime", "home_team_id", "away_team_id"]].rename(
        columns={"home_team_id": "team_id", "away_team_id": "opp_team_id"}
    )
    home["side"] = "home"

    away = games[["game_id", "game_datetime", "away_team_id", "home_team_id"]].rename(
        columns={"away_team_id": "team_id", "home_team_id": "opp_team_id"}
    )
    away["side"] = "away"

    out = pd.concat([home, away], ignore_index=True)
    out["as_of_ts"] = out["game_datetime"] - pd.Timedelta(days=1)
    return out


def _rolling_and_season(team_games: pd.DataFrame) -> pd.DataFrame:
    team_games = team_games.sort_values(["team_id", "game_datetime"]).copy()
    # Shift all stats to prevent current game leakage.
    for col in ["runs_scored", "runs_allowed", "hits", "walks", "strikeouts", "innings_pitched", "batters_faced"]:
        team_games[f"prev_{col}"] = team_games.groupby("team_id")[col].shift(1)

    for win in ROLLING_WINDOWS:
        grp = team_games.groupby("team_id")
        team_games[f"roll{win}_runs_scored"] = grp["prev_runs_scored"].transform(lambda s: s.rolling(win, min_periods=1).mean())
        team_games[f"roll{win}_runs_allowed"] = grp["prev_runs_allowed"].transform(lambda s: s.rolling(win, min_periods=1).mean())
        team_games[f"roll{win}_k_rate"] = (
            grp["prev_strikeouts"].transform(lambda s: s.rolling(win, min_periods=1).sum())
            / grp["prev_batters_faced"].transform(lambda s: s.rolling(win, min_periods=1).sum())
        )

    grp = team_games.groupby("team_id")
    team_games["s2d_batting_avg"] = grp["prev_hits"].cumsum() / grp["prev_batters_faced"].cumsum()
    team_games["s2d_pitching_era"] = 9 * grp["prev_runs_allowed"].cumsum() / grp["prev_innings_pitched"].cumsum()
    return team_games


def _pitcher_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    pitcher_games = pitcher_games.sort_values(["pitcher_id", "game_datetime"]).copy()
    grp = pitcher_games.groupby("pitcher_id")
    for col in ["earned_runs", "walks", "strikeouts", "innings_pitched", "hits_allowed", "fip", "bb", "k"]:
        pitcher_games[f"prev_{col}"] = grp[col].shift(1)

    pitcher_games["sp_era"] = 9 * grp["prev_earned_runs"].cumsum() / grp["prev_innings_pitched"].cumsum()
    pitcher_games["sp_whip"] = (grp["prev_walks"].cumsum() + grp["prev_hits_allowed"].cumsum()) / grp["prev_innings_pitched"].cumsum()
    pitcher_games["sp_fip"] = grp["prev_fip"].expanding().mean().reset_index(level=0, drop=True)
    pitcher_games["sp_k_bb"] = grp["prev_k"].cumsum() / grp["prev_bb"].cumsum()
    pitcher_games["sp_rest_days"] = (pitcher_games["game_datetime"] - grp["game_datetime"].shift(1)).dt.days
    return pitcher_games


def _bullpen_workload(bullpen_appearances: pd.DataFrame) -> pd.DataFrame:
    bp = bullpen_appearances.sort_values(["team_id", "appearance_datetime"]).copy()
    bp["recent_innings_7d"] = bp.groupby("team_id").apply(
        lambda g: g.set_index("appearance_datetime")["innings"].rolling("7D", closed="left").sum()
    ).reset_index(level=0, drop=True).values
    bp["high_lev_apps_7d"] = bp.groupby("team_id").apply(
        lambda g: g.set_index("appearance_datetime")["is_high_leverage"].rolling("7D", closed="left").sum()
    ).reset_index(level=0, drop=True).values
    return bp


def build_pregame_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    pitcher_games: pd.DataFrame,
    bullpen_appearances: pd.DataFrame,
    elo_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    games = games.copy()
    games["game_datetime"] = _to_utc(games["game_datetime"])
    team_games["game_datetime"] = _to_utc(team_games["game_datetime"])
    pitcher_games["game_datetime"] = _to_utc(pitcher_games["game_datetime"])
    bullpen_appearances["appearance_datetime"] = _to_utc(bullpen_appearances["appearance_datetime"])
    elo_snapshots["rating_ts"] = _to_utc(elo_snapshots["rating_ts"])

    base = _team_view(games)
    team_rates = _rolling_and_season(team_games)
    pitcher_rates = _pitcher_features(pitcher_games)
    bp = _bullpen_workload(bullpen_appearances)

    out = base.merge(
        team_rates,
        on=["team_id", "game_id", "game_datetime"],
        how="left",
    )
    out = out.merge(
        games[["game_id", "home_starting_pitcher_id", "away_starting_pitcher_id"]],
        on="game_id",
        how="left",
    )
    out["starting_pitcher_id"] = out["home_starting_pitcher_id"].where(out["side"] == "home", out["away_starting_pitcher_id"])
    out = out.merge(
        pitcher_rates[["pitcher_id", "game_datetime", "sp_era", "sp_fip", "sp_whip", "sp_k_bb", "sp_rest_days"]],
        left_on=["starting_pitcher_id", "game_datetime"],
        right_on=["pitcher_id", "game_datetime"],
        how="left",
    )

    bp_aligned = pd.merge_asof(
        out.sort_values("as_of_ts"),
        bp[["team_id", "appearance_datetime", "recent_innings_7d", "high_lev_apps_7d"]].sort_values("appearance_datetime"),
        left_on="as_of_ts",
        right_on="appearance_datetime",
        by="team_id",
        direction="backward",
    )

    # home/away and handedness splits from pre-computed table without future rows.
    split_cols = ["game_id", "team_id", "split_home_wrc_plus", "split_vs_rhp_ops", "split_vs_lhp_ops"]
    if all(c in team_games.columns for c in split_cols[2:]):
        bp_aligned = bp_aligned.merge(team_games[split_cols], on=["game_id", "team_id"], how="left")

    elo = elo_snapshots.sort_values("rating_ts")
    bp_aligned = pd.merge_asof(
        bp_aligned.sort_values("as_of_ts"),
        elo[["team_id", "rating_ts", "elo"]],
        left_on="as_of_ts",
        right_on="rating_ts",
        by="team_id",
        direction="backward",
    )

    keep = [
        "game_id", "side", "as_of_ts", "team_id", "opp_team_id",
        "roll7_runs_scored", "roll14_runs_scored", "roll30_runs_scored",
        "roll7_runs_allowed", "roll14_runs_allowed", "roll30_runs_allowed",
        "roll7_k_rate", "roll14_k_rate", "roll30_k_rate",
        "s2d_batting_avg", "s2d_pitching_era",
        "sp_era", "sp_fip", "sp_whip", "sp_k_bb", "sp_rest_days",
        "recent_innings_7d", "high_lev_apps_7d",
        "split_home_wrc_plus", "split_vs_rhp_ops", "split_vs_lhp_ops",
        "elo",
    ]
    keep = [c for c in keep if c in bp_aligned.columns]
    return bp_aligned[keep].drop_duplicates(subset=["game_id", "side", "as_of_ts"])


def validate_point_in_time_integrity(features_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
    game_times = games_df[["game_id", "game_datetime"]].copy()
    game_times["game_datetime"] = _to_utc(game_times["game_datetime"])
    merged = features_df.merge(game_times, on="game_id", how="left")
    bad = merged[merged["as_of_ts"] >= merged["game_datetime"]]
    if not bad.empty:
        raise ValueError(f"Found {len(bad)} rows with feature timestamp at/after game start")


def run_pipeline(
    games_source: str | Path,
    team_games_source: str | Path,
    pitcher_games_source: str | Path,
    bullpen_source: str | Path,
    elo_source: str | Path,
    output: str | Path = OUTPUT_PATH,
) -> Path:
    games = _read_table(games_source)
    team_games = _read_table(team_games_source)
    pitcher_games = _read_table(pitcher_games_source)
    bullpen = _read_table(bullpen_source)
    elo = _read_table(elo_source)

    features = build_pregame_features(games, team_games, pitcher_games, bullpen, elo)
    validate_point_in_time_integrity(features, games)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    return output
