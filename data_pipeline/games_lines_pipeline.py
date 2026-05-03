from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA_PATH = Path("data_pipeline/schemas/games_lines.schema.json")
OUTPUT_PATH = Path("data/warehouse/games_lines.parquet")

# Legacy abbreviations + relocations mapped to canonical IDs.
TEAM_NORMALIZATION_MAP = {
    "NJN": "BKN",  # New Jersey Nets -> Brooklyn Nets
    "NOH": "NOP",  # New Orleans Hornets -> New Orleans Pelicans
    "CHA": "CHO",  # legacy Charlotte abbreviation
    "SEA": "OKC",  # Seattle SuperSonics -> Oklahoma City Thunder
    "VAN": "MEM",  # Vancouver Grizzlies -> Memphis Grizzlies
    "STL": "ARI",  # St. Louis Cardinals -> Arizona Cardinals (NFL)
    "OAK": "LV",   # Oakland Raiders -> Las Vegas Raiders
    "SD": "LAC",   # San Diego Chargers -> Los Angeles Chargers
}


@dataclass
class MissingLineRule:
    flag_column: str
    reason: str


MISSING_LINE_RULES = [
    MissingLineRule("moneyline_home", "moneyline_home_missing"),
    MissingLineRule("moneyline_away", "moneyline_away_missing"),
    MissingLineRule("spread_home", "spread_home_missing"),
    MissingLineRule("spread_away", "spread_away_missing"),
    MissingLineRule("total_points", "total_points_missing"),
]


def _load_schema(path: Path = SCHEMA_PATH) -> dict:
    return json.loads(path.read_text())


def ingest_games(games_source: str | Path) -> pd.DataFrame:
    """Load games table from CSV/Parquet/JSON into a DataFrame."""
    games_source = Path(games_source)
    if games_source.suffix == ".csv":
        df = pd.read_csv(games_source)
    elif games_source.suffix == ".parquet":
        df = pd.read_parquet(games_source)
    elif games_source.suffix in {".json", ".jsonl"}:
        df = pd.read_json(games_source, lines=games_source.suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported games format: {games_source}")

    required = {"game_id", "game_datetime", "home_team", "away_team", "home_score", "away_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required game columns: {sorted(missing)}")
    return df


def ingest_lines(lines_source: str | Path) -> pd.DataFrame:
    """Load sportsbook lines from CSV/Parquet/JSON into a DataFrame."""
    lines_source = Path(lines_source)
    if lines_source.suffix == ".csv":
        df = pd.read_csv(lines_source)
    elif lines_source.suffix == ".parquet":
        df = pd.read_parquet(lines_source)
    elif lines_source.suffix in {".json", ".jsonl"}:
        df = pd.read_json(lines_source, lines=lines_source.suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported lines format: {lines_source}")

    required = {
        "game_id", "book", "line_timestamp", "line_source", "moneyline_home", "moneyline_away", "spread", "total"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required line columns: {sorted(missing)}")
    return df


def normalize_team(team: str) -> str:
    normalized = str(team).strip().upper()
    return TEAM_NORMALIZATION_MAP.get(normalized, normalized)


def _to_utc(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    if timestamps.isna().any():
        bad_count = int(timestamps.isna().sum())
        raise ValueError(f"Found {bad_count} non-parseable datetime values during UTC normalization")
    return timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_and_join(games_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    games = games_df.copy()
    lines = lines_df.copy()

    games["game_datetime_utc"] = _to_utc(games["game_datetime"])
    lines["line_timestamp"] = _to_utc(lines["line_timestamp"])

    games["home_team_id"] = games["home_team"].map(normalize_team)
    games["away_team_id"] = games["away_team"].map(normalize_team)

    # Convert single spread column into home/away pair.
    lines["spread_home"] = pd.to_numeric(lines["spread"], errors="coerce")
    lines["spread_away"] = -1 * lines["spread_home"]
    lines["total_points"] = pd.to_numeric(lines["total"], errors="coerce")
    lines["moneyline_home"] = pd.to_numeric(lines["moneyline_home"], errors="coerce")
    lines["moneyline_away"] = pd.to_numeric(lines["moneyline_away"], errors="coerce")

    # Deduplicate by (game_id, book, line_timestamp), keep latest ingested row.
    lines = lines.sort_values(["game_id", "book", "line_timestamp"]).drop_duplicates(
        subset=["game_id", "book", "line_timestamp"], keep="last"
    )

    joined = games.merge(lines, on="game_id", how="left", suffixes=("", "_line"))

    joined["winner_team_id"] = pd.NA
    joined.loc[joined["home_score"] > joined["away_score"], "winner_team_id"] = joined["home_team_id"]
    joined.loc[joined["away_score"] > joined["home_score"], "winner_team_id"] = joined["away_team_id"]

    # Missing-line handling.
    joined["line_missing_reason"] = pd.NA
    for rule in MISSING_LINE_RULES:
        mask = joined[rule.flag_column].isna()
        joined.loc[mask & joined["line_missing_reason"].isna(), "line_missing_reason"] = rule.reason

    canonical_cols = [
        "game_id",
        "game_datetime_utc",
        "home_team_id",
        "away_team_id",
        "book",
        "line_timestamp",
        "line_source",
        "moneyline_home",
        "moneyline_away",
        "spread_home",
        "spread_away",
        "total_points",
        "line_missing_reason",
        "home_score",
        "away_score",
        "winner_team_id",
    ]
    return joined[canonical_cols]


def _validate_required_columns(df: pd.DataFrame, schema: dict) -> None:
    required = schema.get("required", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical dataset missing required columns: {missing}")


def run_pipeline(games_source: str | Path, lines_source: str | Path, output: str | Path = OUTPUT_PATH) -> Path:
    schema = _load_schema()
    games_df = ingest_games(games_source)
    lines_df = ingest_lines(lines_source)
    canonical = normalize_and_join(games_df, lines_df)
    _validate_required_columns(canonical, schema)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output, index=False)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build canonical games + lines table")
    parser.add_argument("--games", required=True, help="Path to games input file")
    parser.add_argument("--lines", required=True, help="Path to lines input file")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output parquet path")
    args = parser.parse_args()

    result = run_pipeline(args.games, args.lines, args.output)
    print(f"Wrote canonical table to {result}")
