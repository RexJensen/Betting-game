from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
from typing import Iterable, Optional


@dataclass(frozen=True)
class Wager:
    bet_id: str
    game_id: str
    market: str
    odds: int
    stake: float
    timestamp: str
    status: str = "open"


@dataclass
class Account:
    starting_bankroll: float
    current_balance: float
    open_exposure: float


EVENT_HEADERS = [
    "event_id",
    "timestamp",
    "event_type",
    "bet_id",
    "game_id",
    "market",
    "odds",
    "stake",
    "pnl",
    "status",
]


def american_odds_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_ledger_path(run_id: str) -> Path:
    path = Path("data") / "simulations" / run_id / "ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_HEADERS)
            writer.writeheader()
    return path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_event(path: Path, event: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_HEADERS)
        writer.writerow(event)


def place_wager(run_id: str, wager: Wager) -> None:
    if wager.stake <= 0:
        raise ValueError("Stake must be positive")

    path = _ensure_ledger_path(run_id)
    events = _read_events(path)
    if any(e["bet_id"] == wager.bet_id and e["event_type"] == "place" for e in events):
        raise ValueError(f"Duplicate bet id: {wager.bet_id}")

    event = {
        "event_id": f"{wager.bet_id}:place",
        "timestamp": wager.timestamp or _now_iso(),
        "event_type": "place",
        "bet_id": wager.bet_id,
        "game_id": wager.game_id,
        "market": wager.market,
        "odds": str(wager.odds),
        "stake": f"{wager.stake:.6f}",
        "pnl": f"{-wager.stake:.6f}",
        "status": "open",
    }
    _append_event(path, event)


def settle_wager(run_id: str, bet_id: str, result: str, *, odds_convention: str = "american", timestamp: Optional[str] = None) -> float:
    path = _ensure_ledger_path(run_id)
    events = _read_events(path)

    placed = [e for e in events if e["bet_id"] == bet_id and e["event_type"] == "place"]
    if not placed:
        raise ValueError(f"No placed wager for bet id: {bet_id}")

    if any(e["bet_id"] == bet_id and e["event_type"] in {"settle", "void"} for e in events):
        raise ValueError(f"Wager already closed for bet id: {bet_id}")

    place = placed[-1]
    stake = float(place["stake"])
    odds = int(float(place["odds"]))

    result_norm = result.lower()
    if result_norm not in {"win", "loss", "push"}:
        raise ValueError("Result must be one of: win, loss, push")
    if odds_convention.lower() != "american":
        raise ValueError("Only american odds convention is supported")

    if result_norm == "win":
        payout = stake * american_odds_to_decimal(odds)
        pnl = payout
    elif result_norm == "push":
        pnl = stake
    else:
        pnl = 0.0

    event = {
        "event_id": f"{bet_id}:settle",
        "timestamp": timestamp or _now_iso(),
        "event_type": "settle",
        "bet_id": bet_id,
        "game_id": place["game_id"],
        "market": place["market"],
        "odds": place["odds"],
        "stake": place["stake"],
        "pnl": f"{pnl:.6f}",
        "status": result_norm,
    }
    _append_event(path, event)
    return pnl


def void_wager(run_id: str, bet_id: str, timestamp: Optional[str] = None) -> float:
    path = _ensure_ledger_path(run_id)
    events = _read_events(path)
    placed = [e for e in events if e["bet_id"] == bet_id and e["event_type"] == "place"]
    if not placed:
        raise ValueError(f"No placed wager for bet id: {bet_id}")
    if any(e["bet_id"] == bet_id and e["event_type"] in {"settle", "void"} for e in events):
        raise ValueError(f"Wager already closed for bet id: {bet_id}")

    place = placed[-1]
    stake = float(place["stake"])

    event = {
        "event_id": f"{bet_id}:void",
        "timestamp": timestamp or _now_iso(),
        "event_type": "void",
        "bet_id": bet_id,
        "game_id": place["game_id"],
        "market": place["market"],
        "odds": place["odds"],
        "stake": place["stake"],
        "pnl": f"{stake:.6f}",
        "status": "void",
    }
    _append_event(path, event)
    return stake


def derive_account(run_id: str, starting_bankroll: float) -> Account:
    path = _ensure_ledger_path(run_id)
    events = _read_events(path)

    pnl_sum = sum(float(e["pnl"]) for e in events)

    open_bet_ids: set[str] = set()
    for e in events:
        if e["event_type"] == "place":
            open_bet_ids.add(e["bet_id"])
        elif e["event_type"] in {"settle", "void"}:
            open_bet_ids.discard(e["bet_id"])

    place_by_id = {e["bet_id"]: float(e["stake"]) for e in events if e["event_type"] == "place"}
    open_exposure = sum(place_by_id[bet_id] for bet_id in open_bet_ids)

    return Account(
        starting_bankroll=starting_bankroll,
        current_balance=starting_bankroll + pnl_sum,
        open_exposure=open_exposure,
    )
