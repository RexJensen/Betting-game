import csv
from pathlib import Path

import pytest

from engine.ledger import (
    Wager,
    american_odds_to_decimal,
    derive_account,
    place_wager,
    settle_wager,
)


def _ledger_path(run_id: str) -> Path:
    return Path("data") / "simulations" / run_id / "ledger.csv"


def test_american_odds_to_decimal_conversion():
    assert american_odds_to_decimal(150) == pytest.approx(2.5)
    assert american_odds_to_decimal(-200) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        american_odds_to_decimal(0)


def test_payout_and_push_and_balance_derivation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run-a"

    place_wager(run_id, Wager("b1", "g1", "moneyline", 150, 100.0, "2026-01-01T00:00:00Z"))
    settle_wager(run_id, "b1", "win", timestamp="2026-01-01T03:00:00Z")

    place_wager(run_id, Wager("b2", "g2", "spread", -110, 50.0, "2026-01-02T00:00:00Z"))
    settle_wager(run_id, "b2", "push", timestamp="2026-01-02T03:00:00Z")

    account = derive_account(run_id, starting_bankroll=1000.0)
    # +150 on b1 win, +0 on push overall net after stake accounting = +150
    assert account.current_balance == pytest.approx(1150.0)
    assert account.open_exposure == pytest.approx(0.0)


def test_zero_or_invalid_stake_and_duplicate_bet_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run-b"

    with pytest.raises(ValueError):
        place_wager(run_id, Wager("x1", "g1", "moneyline", 110, 0.0, "2026-01-01T00:00:00Z"))

    with pytest.raises(ValueError):
        place_wager(run_id, Wager("x2", "g1", "moneyline", 110, -10.0, "2026-01-01T00:00:00Z"))

    place_wager(run_id, Wager("x3", "g1", "moneyline", 110, 10.0, "2026-01-01T00:00:00Z"))
    with pytest.raises(ValueError):
        place_wager(run_id, Wager("x3", "g2", "moneyline", 110, 15.0, "2026-01-01T00:01:00Z"))


def test_ledger_csv_persists_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run-c"

    place_wager(run_id, Wager("z1", "g7", "total", -120, 40.0, "2026-01-05T00:00:00Z"))
    settle_wager(run_id, "z1", "loss", timestamp="2026-01-05T02:00:00Z")

    ledger_file = _ledger_path(run_id)
    with ledger_file.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["event_type"] == "place"
    assert rows[1]["event_type"] == "settle"
    assert rows[1]["status"] == "loss"
