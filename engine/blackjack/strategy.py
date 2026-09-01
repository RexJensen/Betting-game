"""Fixed (total dependent) basic strategy charts.

Effects of removal are measured with the player's decisions held *fixed* --
otherwise a composition change would be rewarded twice, once for the shift in
outcomes and once for the strategy adapting to it.  This module holds the
standard multi-deck charts plus the machinery to resolve a chart cell into a
legal action for a specific hand.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .cards import ACE, NUM_RANKS, RANKS, TEN, state_is_soft, state_total
from .rules import RuleSet

STAND = 0
HIT = 1
DOUBLE = 2
SPLIT = 3
SURRENDER = 4

ACTION_NAMES = {STAND: "stand", HIT: "hit", DOUBLE: "double", SPLIT: "split",
                SURRENDER: "surrender"}
ACTION_LETTERS = {STAND: "S", HIT: "H", DOUBLE: "D", SPLIT: "P", SURRENDER: "R"}

#: Chart columns, in the usual printed order.
COLUMN_UPCARDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)  # 2,3,...,9,T,A as rank indices

# Cell codes:
#   S   stand                       H   hit
#   D   double, else hit            Ds  double, else stand
#   P   split                       Pd  split if DAS, else hit
#   R   surrender, else hit         Rs  surrender, else stand
#   Rp  surrender, else split

_HARD_S17 = {
    4:  "H  H  H  H  H  H  H  H  H  H",
    5:  "H  H  H  H  H  H  H  H  H  H",
    6:  "H  H  H  H  H  H  H  H  H  H",
    7:  "H  H  H  H  H  H  H  H  H  H",
    8:  "H  H  H  H  H  H  H  H  H  H",
    9:  "H  D  D  D  D  H  H  H  H  H",
    10: "D  D  D  D  D  D  D  D  H  H",
    11: "D  D  D  D  D  D  D  D  D  H",
    12: "H  H  S  S  S  H  H  H  H  H",
    13: "S  S  S  S  S  H  H  H  H  H",
    14: "S  S  S  S  S  H  H  H  H  H",
    15: "S  S  S  S  S  H  H  H  H  H",
    16: "S  S  S  S  S  H  H  H  H  H",
    17: "S  S  S  S  S  S  S  S  S  S",
    18: "S  S  S  S  S  S  S  S  S  S",
    19: "S  S  S  S  S  S  S  S  S  S",
    20: "S  S  S  S  S  S  S  S  S  S",
    21: "S  S  S  S  S  S  S  S  S  S",
}

_SOFT_S17 = {
    12: "H  H  H  H  H  H  H  H  H  H",   # A,A when it cannot be split
    13: "H  H  H  D  D  H  H  H  H  H",
    14: "H  H  H  D  D  H  H  H  H  H",
    15: "H  H  D  D  D  H  H  H  H  H",
    16: "H  H  D  D  D  H  H  H  H  H",
    17: "H  D  D  D  D  H  H  H  H  H",
    18: "S  Ds Ds Ds Ds S  S  H  H  H",
    19: "S  S  S  S  S  S  S  S  S  S",
    20: "S  S  S  S  S  S  S  S  S  S",
    21: "S  S  S  S  S  S  S  S  S  S",
}

_PAIRS_S17 = {
    ACE: "P  P  P  P  P  P  P  P  P  P",
    1:   "Pd Pd P  P  P  P  H  H  H  H",   # 2,2
    2:   "Pd Pd P  P  P  P  H  H  H  H",   # 3,3
    3:   "H  H  H  Pd Pd H  H  H  H  H",   # 4,4
    4:   "D  D  D  D  D  D  D  D  H  H",   # 5,5 -- play as hard ten
    5:   "Pd P  P  P  P  H  H  H  H  H",   # 6,6
    6:   "P  P  P  P  P  P  H  H  H  H",   # 7,7
    7:   "P  P  P  P  P  P  P  P  P  P",   # 8,8
    8:   "P  P  P  P  P  S  P  P  S  S",   # 9,9
    TEN: "S  S  S  S  S  S  S  S  S  S",   # T,T
}

# --- dealer hits soft 17 deltas -------------------------------------------
_HARD_H17_OVERRIDES = {(11, ACE): "D"}
_SOFT_H17_OVERRIDES = {(18, 1): "Ds", (19, 5): "Ds"}
_PAIR_H17_OVERRIDES: Dict = {}

# --- late surrender overlays ----------------------------------------------
_SURRENDER_S17 = {("hard", 15, TEN), ("hard", 16, 8), ("hard", 16, TEN),
                  ("hard", 16, ACE)}
_SURRENDER_H17 = _SURRENDER_S17 | {("hard", 15, ACE), ("hard", 17, ACE),
                                   ("pair", 7, ACE)}


def _parse_row(row: str) -> Dict[int, str]:
    cells = row.split()
    if len(cells) != NUM_RANKS:
        raise ValueError("chart row needs %d cells: %r" % (NUM_RANKS, row))
    return {up: cell for up, cell in zip(COLUMN_UPCARDS, cells)}


class BasicStrategy:
    """A total dependent chart resolved against a specific rule set."""

    __slots__ = ("rules", "hard", "soft", "pairs", "name")

    def __init__(self, rules: RuleSet, name: str = "basic"):
        self.rules = rules
        self.name = name
        h17 = rules.dealer_hits_soft_17

        self.hard: Dict[int, Dict[int, str]] = {
            total: _parse_row(row) for total, row in _HARD_S17.items()
        }
        self.soft: Dict[int, Dict[int, str]] = {
            total: _parse_row(row) for total, row in _SOFT_S17.items()
        }
        self.pairs: Dict[int, Dict[int, str]] = {
            rank: _parse_row(row) for rank, row in _PAIRS_S17.items()
        }
        if h17:
            for (total, up), cell in _HARD_H17_OVERRIDES.items():
                self.hard[total][up] = cell
            for (total, up), cell in _SOFT_H17_OVERRIDES.items():
                self.soft[total][up] = cell
            for (rank, up), cell in _PAIR_H17_OVERRIDES.items():
                self.pairs[rank][up] = cell
        if rules.late_surrender:
            overlay = _SURRENDER_H17 if h17 else _SURRENDER_S17
            for kind, key, up in overlay:
                if kind == "hard":
                    base = self.hard[key][up]
                    self.hard[key][up] = "Rs" if base == "S" else "R"
                else:
                    self.pairs[key][up] = "Rp"

    # -- lookups ------------------------------------------------------------
    def cell(self, state: int, up: int, pair_rank: Optional[int]) -> str:
        if pair_rank is not None:
            return self.pairs[pair_rank][up]
        total = state_total(state)
        if state_is_soft(state):
            return self.soft[total][up]
        return self.hard[max(total, 4)][up]

    def should_split(self, pair_rank: int, up: int) -> bool:
        cell = self.pairs[pair_rank][up]
        if cell == "P" or cell == "Rp":
            return True
        if cell == "Pd":
            return self.rules.double_after_split
        return False

    def decide(self, state: int, up: int, pair_rank: Optional[int],
               can_double: bool, can_split: bool, can_surrender: bool) -> int:
        """Resolve the chart cell into a legal action."""
        cell = self.cell(state, up, pair_rank if can_split else None)
        if cell in ("P", "Rp") or (cell == "Pd" and self.rules.double_after_split):
            if cell == "Rp" and can_surrender:
                return SURRENDER
            if can_split:
                return SPLIT
            cell = self.cell(state, up, None)
        elif cell == "Pd" or (cell in ("P", "Rp") and not can_split):
            cell = self.cell(state, up, None)
        if cell in ("R", "Rs"):
            if can_surrender:
                return SURRENDER
            cell = "H" if cell == "R" else "S"
        if cell == "D":
            return DOUBLE if can_double else HIT
        if cell == "Ds":
            return DOUBLE if can_double else STAND
        if cell == "S":
            return STAND
        if cell == "H":
            return HIT
        raise ValueError("unresolved chart cell %r" % (cell,))

    # -- reporting ----------------------------------------------------------
    def render(self) -> str:
        header = "      " + "  ".join("%2s" % RANKS[up] for up in COLUMN_UPCARDS)
        lines = ["Hard totals", header]
        for total in sorted(self.hard, reverse=True):
            if total < 5:
                continue
            row = "  ".join("%2s" % self.hard[total][up] for up in COLUMN_UPCARDS)
            lines.append("%4d  %s" % (total, row))
        lines += ["", "Soft totals", header]
        for total in sorted(self.soft, reverse=True):
            if total < 13:
                continue
            row = "  ".join("%2s" % self.soft[total][up] for up in COLUMN_UPCARDS)
            lines.append("A,%-2d  %s" % (total - 11, row))
        lines += ["", "Pairs", header]
        for rank in (ACE, 1, 2, 3, 4, 5, 6, 7, 8, TEN):
            row = "  ".join("%2s" % self.pairs[rank][up] for up in COLUMN_UPCARDS)
            lines.append("%s,%s   %s" % (RANKS[rank], RANKS[rank], row))
        return "\n".join(lines)
