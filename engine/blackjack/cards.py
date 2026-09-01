"""Card, hand and shoe primitives used by the blackjack engine.

Ranks are collapsed to the ten values that matter for blackjack: an ace plus
2..9 plus a single "T" bucket that covers 10/J/Q/K.  Suits are irrelevant, so a
shoe is fully described by the count of each rank.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

RANKS: tuple = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T")
NUM_RANKS = 10
ACE = 0
TEN = 9

#: Blackjack value of each rank.  The ace enters a hand as 11 and is demoted to
#: 1 by :data:`HAND_TRANSITION` as soon as the hand would otherwise bust.
RANK_VALUES: tuple = (11, 2, 3, 4, 5, 6, 7, 8, 9, 10)

#: Cards of each rank in one 52 card deck.
DECK_COMPOSITION: tuple = (4, 4, 4, 4, 4, 4, 4, 4, 4, 16)

# --- hand states -----------------------------------------------------------
#
# A hand is summarised by ``(total, soft)`` where ``soft`` means one ace is
# still being counted as 11.  That pair is packed into a single integer
# ``state = total * 2 + soft`` so that hitting is a table lookup rather than a
# function call -- the dealer recursion runs this tens of millions of times.

BUST = -1
MAX_STATE = 44  # totals 0..21, two states each


def make_state(total: int, soft: bool) -> int:
    return total * 2 + (1 if soft else 0)


def state_total(state: int) -> int:
    return state >> 1


def state_is_soft(state: int) -> bool:
    return bool(state & 1)


def _next_state(state: int, rank: int) -> int:
    total = state >> 1
    soft = state & 1
    total += RANK_VALUES[rank]
    if rank == ACE and soft:
        # A second ace can never be worth 11 as well.
        total -= 10
    elif total > 21 and soft:
        total -= 10
        soft = 0
    if rank == ACE and not soft:
        soft = 1
        if total > 21:
            total -= 10
            soft = 0
    if total > 21:
        return BUST
    return total * 2 + soft


#: ``HAND_TRANSITION[state][rank]`` -> new state, or :data:`BUST`.
HAND_TRANSITION: tuple = tuple(
    tuple(_next_state(state, rank) for rank in range(NUM_RANKS))
    for state in range(MAX_STATE)
)

EMPTY_STATE = make_state(0, False)

#: ``TWO_CARD_STATE[a][b]`` -> state of the two card hand ``a`` + ``b``.
TWO_CARD_STATE: tuple = tuple(
    tuple(HAND_TRANSITION[HAND_TRANSITION[EMPTY_STATE][a]][b] for b in range(NUM_RANKS))
    for a in range(NUM_RANKS)
)

# --- shoe ------------------------------------------------------------------

_KEY_BITS = 10  # up to 1023 cards of a single rank
_KEY_UNITS: tuple = tuple(1 << (_KEY_BITS * r) for r in range(NUM_RANKS))


class Shoe:
    """A mutable multiset of cards with a cheap, exact memo key.

    ``key`` is a packed integer of the rank counts.  It is maintained
    incrementally by :meth:`draw` / :meth:`put_back` so that recursive
    evaluators can memoise on the exact remaining composition.
    """

    __slots__ = ("counts", "total", "key")

    def __init__(self, counts: Sequence[int]):
        if len(counts) != NUM_RANKS:
            raise ValueError("a shoe needs %d rank counts" % NUM_RANKS)
        if any(c < 0 for c in counts):
            raise ValueError("rank counts must be non-negative")
        self.counts: List[int] = list(counts)
        self.total: int = sum(counts)
        self.key: int = 0
        for rank, count in enumerate(counts):
            self.key += count * _KEY_UNITS[rank]

    # -- construction -------------------------------------------------------
    @classmethod
    def from_decks(cls, decks: int) -> "Shoe":
        return cls([count * decks for count in DECK_COMPOSITION])

    def copy(self) -> "Shoe":
        return Shoe(self.counts)

    # -- mutation -----------------------------------------------------------
    def draw(self, rank: int) -> None:
        self.counts[rank] -= 1
        self.total -= 1
        self.key -= _KEY_UNITS[rank]

    def put_back(self, rank: int) -> None:
        self.counts[rank] += 1
        self.total += 1
        self.key += _KEY_UNITS[rank]

    def adjusted(self, deltas: Iterable[tuple]) -> "Shoe":
        """Return a copy with ``(rank, delta)`` pairs applied."""
        counts = list(self.counts)
        for rank, delta in deltas:
            counts[rank] += delta
        return Shoe(counts)

    # -- reporting ----------------------------------------------------------
    def penetration_string(self) -> str:
        return " ".join(
            "%s:%d" % (RANKS[r], self.counts[r]) for r in range(NUM_RANKS)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "Shoe(%s, total=%d)" % (self.penetration_string(), self.total)
