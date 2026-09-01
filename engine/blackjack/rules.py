"""Rule configuration for the blackjack engine."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class RuleSet:
    """The table rules a round is evaluated under.

    Defaults describe the most common "standard rules" shoe game found on a US
    casino floor: six decks, dealer stands on soft 17, blackjack pays 3:2,
    double on any two cards, double after split, split to four hands, aces get
    one card each, and no surrender.
    """

    decks: int = 6
    dealer_hits_soft_17: bool = False
    blackjack_pays: float = 1.5
    #: "any2", "9-11" or "10-11"
    double_rule: str = "any2"
    double_after_split: bool = True
    #: 2 (no resplitting) or 4 (each split hand may split once more)
    max_split_hands: int = 4
    resplit_aces: bool = False
    hit_split_aces: bool = False
    late_surrender: bool = False
    #: The dealer peeks at the hole card, so a dealer blackjack costs the
    #: player only the original wager.  The engine only models peek games.
    peek: bool = True

    def __post_init__(self) -> None:
        if self.decks < 1:
            raise ValueError("decks must be >= 1")
        if self.double_rule not in ("any2", "9-11", "10-11", "none"):
            raise ValueError("unknown double_rule %r" % (self.double_rule,))
        if self.max_split_hands not in (2, 4):
            raise ValueError("max_split_hands must be 2 or 4")
        if not self.peek:
            raise NotImplementedError("only peek (US hole card) games are modelled")

    # -- helpers ------------------------------------------------------------
    def may_double(self, total: int, soft: bool, num_cards: int, after_split: bool) -> bool:
        if self.double_rule == "none":
            return False
        if num_cards != 2:
            return False
        if after_split and not self.double_after_split:
            return False
        if self.double_rule == "any2":
            return True
        if soft:
            return False
        if self.double_rule == "9-11":
            return 9 <= total <= 11
        return 10 <= total <= 11

    def describe(self) -> str:
        parts = [
            "%d deck%s" % (self.decks, "s" if self.decks != 1 else ""),
            "H17" if self.dealer_hits_soft_17 else "S17",
            "BJ pays %s" % _ratio(self.blackjack_pays),
            {"any2": "DOA", "9-11": "D9-11", "10-11": "D10-11", "none": "no double"}[
                self.double_rule
            ],
            "DAS" if self.double_after_split else "no DAS",
            "split to %d" % self.max_split_hands,
            "RSA" if self.resplit_aces else "no RSA",
            "hit split aces" if self.hit_split_aces else "one card to split aces",
            "late surrender" if self.late_surrender else "no surrender",
            "dealer peeks",
        ]
        return ", ".join(parts)


def _ratio(payout: float) -> str:
    frac = Fraction(payout).limit_denominator(100)
    return "%d:%d" % (frac.numerator, frac.denominator)
