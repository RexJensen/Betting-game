"""Exact dealer outcome probabilities for a given shoe composition.

The dealer's final hand distribution is computed by exhaustive recursion over
every card the dealer could draw, with the shoe depleting as cards come out --
no independence or infinite-deck approximation.  Results are memoised on the
exact remaining composition, which is what makes a full combinatorial analysis
tractable.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .cards import ACE, BUST, EMPTY_STATE, HAND_TRANSITION, NUM_RANKS, TEN, Shoe
from .rules import RuleSet

#: Index of each dealer outcome inside the probability vector.
OUTCOMES = ("17", "18", "19", "20", "21", "bust")
BUST_INDEX = 5

_STAND_VECTORS: Dict[int, Tuple[float, ...]] = {}
for _total in range(17, 22):
    _vec = [0.0] * 6
    _vec[_total - 17] = 1.0
    _STAND_VECTORS[_total] = tuple(_vec)
_BUST_VECTOR = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


class Dealer:
    """Dealer outcome probabilities, conditioned on the dealer not having a
    natural (the hole card has already been peeked at)."""

    __slots__ = ("hits_soft_17", "_play_memo", "_top_memo", "nodes")

    def __init__(self, rules: RuleSet):
        self.hits_soft_17 = rules.dealer_hits_soft_17
        self._play_memo: Dict[int, Tuple[float, ...]] = {}
        self._top_memo: Dict[int, Tuple[float, ...]] = {}
        self.nodes = 0

    def distribution(self, shoe: Shoe, upcard: int) -> Tuple[float, ...]:
        """Probability of each dealer final total given ``upcard``.

        ``shoe`` must already have the upcard (and every other exposed card)
        removed.  The hole card is integrated over here; hole cards that would
        make a natural are excluded and the rest renormalised, which is exactly
        the information the player has once the dealer has peeked.
        """
        cache_key = (shoe.key << 4) | upcard
        cached = self._top_memo.get(cache_key)
        if cached is not None:
            return cached

        counts = shoe.counts
        if upcard == ACE:
            excluded = TEN
        elif upcard == TEN:
            excluded = ACE
        else:
            excluded = -1
        denominator = shoe.total - (counts[excluded] if excluded >= 0 else 0)
        if denominator <= 0:
            raise ValueError("no legal hole card remains in the shoe")

        upstate = HAND_TRANSITION[EMPTY_STATE][upcard]
        transitions = HAND_TRANSITION[upstate]
        acc = [0.0] * 6
        for rank in range(NUM_RANKS):
            count = counts[rank]
            if count == 0 or rank == excluded:
                continue
            probability = count / denominator
            shoe.draw(rank)
            vector = self._play(shoe, transitions[rank])
            shoe.put_back(rank)
            for i in range(6):
                acc[i] += probability * vector[i]
        result = tuple(acc)
        self._top_memo[cache_key] = result
        return result

    def natural_probability(self, shoe: Shoe, upcard: int) -> float:
        """Probability the dealer's hole card completes a natural."""
        if upcard == ACE:
            return shoe.counts[TEN] / shoe.total
        if upcard == TEN:
            return shoe.counts[ACE] / shoe.total
        return 0.0

    # -- internals ----------------------------------------------------------
    def _play(self, shoe: Shoe, state: int) -> Tuple[float, ...]:
        total = state >> 1
        if total >= 17 and not (self.hits_soft_17 and state == 35):
            return _STAND_VECTORS[total]

        cache_key = (shoe.key << 6) | state
        memo = self._play_memo
        cached = memo.get(cache_key)
        if cached is not None:
            return cached

        self.nodes += 1
        counts = shoe.counts
        remaining = shoe.total
        transitions = HAND_TRANSITION[state]
        a0 = a1 = a2 = a3 = a4 = a5 = 0.0
        play = self._play
        for rank in range(NUM_RANKS):
            count = counts[rank]
            if count == 0:
                continue
            probability = count / remaining
            next_state = transitions[rank]
            if next_state == BUST:
                a5 += probability
                continue
            shoe.draw(rank)
            v = play(shoe, next_state)
            shoe.put_back(rank)
            a0 += probability * v[0]
            a1 += probability * v[1]
            a2 += probability * v[2]
            a3 += probability * v[3]
            a4 += probability * v[4]
            a5 += probability * v[5]
        result = (a0, a1, a2, a3, a4, a5)
        memo[cache_key] = result
        return result
