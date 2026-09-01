"""Exact combinatorial analysis of one round of blackjack.

Every player hand and dealer upcard is enumerated with its exact dealing
probability, and every decision is evaluated against the exact remaining shoe:
there is no simulation and no infinite-deck approximation, so the expected
value returned here is a deterministic number rather than an estimate.

Two documented approximations remain, both standard for combinatorial
analysers:

* post-split hands are valued independently of one another (each hand sees the
  shoe minus the split cards and the dealer upcard, not minus its sibling's
  draws), and resplitting is budgeted per hand rather than globally;
* the "dealer has peeked" conditioning is applied to the shoe as it stands when
  the player acts, rather than to the shoe as it stood at the peek.

Both are worth well under 0.01% and, more importantly, they are held constant
across every composition compared here, so they cancel out of an effect of
removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cards import (ACE, BUST, EMPTY_STATE, HAND_TRANSITION, NUM_RANKS, RANKS,
                    TEN, TWO_CARD_STATE, Shoe, state_is_soft, state_total)
from .dealer import Dealer
from .rules import RuleSet
from .strategy import BasicStrategy, DOUBLE, HIT, SPLIT, STAND, SURRENDER


@dataclass
class HandCase:
    """One (player two cards, dealer upcard) starting position."""

    cards: Tuple[int, int]
    upcard: int
    probability: float
    action_evs: Dict[int, float]
    action: int
    ev: float

    @property
    def label(self) -> str:
        return "%s%s vs %s" % (RANKS[self.cards[0]], RANKS[self.cards[1]],
                               RANKS[self.upcard])


@dataclass
class RoundResult:
    """Expected value of one round, per unit of the original wager."""

    ev: float
    rules: RuleSet
    strategy_name: str
    cases: List[HandCase]

    @property
    def house_edge(self) -> float:
        return -self.ev

    @property
    def house_edge_pct(self) -> float:
        return -100.0 * self.ev


class Analyzer:
    """Evaluates a round for one fixed shoe composition."""

    def __init__(self, rules: RuleSet, shoe: Shoe,
                 strategy: Optional[BasicStrategy] = None):
        self.rules = rules
        self.shoe = shoe
        self.strategy = strategy  # None -> composition dependent optimal play
        self.dealer = Dealer(rules)
        self._hit_memo: Dict[int, float] = {}
        self._resplits = 1 if rules.max_split_hands >= 4 else 0

    # -- public API ---------------------------------------------------------
    def round_ev(self, collect_actions: bool = False) -> RoundResult:
        """Enumerate every deal and return the exact expected value."""
        shoe = self.shoe
        counts = shoe.counts
        cases: List[HandCase] = []
        total_ev = 0.0

        for first in range(NUM_RANKS):
            if counts[first] == 0:
                continue
            p_first = counts[first] / shoe.total
            shoe.draw(first)
            for second in range(first, NUM_RANKS):
                if counts[second] == 0:
                    continue
                p_second = counts[second] / shoe.total
                # {first, second} is unordered, so both deal orders count.
                p_hand = p_first * p_second * (1.0 if first == second else 2.0)
                shoe.draw(second)
                for up in range(NUM_RANKS):
                    if counts[up] == 0:
                        continue
                    probability = p_hand * counts[up] / shoe.total
                    shoe.draw(up)
                    case = self._evaluate_case(first, second, up, probability,
                                               collect_actions)
                    shoe.put_back(up)
                    total_ev += probability * case.ev
                    if collect_actions:
                        cases.append(case)
                shoe.put_back(second)
            shoe.put_back(first)

        name = self.strategy.name if self.strategy is not None else "optimal (CD)"
        return RoundResult(ev=total_ev, rules=self.rules, strategy_name=name,
                           cases=cases)

    def insurance_ev(self) -> float:
        """EV per unit staked on insurance, offered when the dealer shows an
        ace and the player has seen only their own two cards.  Averaged over
        every player hand, weighted by how often it is dealt."""
        shoe = self.shoe
        counts = shoe.counts
        weight_total = 0.0
        ev_total = 0.0
        for first in range(NUM_RANKS):
            if counts[first] == 0:
                continue
            p_first = counts[first] / shoe.total
            shoe.draw(first)
            for second in range(first, NUM_RANKS):
                if counts[second] == 0:
                    continue
                p_second = counts[second] / shoe.total
                p_hand = p_first * p_second * (1.0 if first == second else 2.0)
                shoe.draw(second)
                if counts[ACE]:
                    probability = p_hand * counts[ACE] / shoe.total
                    shoe.draw(ACE)
                    p_ten = counts[TEN] / shoe.total
                    ev_total += probability * (2.0 * p_ten - (1.0 - p_ten))
                    weight_total += probability
                    shoe.put_back(ACE)
                shoe.put_back(second)
            shoe.put_back(first)
        return ev_total / weight_total if weight_total else 0.0

    # -- one starting position ---------------------------------------------
    def _evaluate_case(self, first: int, second: int, up: int,
                       probability: float, collect_actions: bool) -> HandCase:
        shoe = self.shoe
        rules = self.rules
        state = TWO_CARD_STATE[first][second]
        p_natural = self.dealer.natural_probability(shoe, up)

        player_natural = state_total(state) == 21
        if player_natural:
            ev = (1.0 - p_natural) * rules.blackjack_pays
            return HandCase((first, second), up, probability,
                            {STAND: ev} if collect_actions else {}, STAND, ev)

        pair_rank = first if first == second else None
        can_split = pair_rank is not None and rules.max_split_hands >= 2
        can_surrender = rules.late_surrender
        can_double = rules.may_double(state_total(state), state_is_soft(state),
                                      2, after_split=False)

        evs: Dict[int, float] = {}
        if collect_actions or self.strategy is None:
            evs[STAND] = self._stand_ev(up, state)
            evs[HIT] = self._hit_ev(up, state)
            if can_double:
                evs[DOUBLE] = self._double_ev(up, state)
            if can_split:
                evs[SPLIT] = self._split_ev(up, pair_rank)
            if can_surrender:
                evs[SURRENDER] = -0.5
            action = max(evs, key=evs.__getitem__)
            play_ev = evs[action]
        else:
            action = self.strategy.decide(state, up, pair_rank, can_double,
                                          can_split, can_surrender)
            play_ev = self._action_ev(action, up, state, pair_rank)

        ev = p_natural * -1.0 + (1.0 - p_natural) * play_ev
        return HandCase((first, second), up, probability, evs, action, ev)

    def _action_ev(self, action: int, up: int, state: int,
                   pair_rank: Optional[int]) -> float:
        if action == STAND:
            return self._stand_ev(up, state)
        if action == HIT:
            return self._hit_ev(up, state)
        if action == DOUBLE:
            return self._double_ev(up, state)
        if action == SPLIT:
            return self._split_ev(up, pair_rank)
        return -0.5

    # -- action values ------------------------------------------------------
    def _stand_ev(self, up: int, state: int) -> float:
        total = state >> 1
        dist = self.dealer.distribution(self.shoe, up)
        win = dist[5]
        lose = 0.0
        for i in range(5):
            dealer_total = 17 + i
            if dealer_total < total:
                win += dist[i]
            elif dealer_total > total:
                lose += dist[i]
        return win - lose

    def _hit_ev(self, up: int, state: int) -> float:
        shoe = self.shoe
        key = (((shoe.key << 4) | up) << 6) | state
        memo = self._hit_memo
        cached = memo.get(key)
        if cached is not None:
            return cached

        counts = shoe.counts
        remaining = shoe.total
        transitions = HAND_TRANSITION[state]
        total_ev = 0.0
        for rank in range(NUM_RANKS):
            count = counts[rank]
            if count == 0:
                continue
            probability = count / remaining
            next_state = transitions[rank]
            if next_state == BUST:
                total_ev -= probability
                continue
            shoe.draw(rank)
            total_ev += probability * self._draw_ev(up, next_state)
            shoe.put_back(rank)
        memo[key] = total_ev
        return total_ev

    def _draw_ev(self, up: int, state: int) -> float:
        """Value of a three-or-more card hand: only hit or stand remain."""
        if state >= 42:  # 21, hard or soft
            return self._stand_ev(up, state)
        if self.strategy is None:
            return max(self._stand_ev(up, state), self._hit_ev(up, state))
        action = self.strategy.decide(state, up, None, False, False, False)
        if action == STAND:
            return self._stand_ev(up, state)
        return self._hit_ev(up, state)

    def _double_ev(self, up: int, state: int) -> float:
        shoe = self.shoe
        counts = shoe.counts
        remaining = shoe.total
        transitions = HAND_TRANSITION[state]
        total_ev = 0.0
        for rank in range(NUM_RANKS):
            count = counts[rank]
            if count == 0:
                continue
            probability = count / remaining
            next_state = transitions[rank]
            if next_state == BUST:
                total_ev -= 2.0 * probability
                continue
            shoe.draw(rank)
            total_ev += 2.0 * probability * self._stand_ev(up, next_state)
            shoe.put_back(rank)
        return total_ev

    # -- splitting ----------------------------------------------------------
    def _split_ev(self, up: int, pair_rank: int) -> float:
        return 2.0 * self._split_slot(up, pair_rank, self._resplits)

    def _split_slot(self, up: int, pair_rank: int, resplits_left: int) -> float:
        """EV of one hand that starts with a single ``pair_rank`` card."""
        shoe = self.shoe
        counts = shoe.counts
        remaining = shoe.total
        base_state = HAND_TRANSITION[EMPTY_STATE][pair_rank]
        transitions = HAND_TRANSITION[base_state]
        may_resplit = resplits_left > 0 and (pair_rank != ACE or self.rules.resplit_aces)

        total_ev = 0.0
        for rank in range(NUM_RANKS):
            count = counts[rank]
            if count == 0:
                continue
            probability = count / remaining
            shoe.draw(rank)
            if rank == pair_rank and may_resplit:
                resplit_ev = 2.0 * self._split_slot(up, pair_rank, resplits_left - 1)
                if self.strategy is None:
                    value = max(resplit_ev,
                                self._post_split_ev(up, transitions[rank], pair_rank))
                elif self.strategy.should_split(pair_rank, up):
                    value = resplit_ev
                else:
                    value = self._post_split_ev(up, transitions[rank], pair_rank)
            else:
                value = self._post_split_ev(up, transitions[rank], pair_rank)
            shoe.put_back(rank)
            total_ev += probability * value
        return total_ev

    def _post_split_ev(self, up: int, state: int, pair_rank: int) -> float:
        if pair_rank == ACE and not self.rules.hit_split_aces:
            return self._stand_ev(up, state)  # one card only
        return self._two_card_ev(up, state, after_split=True)

    def _two_card_ev(self, up: int, state: int, after_split: bool) -> float:
        total = state >> 1
        if total == 21:
            return self._stand_ev(up, state)
        can_double = self.rules.may_double(total, bool(state & 1), 2, after_split)
        if self.strategy is None:
            best = max(self._stand_ev(up, state), self._hit_ev(up, state))
            if can_double:
                best = max(best, self._double_ev(up, state))
            return best
        action = self.strategy.decide(state, up, None, can_double, False, False)
        if action == STAND:
            return self._stand_ev(up, state)
        if action == DOUBLE:
            return self._double_ev(up, state)
        return self._hit_ev(up, state)


def evaluate(rules: RuleSet, shoe: Shoe,
             strategy: Optional[BasicStrategy] = None,
             collect_actions: bool = False) -> RoundResult:
    """Convenience wrapper: exact EV of one round for ``shoe`` under ``rules``."""
    return Analyzer(rules, shoe, strategy).round_ev(collect_actions=collect_actions)
