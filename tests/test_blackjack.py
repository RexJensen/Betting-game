"""Tests for the blackjack engine.

The engine is heavily memoised and table driven for speed, so the important
tests here re-derive the same quantities with a deliberately naive, obviously
correct implementation (plain card lists, no memoisation, ace counted as one
and promoted) and check the two agree exactly.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest

from engine.blackjack.analyzer import Analyzer, evaluate
from engine.blackjack.cards import (ACE, BUST, EMPTY_STATE, HAND_TRANSITION,
                                    NUM_RANKS, RANKS, TEN, TWO_CARD_STATE, Shoe,
                                    state_is_soft, state_total)
from engine.blackjack.cli import main
from engine.blackjack.dealer import Dealer
from engine.blackjack.rules import RuleSet
from engine.blackjack.sensitivity import (COUNT_SYSTEMS, Evaluator, apply_deltas,
                                          base_counts, evaluate_shoe,
                                          find_threshold, make_strategy,
                                          preferred_direction,
                                          sensitivity_report, sweep)
from engine.blackjack.strategy import (BasicStrategy, DOUBLE, HIT, SPLIT, STAND,
                                       SURRENDER)

#: Rank indices by name, so tests never confuse the card "8" with index 8.
TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE = range(1, 9)

SMALL_SHOE = [2, 2, 2, 2, 2, 2, 2, 2, 2, 4]
LOW_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


# --------------------------------------------------------------------------
# naive reference implementation
# --------------------------------------------------------------------------
def naive_value(cards: Sequence[int]) -> Tuple[int, bool]:
    """Hand total counting every ace as one, then promoting one if it fits."""
    total = sum(LOW_VALUES[card] for card in cards)
    if ACE in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def naive_dealer_dist(counts: List[int], cards: List[int], h17: bool) -> List[float]:
    total, soft = naive_value(cards)
    if total > 21:
        return [0.0] * 5 + [1.0]
    if total >= 17 and not (h17 and soft and total == 17):
        out = [0.0] * 6
        out[total - 17] = 1.0
        return out
    remaining = sum(counts)
    out = [0.0] * 6
    for rank in range(NUM_RANKS):
        if not counts[rank]:
            continue
        probability = counts[rank] / remaining
        counts[rank] -= 1
        sub = naive_dealer_dist(counts, cards + [rank], h17)
        counts[rank] += 1
        for i in range(6):
            out[i] += probability * sub[i]
    return out


def naive_dealer_peeked(counts: List[int], upcard: int, h17: bool) -> List[float]:
    """Dealer distribution given the dealer has peeked and has no natural."""
    excluded = TEN if upcard == ACE else (ACE if upcard == TEN else None)
    denominator = sum(counts) - (counts[excluded] if excluded is not None else 0)
    out = [0.0] * 6
    for hole in range(NUM_RANKS):
        if not counts[hole] or hole == excluded:
            continue
        probability = counts[hole] / denominator
        counts[hole] -= 1
        sub = naive_dealer_dist(counts, [upcard, hole], h17)
        counts[hole] += 1
        for i in range(6):
            out[i] += probability * sub[i]
    return out


def naive_stand_ev(counts: List[int], cards: List[int], upcard: int,
                   h17: bool) -> float:
    total, _ = naive_value(cards)
    dist = naive_dealer_peeked(counts, upcard, h17)
    ev = dist[5]
    for i in range(5):
        dealer_total = 17 + i
        if dealer_total < total:
            ev += dist[i]
        elif dealer_total > total:
            ev -= dist[i]
    return ev


def naive_best_ev(counts: List[int], cards: List[int], upcard: int,
                  h17: bool) -> float:
    """Value of hitting or standing optimally, no doubling or splitting."""
    total, _ = naive_value(cards)
    if total > 21:
        return -1.0
    stand = naive_stand_ev(counts, cards, upcard, h17)
    if total == 21:
        return stand
    remaining = sum(counts)
    hit = 0.0
    for rank in range(NUM_RANKS):
        if not counts[rank]:
            continue
        probability = counts[rank] / remaining
        counts[rank] -= 1
        hit += probability * naive_best_ev(counts, cards + [rank], upcard, h17)
        counts[rank] += 1
    return max(stand, hit)


# --------------------------------------------------------------------------
# cards and shoe
# --------------------------------------------------------------------------
def state_of(cards: Sequence[int]) -> int:
    state = EMPTY_STATE
    for card in cards:
        state = HAND_TRANSITION[state][card]
        if state == BUST:
            return BUST
    return state


@pytest.mark.parametrize("cards", [
    [ACE], [ACE, ACE], [ACE, ACE, ACE], [ACE, TEN], [ACE, 4, 4], [TEN, TEN, ACE],
    [ACE, ACE, 8, 8], [5, 5, ACE], [ACE, 8, 2], [2, 3, 4, ACE, ACE],
])
def test_hand_transitions_match_naive_valuation(cards):
    state = state_of(cards)
    total, soft = naive_value(cards)
    if total > 21:
        assert state == BUST
    else:
        assert (state_total(state), state_is_soft(state)) == (total, soft)


def test_ten_ten_ten_busts():
    assert state_of([TEN, TEN, TEN]) == BUST


def test_shoe_draw_and_put_back_restore_the_key():
    shoe = Shoe.from_decks(6)
    key, counts, total = shoe.key, list(shoe.counts), shoe.total
    for rank in (ACE, 4, TEN, TEN):
        shoe.draw(rank)
    assert shoe.total == total - 4
    assert shoe.key != key
    for rank in (ACE, 4, TEN, TEN):
        shoe.put_back(rank)
    assert (shoe.key, shoe.counts, shoe.total) == (key, counts, total)


def test_shoe_keys_are_unique_per_composition():
    shoe = Shoe.from_decks(8)
    seen = {}
    for first in range(NUM_RANKS):
        shoe.draw(first)
        for second in range(NUM_RANKS):
            shoe.draw(second)
            composition = tuple(shoe.counts)
            assert seen.setdefault(shoe.key, composition) == composition
            shoe.put_back(second)
        shoe.put_back(first)


def test_shoe_rejects_bad_compositions():
    with pytest.raises(ValueError):
        Shoe([1, 2, 3])
    with pytest.raises(ValueError):
        Shoe([-1] + [4] * 9)


# --------------------------------------------------------------------------
# dealer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("h17", [False, True])
@pytest.mark.parametrize("upcard", range(NUM_RANKS))
def test_dealer_distribution_matches_naive_enumeration(upcard, h17):
    rules = RuleSet(decks=1, dealer_hits_soft_17=h17)
    shoe = Shoe(SMALL_SHOE)
    shoe.draw(upcard)
    engine = Dealer(rules).distribution(shoe, upcard)
    expected = naive_dealer_peeked(list(shoe.counts), upcard, h17)
    assert sum(engine) == pytest.approx(1.0)
    for got, want in zip(engine, expected):
        assert got == pytest.approx(want, abs=1e-12)


def test_dealer_probabilities_sum_to_one_for_a_six_deck_shoe():
    rules = RuleSet(decks=6)
    dealer = Dealer(rules)
    shoe = Shoe.from_decks(6)
    for upcard in range(NUM_RANKS):
        shoe.draw(upcard)
        assert sum(dealer.distribution(shoe, upcard)) == pytest.approx(1.0)
        shoe.put_back(upcard)


def test_dealer_bust_rates_match_published_six_deck_values():
    # Standard six deck S17 bust rates, given the dealer has no natural.
    expected = {1: 0.3535, 2: 0.3742, 3: 0.3958, 4: 0.4184, 5: 0.4228,
                6: 0.2619, 7: 0.2437, 8: 0.2292, TEN: 0.2302, ACE: 0.1670}
    dealer = Dealer(RuleSet(decks=6))
    shoe = Shoe.from_decks(6)
    for upcard, bust in expected.items():
        shoe.draw(upcard)
        assert dealer.distribution(shoe, upcard)[5] == pytest.approx(bust, abs=5e-4)
        shoe.put_back(upcard)


def test_dealer_natural_probability_only_applies_to_ace_and_ten():
    dealer = Dealer(RuleSet(decks=6))
    shoe = Shoe.from_decks(6)
    shoe.draw(ACE)
    assert dealer.natural_probability(shoe, ACE) == pytest.approx(96 / 311)
    shoe.put_back(ACE)
    shoe.draw(TEN)
    assert dealer.natural_probability(shoe, TEN) == pytest.approx(24 / 311)
    shoe.put_back(TEN)
    assert dealer.natural_probability(shoe, 5) == 0.0


# --------------------------------------------------------------------------
# player decisions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cards,upcard", [
    ((TEN, 5), TEN), ((8, 6), 4), ((ACE, 5), 5), ((2, 3), ACE), ((TEN, TEN), 6),
])
def test_stand_and_hit_evs_match_naive_enumeration(cards, upcard):
    rules = RuleSet(decks=1)
    shoe = Shoe(SMALL_SHOE)
    analyzer = Analyzer(rules, shoe, None)
    for card in cards + (upcard,):
        shoe.draw(card)
    state = TWO_CARD_STATE[cards[0]][cards[1]]
    counts = list(shoe.counts)
    assert analyzer._stand_ev(upcard, state) == pytest.approx(
        naive_stand_ev(counts, list(cards), upcard, False), abs=1e-12)

    naive_hit = 0.0
    remaining = sum(counts)
    for rank in range(NUM_RANKS):
        if not counts[rank]:
            continue
        probability = counts[rank] / remaining
        counts[rank] -= 1
        naive_hit += probability * naive_best_ev(counts, list(cards) + [rank],
                                                 upcard, False)
        counts[rank] += 1
    assert analyzer._hit_ev(upcard, state) == pytest.approx(naive_hit, abs=1e-12)


def test_doubling_is_exactly_twice_a_single_card_draw():
    rules = RuleSet(decks=1)
    shoe = Shoe(SMALL_SHOE)
    analyzer = Analyzer(rules, shoe, None)
    upcard, cards = 5, (5, 5)  # eleven against a six
    for card in cards + (upcard,):
        shoe.draw(card)
    state = TWO_CARD_STATE[cards[0]][cards[1]]
    counts = list(shoe.counts)
    remaining = sum(counts)
    expected = 0.0
    for rank in range(NUM_RANKS):
        if not counts[rank]:
            continue
        probability = counts[rank] / remaining
        counts[rank] -= 1
        total, _ = naive_value(list(cards) + [rank])
        value = -1.0 if total > 21 else naive_stand_ev(
            counts, list(cards) + [rank], upcard, False)
        counts[rank] += 1
        expected += 2.0 * probability * value
    assert analyzer._double_ev(upcard, state) == pytest.approx(expected, abs=1e-12)


def test_split_aces_receive_exactly_one_card():
    rules = RuleSet(decks=6, hit_split_aces=False)
    shoe = Shoe.from_decks(6)
    analyzer = Analyzer(rules, shoe, None)
    for card in (ACE, ACE, 5):
        shoe.draw(card)
    # Each hand stands on its single drawn card, so the split is worth twice
    # the average stand value of ace plus one card.
    expected = 0.0
    remaining = shoe.total
    for rank in range(NUM_RANKS):
        probability = shoe.counts[rank] / remaining
        shoe.draw(rank)
        expected += probability * analyzer._stand_ev(5, TWO_CARD_STATE[ACE][rank])
        shoe.put_back(rank)
    assert analyzer._split_ev(5, ACE) == pytest.approx(2.0 * expected, abs=1e-12)


def test_hitting_split_aces_is_worth_more_than_standing_on_them():
    shoe_counts = base_counts(RuleSet(decks=6))
    one_card = evaluate_shoe(RuleSet(decks=6, hit_split_aces=False), shoe_counts).ev
    hit_them = evaluate_shoe(RuleSet(decks=6, hit_split_aces=True), shoe_counts).ev
    assert hit_them > one_card


# --------------------------------------------------------------------------
# strategy chart
# --------------------------------------------------------------------------
def test_chart_resolves_representative_cells():
    strategy = BasicStrategy(RuleSet(decks=6))
    decide = strategy.decide
    hard16 = TWO_CARD_STATE[TEN][5]
    assert decide(hard16, TEN, None, True, False, False) == HIT
    assert decide(hard16, 4, None, True, False, False) == STAND
    soft18 = TWO_CARD_STATE[ACE][6]
    assert decide(soft18, 2, None, True, False, False) == DOUBLE   # Ds vs three
    assert decide(soft18, 2, None, False, False, False) == STAND   # ... else stand
    assert decide(soft18, 8, None, True, False, False) == HIT      # vs nine
    eleven = TWO_CARD_STATE[5][4]
    assert decide(eleven, ACE, None, True, False, False) == HIT    # S17
    assert decide(eleven, 8, None, True, False, False) == DOUBLE
    assert decide(TWO_CARD_STATE[7][7], TEN, 7, False, True, False) == SPLIT
    assert decide(TWO_CARD_STATE[TEN][TEN], 5, TEN, False, True, False) == STAND
    assert decide(TWO_CARD_STATE[4][4], 5, 4, True, True, False) == DOUBLE  # 5,5


def test_h17_and_surrender_overlays_change_the_expected_cells():
    h17 = BasicStrategy(RuleSet(decks=6, dealer_hits_soft_17=True))
    eleven = TWO_CARD_STATE[5][4]
    assert h17.decide(eleven, ACE, None, True, False, False) == DOUBLE
    surrender = BasicStrategy(RuleSet(decks=6, late_surrender=True))
    hard16 = TWO_CARD_STATE[TEN][5]
    assert surrender.decide(hard16, TEN, None, True, False, True) == SURRENDER
    assert surrender.decide(hard16, TEN, None, True, False, False) == HIT
    assert surrender.decide(TWO_CARD_STATE[7][7], TEN, 7, False, True, True) == SPLIT


def test_das_toggles_the_marginal_pair_splits():
    with_das = BasicStrategy(RuleSet(decks=6, double_after_split=True))
    without = BasicStrategy(RuleSet(decks=6, double_after_split=False))
    pair_fours = TWO_CARD_STATE[3][3]
    assert with_das.decide(pair_fours, 5, 3, True, True, False) == SPLIT
    assert without.decide(pair_fours, 5, 3, True, True, False) != SPLIT


# --------------------------------------------------------------------------
# round expected value
# --------------------------------------------------------------------------
def test_six_deck_house_edge_matches_the_published_figure():
    rules = RuleSet(decks=6)
    result = evaluate(rules, Shoe.from_decks(6), BasicStrategy(rules))
    assert 0.39 <= result.house_edge_pct <= 0.43


def test_optimal_play_beats_the_fixed_chart_but_only_barely():
    counts = base_counts(RuleSet(decks=2))
    rules = RuleSet(decks=2)
    basic = evaluate_shoe(rules, counts, "basic").ev
    optimal = evaluate_shoe(rules, counts, "optimal").ev
    assert optimal > basic
    assert optimal - basic < 0.002


@pytest.mark.parametrize("change,worse", [
    ({"blackjack_pays": 1.2}, 0.013),          # 6:5 costs well over a percent
    ({"dealer_hits_soft_17": True}, 0.0015),   # H17 costs about 0.2%
    ({"double_rule": "10-11"}, 0.001),
    ({"double_after_split": False}, 0.0005),
])
def test_stingier_rules_cost_the_player(change, worse):
    counts = base_counts(RuleSet(decks=6))
    baseline = evaluate_shoe(RuleSet(decks=6), counts).ev
    stingy = evaluate_shoe(RuleSet(decks=6, **change), counts).ev
    assert baseline - stingy > worse


def test_insurance_is_a_bad_bet_until_the_tens_pile_up():
    rules = RuleSet(decks=6)
    counts = base_counts(rules)
    fair = Analyzer(rules, Shoe(counts), None).insurance_ev()
    assert fair == pytest.approx(-0.074, abs=0.002)
    rich = Analyzer(rules, Shoe(apply_deltas(counts, {TEN: 40})), None).insurance_ev()
    assert rich > 0


def test_a_natural_pays_the_stated_odds():
    rules = RuleSet(decks=8)
    shoe = Shoe.from_decks(8)
    analyzer = Analyzer(rules, shoe, BasicStrategy(rules))
    for card in (ACE, TEN, 5):
        shoe.draw(card)
    case = analyzer._evaluate_case(ACE, TEN, 5, 1.0, False)
    assert case.ev == pytest.approx(1.5)  # dealer showing a six cannot have one


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def six_deck_report():
    return sensitivity_report(RuleSet(decks=6), "basic", workers=4)


@pytest.fixture(scope="session")
def single_deck_report():
    return sensitivity_report(RuleSet(decks=1), "basic", workers=4)


def test_effects_of_removal_have_the_classic_signs_and_ordering(six_deck_report):
    effects = {effect.rank: effect.removal_effect
               for effect in six_deck_report.effects}
    # Taking a low card out helps the player, taking a high card out hurts.
    for rank in (TWO, THREE, FOUR, FIVE, SIX, SEVEN):
        assert effects[rank] > 0, RANKS[rank]
    for rank in (ACE, NINE, TEN):
        assert effects[rank] < 0, RANKS[rank]
    assert abs(effects[EIGHT]) < 0.0002        # the eight is nearly neutral
    assert effects[FIVE] > effects[FOUR] > effects[TWO]
    assert effects[FIVE] > -effects[ACE] > -effects[TEN]


def test_removal_effects_rescale_to_griffins_single_deck_table(single_deck_report):
    # Griffin's classic effects of removal for a single deck, in percent.
    published = {ACE: -0.61, TWO: 0.38, THREE: 0.44, FOUR: 0.55, FIVE: 0.69,
                 SIX: 0.46, SEVEN: 0.28, EIGHT: 0.00, NINE: -0.18, TEN: -0.51}
    for rank, expected in published.items():
        got = 100 * single_deck_report.effects[rank].removal_effect
        assert got == pytest.approx(expected, abs=0.09), RANKS[rank]


def test_adding_a_card_is_the_mirror_of_removing_one(six_deck_report):
    for effect in six_deck_report.effects:
        assert effect.removal_effect == pytest.approx(-effect.addition_effect,
                                                      rel=0.05, abs=1e-5)


def test_the_linear_estimate_tracks_the_exact_answer_for_small_changes(
        single_deck_report):
    evaluator = Evaluator(RuleSet(decks=1), "basic")
    deltas = {ACE: 2, FIVE: -2}
    exact = evaluator.ev(apply_deltas(single_deck_report.counts, deltas))
    assert single_deck_report.linear_ev(deltas) == pytest.approx(exact, abs=0.004)


def test_hi_lo_correlates_almost_perfectly_with_the_removal_effects(
        single_deck_report):
    report = single_deck_report
    assert report.betting_correlation(COUNT_SYSTEMS["Hi-Lo"]) > 0.94
    assert report.betting_correlation(COUNT_SYSTEMS["Hi-Opt I"]) < \
        report.betting_correlation(COUNT_SYSTEMS["Hi-Lo"])
    # Removing every rank in proportion changes depth, not composition.
    weighted = sum(count * effect.removal_effect
                   for count, effect in zip(report.counts, report.effects))
    assert abs(weighted) < 0.002


def test_threshold_search_finds_the_smallest_count_that_flips_the_game(
        six_deck_report):
    evaluator = Evaluator(RuleSet(decks=6), "basic", workers=2)
    threshold = find_threshold(FIVE, "remove", evaluator, six_deck_report)
    assert threshold.cards is not None and 1 <= threshold.cards <= 8
    assert threshold.ev >= 0
    fewer = evaluator.ev(apply_deltas(six_deck_report.counts,
                                      {FIVE: -(threshold.cards - 1)}))
    assert fewer < 0  # one card fewer does not get there
    assert "removed" in threshold.describe()


def test_threshold_search_reports_when_the_target_is_out_of_reach(
        six_deck_report):
    evaluator = Evaluator(RuleSet(decks=6), "basic")
    threshold = find_threshold(TEN, "remove", evaluator, six_deck_report,
                               limit=4)
    assert threshold.cards is None
    assert threshold.ev < six_deck_report.baseline_ev
    assert "never reaches" in threshold.describe()


def test_a_shoe_that_already_clears_the_target_needs_no_cards(six_deck_report):
    evaluator = Evaluator(RuleSet(decks=6), "basic")
    threshold = find_threshold(ACE, "add", evaluator, six_deck_report,
                               target_ev=-0.05)
    assert threshold.cards == 0
    assert threshold.evaluations == []  # the answer needs no evaluation at all


def test_a_single_deck_game_is_already_in_the_players_favour(single_deck_report):
    # Six decks costs the player about half a percent versus one deck.
    assert single_deck_report.baseline_ev > 0


def test_preferred_direction_matches_the_sign_of_the_effect(six_deck_report):
    assert preferred_direction(six_deck_report, ACE) == "add"
    assert preferred_direction(six_deck_report, TEN) == "add"
    assert preferred_direction(six_deck_report, FIVE) == "remove"
    assert preferred_direction(six_deck_report, SIX) == "remove"


def test_sweeping_a_rank_is_monotone_in_the_helpful_direction(single_deck_report):
    evaluator = Evaluator(RuleSet(decks=1), "basic", workers=4)
    curve = sweep(FIVE, range(-3, 1), evaluator, single_deck_report.counts)
    values = [ev for _, ev in curve]
    assert values == sorted(values, reverse=True)  # fewer fives is better
    assert len(curve) == 4


def test_sweep_skips_changes_that_would_empty_the_rank(single_deck_report):
    evaluator = Evaluator(RuleSet(decks=1), "basic")
    curve = sweep(ACE, [-6, -4, 0], evaluator, single_deck_report.counts)
    assert [delta for delta, _ in curve] == [-4, 0]


def test_apply_deltas_refuses_to_remove_cards_that_are_not_there():
    with pytest.raises(ValueError):
        apply_deltas(base_counts(RuleSet(decks=1)), {ACE: -5})


def test_evaluator_caches_repeated_compositions():
    evaluator = Evaluator(RuleSet(decks=1), "basic")
    counts = base_counts(RuleSet(decks=1))
    first = evaluator.ev(counts)
    assert evaluator.ev(counts) == first
    assert len(evaluator._cache) == 1


def test_make_strategy_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        make_strategy(RuleSet(decks=1), "psychic")


def test_rules_validate_their_arguments():
    with pytest.raises(ValueError):
        RuleSet(decks=0)
    with pytest.raises(ValueError):
        RuleSet(double_rule="whenever")
    with pytest.raises(ValueError):
        RuleSet(max_split_hands=3)
    with pytest.raises(NotImplementedError):
        RuleSet(peek=False)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def test_cli_edge_command_reports_a_house_edge(capsys):
    assert main(["edge", "--decks", "6"]) == 0
    out = capsys.readouterr().out
    assert "house edge 0.4" in out
    assert "Insurance bet" in out


def test_cli_whatif_command_reports_the_difference(capsys):
    assert main(["whatif", "--decks", "6", "--change", "A:+5"]) == 0
    out = capsys.readouterr().out
    assert "+5 As" in out
    assert "player edge" in out


def test_cli_chart_command_prints_the_grid(capsys):
    assert main(["chart", "--decks", "6"]) == 0
    out = capsys.readouterr().out
    assert "Hard totals" in out and "Pairs" in out


def test_cli_rejects_a_nonsense_change():
    with pytest.raises(SystemExit):
        main(["whatif", "--change", "Z:+1"])
    with pytest.raises(SystemExit):
        main(["whatif", "--change", "A5"])
