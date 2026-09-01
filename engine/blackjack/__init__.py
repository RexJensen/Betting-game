"""An exact blackjack engine and house edge sensitivity analysis.

The engine enumerates every deal and every decision against the exact
remaining shoe, so the expected values it reports are computed rather than
simulated.  Its purpose here is to answer composition questions: what one card
of each rank is worth to the player, and how many of them have to be added to
or taken out of the shoe before the game turns in the player's favour.

    from engine.blackjack import BasicStrategy, RuleSet, Shoe, evaluate

    rules = RuleSet(decks=6)
    print(evaluate(rules, Shoe.from_decks(6), BasicStrategy(rules)).house_edge_pct)

Leaving the strategy out instead values the round under composition dependent
optimal play.
"""

from .analyzer import Analyzer, HandCase, RoundResult, evaluate
from .cards import ACE, DECK_COMPOSITION, RANKS, TEN, Shoe
from .dealer import Dealer
from .rules import RuleSet
from .sensitivity import (COUNT_SYSTEMS, CardEffect, Evaluator,
                          SensitivityReport, Threshold, apply_deltas,
                          base_counts, evaluate_shoe, find_threshold,
                          find_thresholds, preferred_direction,
                          sensitivity_report, sweep)
from .strategy import BasicStrategy

__all__ = [
    "ACE", "COUNT_SYSTEMS", "DECK_COMPOSITION", "RANKS", "TEN",
    "Analyzer", "BasicStrategy", "CardEffect", "Dealer", "Evaluator",
    "HandCase", "RoundResult", "RuleSet", "SensitivityReport", "Shoe",
    "Threshold", "apply_deltas", "base_counts", "evaluate", "evaluate_shoe",
    "find_threshold", "find_thresholds", "preferred_direction",
    "sensitivity_report", "sweep",
]
