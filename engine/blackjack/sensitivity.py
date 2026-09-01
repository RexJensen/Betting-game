"""How the house edge responds to the composition of the shoe.

The central quantity is the *effect of removal* (EOR): the change in the
player's expected value caused by taking one card of a given rank out of the
shoe (or, with the sign flipped, adding one).  Everything else here -- how many
aces make the game beatable, how many sixes have to go, what extra nines are
worth -- falls out of re-running the exact analysis on a modified shoe.

The player's strategy is held fixed (standard basic strategy) by default, which
is what makes the numbers comparable: the edge moves because the cards moved,
not because the player started playing differently.  Pass ``optimal`` strategy
mode to let the player also re-optimise against the new composition.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .analyzer import Analyzer, RoundResult
from .cards import ACE, DECK_COMPOSITION, NUM_RANKS, RANKS, TEN, Shoe
from .rules import RuleSet
from .strategy import BasicStrategy

#: Count tag vectors, indexed by rank (A,2..9,T), for betting correlation.
COUNT_SYSTEMS: Dict[str, Tuple[float, ...]] = {
    "Hi-Lo":     (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1),
    "Hi-Opt I":  (0, 0, 1, 1, 1, 1, 0, 0, 0, -1),
    "Hi-Opt II": (0, 1, 1, 2, 2, 1, 1, 0, 0, -2),
    "Omega II":  (0, 1, 1, 2, 2, 2, 1, 0, -1, -2),
    "Zen Count": (-1, 1, 1, 2, 2, 2, 1, 0, 0, -2),
}


def make_strategy(rules: RuleSet, mode: str) -> Optional[BasicStrategy]:
    """``"basic"`` -> the fixed chart, ``"optimal"`` -> composition dependent."""
    if mode == "basic":
        return BasicStrategy(rules)
    if mode == "optimal":
        return None
    raise ValueError("strategy mode must be 'basic' or 'optimal'")


def evaluate_shoe(rules: RuleSet, counts: Sequence[int], mode: str = "basic",
                  collect_actions: bool = False) -> RoundResult:
    strategy = make_strategy(rules, mode)
    return Analyzer(rules, Shoe(counts), strategy).round_ev(collect_actions)


def base_counts(rules: RuleSet) -> List[int]:
    return [count * rules.decks for count in DECK_COMPOSITION]


def apply_deltas(counts: Sequence[int], deltas: Dict[int, int]) -> List[int]:
    modified = list(counts)
    for rank, delta in deltas.items():
        modified[rank] += delta
        if modified[rank] < 0:
            raise ValueError("cannot remove %d %s cards: only %d in the shoe"
                             % (-delta, RANKS[rank], counts[rank]))
    return modified


# -- parallel helper --------------------------------------------------------
_JOB_CONTEXT: Dict[str, object] = {}


def _job_init(rules: RuleSet, mode: str) -> None:  # pragma: no cover - subprocess
    _JOB_CONTEXT["rules"] = rules
    _JOB_CONTEXT["mode"] = mode


def _job_eval(counts: Sequence[int]) -> float:  # pragma: no cover - subprocess
    rules = _JOB_CONTEXT["rules"]
    mode = _JOB_CONTEXT["mode"]
    return evaluate_shoe(rules, counts, mode).ev


class Evaluator:
    """Evaluates shoe compositions, memoised, optionally across processes."""

    def __init__(self, rules: RuleSet, mode: str = "basic", workers: int = 1):
        self.rules = rules
        self.mode = mode
        self.workers = max(1, workers)
        self._cache: Dict[Tuple[int, ...], float] = {}

    def ev(self, counts: Sequence[int]) -> float:
        return self.ev_many([counts])[0]

    def ev_many(self, batch: Sequence[Sequence[int]]) -> List[float]:
        keys = [tuple(counts) for counts in batch]
        pending = [key for key in dict.fromkeys(keys) if key not in self._cache]
        if pending:
            if self.workers > 1 and len(pending) > 1:
                with multiprocessing.Pool(
                    processes=min(self.workers, len(pending)),
                    initializer=_job_init,
                    initargs=(self.rules, self.mode),
                ) as pool:
                    values = pool.map(_job_eval, pending)
            else:
                values = [evaluate_shoe(self.rules, key, self.mode).ev
                          for key in pending]
            self._cache.update(zip(pending, values))
        return [self._cache[key] for key in keys]


# -- effects of removal -----------------------------------------------------
@dataclass
class CardEffect:
    """What one card of a rank is worth to the player, in EV per unit bet."""

    rank: int
    baseline_ev: float
    ev_removed: float
    ev_added: float

    @property
    def removal_effect(self) -> float:
        """Change in player EV when one card of this rank leaves the shoe."""
        return self.ev_removed - self.baseline_ev

    @property
    def addition_effect(self) -> float:
        return self.ev_added - self.baseline_ev

    @property
    def name(self) -> str:
        return RANKS[self.rank]


@dataclass
class SensitivityReport:
    rules: RuleSet
    mode: str
    counts: List[int]
    baseline_ev: float
    effects: List[CardEffect]

    @property
    def house_edge_pct(self) -> float:
        return -100.0 * self.baseline_ev

    def effect(self, rank: int) -> CardEffect:
        return self.effects[rank]

    def linear_ev(self, deltas: Dict[int, int]) -> float:
        """First order estimate of EV after adding/removing several cards."""
        ev = self.baseline_ev
        for rank, delta in deltas.items():
            effect = self.effects[rank]
            per_card = (effect.addition_effect if delta > 0
                        else effect.removal_effect)
            ev += abs(delta) * per_card
        return ev

    def betting_correlation(self, tags: Sequence[float]) -> float:
        """Correlation of a count system's tags with the removal effects.

        A card's tag is positive when seeing it leave the shoe should make the
        player bet more, so it is correlated against the gain from removing
        that card, weighted by how many of them the shoe holds.
        """
        weights = [float(count) for count in self.counts]
        removal = [effect.removal_effect for effect in self.effects]
        return _weighted_correlation(weights, removal, list(tags))


def _weighted_correlation(weights: Sequence[float], xs: Sequence[float],
                          ys: Sequence[float]) -> float:
    total = sum(weights)
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    cov = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
    var_x = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    var_y = sum(w * (y - mean_y) ** 2 for w, y in zip(weights, ys))
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def sensitivity_report(rules: RuleSet, mode: str = "basic", workers: int = 1,
                       counts: Optional[Sequence[int]] = None,
                       evaluator: Optional[Evaluator] = None) -> SensitivityReport:
    """Remove one card of each rank, then add one of each, exactly."""
    counts = list(counts) if counts is not None else base_counts(rules)
    evaluator = evaluator or Evaluator(rules, mode, workers)

    batch: List[List[int]] = [list(counts)]
    for rank in range(NUM_RANKS):
        batch.append(apply_deltas(counts, {rank: -1}))
    for rank in range(NUM_RANKS):
        batch.append(apply_deltas(counts, {rank: +1}))

    values = evaluator.ev_many(batch)
    baseline = values[0]
    effects = [
        CardEffect(rank=rank, baseline_ev=baseline,
                   ev_removed=values[1 + rank], ev_added=values[11 + rank])
        for rank in range(NUM_RANKS)
    ]
    return SensitivityReport(rules=rules, mode=mode, counts=counts,
                             baseline_ev=baseline, effects=effects)


# -- "how many cards does it take" -----------------------------------------
@dataclass
class Threshold:
    rank: int
    direction: str          # "add" or "remove"
    target_ev: float
    cards: Optional[int]    # None when the target cannot be reached
    ev: Optional[float]
    limit: int
    evaluations: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return RANKS[self.rank]

    def describe(self) -> str:
        verb = "added" if self.direction == "add" else "removed"
        if self.cards is None:
            return ("%s: never reaches EV %+.4f%% -- even with %d %s (EV %+.4f%%)"
                    % (self.name, 100 * self.target_ev, self.limit, verb,
                       100 * (self.ev if self.ev is not None else float("nan"))))
        return ("%s: %d %s (EV %+.4f%%)"
                % (self.name, self.cards, verb, 100 * self.ev))


def find_threshold(rank: int, direction: str, evaluator: Evaluator,
                   report: SensitivityReport, target_ev: float = 0.0,
                   limit: Optional[int] = None) -> Threshold:
    """Smallest number of cards of ``rank`` to add/remove to reach ``target_ev``.

    The search starts from the linear EOR estimate, brackets the answer by
    doubling, then bisects, so it costs a handful of full evaluations rather
    than one per card.  Expected value is monotone in the count of a single
    rank over the range that matters here, and the returned answer is checked
    against one fewer card, so the boundary is exact either way.
    """
    if direction not in ("add", "remove"):
        raise ValueError("direction must be 'add' or 'remove'")
    counts = report.counts
    sign = 1 if direction == "add" else -1
    hard_limit = counts[rank] if direction == "remove" else 10 * counts[rank]
    limit = min(limit, hard_limit) if limit is not None else hard_limit

    evaluations: List[Tuple[int, float]] = []
    cache: Dict[int, float] = {0: report.baseline_ev}

    def ev_at(cards: int) -> float:
        if cards not in cache:
            cache[cards] = evaluator.ev(apply_deltas(counts, {rank: sign * cards}))
            evaluations.append((cards, cache[cards]))
        return cache[cards]

    if report.baseline_ev >= target_ev:
        return Threshold(rank, direction, target_ev, 0, report.baseline_ev,
                         limit, evaluations)
    if limit < 1:
        return Threshold(rank, direction, target_ev, None, report.baseline_ev,
                         limit, evaluations)

    effect = report.effects[rank]
    per_card = effect.addition_effect if sign > 0 else effect.removal_effect
    if per_card > 0:
        guess = max(1, int((target_ev - report.baseline_ev) / per_card))
    else:
        guess = limit
    high = min(guess, limit)

    # Bracket: `low` never reaches the target, `high` does.
    low = 0
    while ev_at(high) < target_ev:
        if high >= limit:
            return Threshold(rank, direction, target_ev, None, cache[high],
                             limit, evaluations)
        low = high
        high = min(limit, high * 2)

    while high - low > 1:
        middle = (low + high) // 2
        if ev_at(middle) >= target_ev:
            high = middle
        else:
            low = middle
    return Threshold(rank, direction, target_ev, high, cache[high], limit,
                     evaluations)


def _threshold_job(payload):  # pragma: no cover - subprocess
    report, rank, direction, target_ev, limit = payload
    evaluator = Evaluator(report.rules, report.mode, workers=1)
    return find_threshold(rank, direction, evaluator, report, target_ev, limit)


def preferred_direction(report: SensitivityReport, rank: int) -> str:
    """Whether adding or removing this rank is the move that helps the player."""
    return "add" if report.effects[rank].addition_effect > 0 else "remove"


def find_thresholds(report: SensitivityReport, target_ev: float = 0.0,
                    workers: int = 1, ranks: Optional[Sequence[int]] = None,
                    evaluator: Optional[Evaluator] = None,
                    limit: Optional[int] = None) -> List[Threshold]:
    """Run :func:`find_threshold` for several ranks, one process per rank."""
    ranks = list(ranks) if ranks is not None else list(range(NUM_RANKS))
    jobs = [(report, rank, preferred_direction(report, rank), target_ev, limit)
            for rank in ranks]
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.Pool(processes=min(workers, len(jobs))) as pool:
            return pool.map(_threshold_job, jobs)
    evaluator = evaluator or Evaluator(report.rules, report.mode, workers=1)
    return [find_threshold(rank, direction, evaluator, report, target_ev, limit)
            for _, rank, direction, _, _ in jobs]


def sweep(rank: int, deltas: Iterable[int], evaluator: Evaluator,
          counts: Sequence[int]) -> List[Tuple[int, float]]:
    """Exact EV for a range of changes to the count of a single rank."""
    wanted = [delta for delta in deltas if counts[rank] + delta >= 0]
    batch = [apply_deltas(counts, {rank: delta}) for delta in wanted]
    return list(zip(wanted, evaluator.ev_many(batch)))
