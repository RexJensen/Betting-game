"""Command line front end for the blackjack engine.

    python -m engine.blackjack.cli edge
    python -m engine.blackjack.cli eor
    python -m engine.blackjack.cli thresholds
    python -m engine.blackjack.cli whatif --change A:+5 --change 6:-3
    python -m engine.blackjack.cli report
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Sequence

from .analyzer import Analyzer
from .cards import ACE, NUM_RANKS, RANKS, TEN, TWO_CARD_STATE, Shoe
from .rules import RuleSet
from .sensitivity import (COUNT_SYSTEMS, Evaluator, SensitivityReport,
                          apply_deltas, base_counts, evaluate_shoe,
                          find_threshold, make_strategy,
                          find_thresholds, sensitivity_report, sweep)
from .strategy import ACTION_LETTERS, BasicStrategy

RANK_INDEX = {name: index for index, name in enumerate(RANKS)}
RANK_INDEX["10"] = TEN
for _face in ("J", "Q", "K"):
    RANK_INDEX[_face] = TEN


def parse_rank(text: str) -> int:
    key = text.strip().upper()
    if key not in RANK_INDEX:
        raise argparse.ArgumentTypeError("unknown rank %r" % (text,))
    return RANK_INDEX[key]


def parse_change(text: str) -> Dict[int, int]:
    """Parse ``A:+5`` / ``6:-3`` / ``T:2`` into ``{rank: delta}``."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            "changes look like RANK:DELTA, for example A:+5 or 6:-3")
    rank_text, delta_text = text.split(":", 1)
    try:
        delta = int(delta_text)
    except ValueError:
        raise argparse.ArgumentTypeError("bad delta in %r" % (text,))
    return {parse_rank(rank_text): delta}


def merge_changes(changes: Sequence[Dict[int, int]]) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for change in changes or ():
        for rank, delta in change.items():
            merged[rank] = merged.get(rank, 0) + delta
    return {rank: delta for rank, delta in merged.items() if delta}


def rules_from_args(args: argparse.Namespace) -> RuleSet:
    return RuleSet(
        decks=args.decks,
        dealer_hits_soft_17=args.h17,
        blackjack_pays=args.bj_pays,
        double_rule=args.double,
        double_after_split=not args.no_das,
        max_split_hands=2 if args.no_resplit else 4,
        resplit_aces=args.rsa,
        hit_split_aces=args.hit_split_aces,
        late_surrender=args.surrender,
    )


def describe_change(deltas: Dict[int, int]) -> str:
    if not deltas:
        return "unchanged shoe"
    parts = []
    for rank in sorted(deltas, key=lambda r: (r != ACE, r)):
        delta = deltas[rank]
        parts.append("%+d %s%s" % (delta, RANKS[rank], "s" if abs(delta) != 1 else ""))
    return ", ".join(parts)


def _play_label(mode: str) -> str:
    return ("fixed basic strategy chart" if mode == "basic"
            else "composition dependent optimal")


def edge_line(label: str, ev: float) -> str:
    side = "player" if ev > 0 else "house"
    return "%-28s EV %+.6f   %s edge %.4f%%" % (label, ev, side, abs(100 * ev))


# -- commands ---------------------------------------------------------------
def cmd_edge(args: argparse.Namespace) -> int:
    rules = rules_from_args(args)
    counts = apply_deltas(base_counts(rules), merge_changes(args.change))
    result = evaluate_shoe(rules, counts, args.strategy, collect_actions=False)
    print("Rules: %s" % rules.describe())
    print("Shoe : %d cards (%s)" % (sum(counts), Shoe(counts).penetration_string()))
    print("Play : %s" % _play_label(args.strategy))
    print(edge_line("Round expected value", result.ev))
    insurance = Analyzer(rules, Shoe(counts), None).insurance_ev()
    print("%-28s EV %+.6f per unit insured" % ("Insurance bet", insurance))
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    rules = rules_from_args(args)
    print("Basic strategy for: %s" % rules.describe())
    print()
    print(BasicStrategy(rules).render())
    print()
    print("S stand  H hit  D double else hit  Ds double else stand  "
          "P split  Pd split if DAS  R surrender")
    return 0


def _print_eor_table(report: SensitivityReport) -> None:
    print("Effect of one card leaving / joining the shoe, in percentage points")
    print("of the player's expected value (positive = better for the player).")
    print()
    print("  card   in shoe   remove one   add one     per single deck")
    print("  " + "-" * 58)
    single_deck_scale = sum(report.counts) / 52.0
    for effect in report.effects:
        print("  %-4s   %5d    %+9.5f   %+9.5f   %+9.5f"
              % (effect.name, report.counts[effect.rank],
                 100 * effect.removal_effect, 100 * effect.addition_effect,
                 100 * effect.removal_effect * single_deck_scale))
    print()
    print("  The last column rescales the removal effect to a 52 card deck, the")
    print("  form Griffin's effect of removal tables are usually quoted in.")


def cmd_eor(args: argparse.Namespace) -> int:
    rules = rules_from_args(args)
    report = sensitivity_report(rules, args.strategy, args.workers)
    print("Rules: %s" % rules.describe())
    print("Play : %s" % _play_label(args.strategy))
    print(edge_line("Baseline", report.baseline_ev))
    print()
    _print_eor_table(report)
    print()
    print("  Consistency check: removing one of every card in proportion should")
    print("  leave the edge alone -- count weighted sum = %+0.6f%%"
          % (100 * sum(count * effect.removal_effect
                       for count, effect in zip(report.counts, report.effects))))
    print()
    print("Betting correlation of standard counting systems with these effects:")
    for name, tags in COUNT_SYSTEMS.items():
        print("  %-10s %+.4f" % (name, report.betting_correlation(tags)))
    return 0


def cmd_thresholds(args: argparse.Namespace,
                   evaluator: Optional[Evaluator] = None,
                   report: Optional[SensitivityReport] = None) -> int:
    rules = rules_from_args(args)
    evaluator = evaluator or Evaluator(rules, args.strategy, args.workers)
    report = report or sensitivity_report(rules, args.strategy, args.workers,
                                          evaluator=evaluator)
    target = args.target / 100.0
    if not getattr(args, "_embedded", False):
        print("Rules: %s" % rules.describe())
        print(edge_line("Baseline", report.baseline_ev))
        print()
    print("Cards of a single rank that must be added to, or removed from, the")
    print("shoe for the player's expected value to reach %+.4f%%." % (100 * target))
    print()
    print("  card   helpful move   cards needed   resulting EV   shoe")
    print("  " + "-" * 62)
    thresholds = find_thresholds(report, target, workers=args.workers,
                                 evaluator=evaluator)
    for threshold in thresholds:
        if threshold.cards is None:
            needed = "impossible"
            ev_text = ("%+.4f%%" % (100 * threshold.ev)
                       if threshold.ev is not None else "n/a")
            shoe_text = "limit %d" % threshold.limit
        else:
            needed = "%d" % threshold.cards
            ev_text = "%+.4f%%" % (100 * threshold.ev)
            total = sum(report.counts) + (threshold.cards
                                          if threshold.direction == "add"
                                          else -threshold.cards)
            shoe_text = "%d cards" % total
        print("  %-4s   %-12s   %-12s   %-12s   %s"
              % (threshold.name, threshold.direction, needed, ev_text, shoe_text))
    print()
    print("  'impossible' means the whole rank runs out (removal) or ten times the")
    print("  original count is reached (addition) before the target is met.")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    rules = rules_from_args(args)
    counts = base_counts(rules)
    evaluator = Evaluator(rules, args.strategy, args.workers)
    low, high = args.range
    deltas = list(range(low, high + 1, max(1, args.step)))
    if 0 not in deltas:
        deltas.append(0)
        deltas.sort()
    results = sweep(args.rank, deltas, evaluator, counts)
    baseline = dict(results)[0]
    print("Rules: %s" % rules.describe())
    print("Sweeping the number of %ss in the shoe." % RANKS[args.rank])
    print()
    print("  change   %ss in shoe   player EV    house edge   vs standard shoe"
          % RANKS[args.rank])
    print("  " + "-" * 66)
    for delta, ev in results:
        marker = "  <- standard shoe" if delta == 0 else ""
        print("  %+5d    %8d      %+.6f    %+8.4f%%   %+8.4f pts%s"
              % (delta, counts[args.rank] + delta, ev, -100 * ev,
                 100 * (ev - baseline), marker))
    return 0


def cmd_whatif(args: argparse.Namespace) -> int:
    rules = rules_from_args(args)
    deltas = merge_changes(args.change)
    if not deltas:
        print("nothing to change: pass --change RANK:DELTA", file=sys.stderr)
        return 2
    counts = base_counts(rules)
    modified = apply_deltas(counts, deltas)
    evaluator = Evaluator(rules, args.strategy, args.workers)
    baseline, changed = evaluator.ev_many([counts, modified])
    print("Rules: %s" % rules.describe())
    print("Change: %s" % describe_change(deltas))
    print()
    print(edge_line("Standard shoe", baseline))
    print(edge_line("Modified shoe", changed))
    print("%-28s %+.6f (%+.4f percentage points)"
          % ("Difference", changed - baseline, 100 * (changed - baseline)))
    print()
    print("Shoe: %d cards (%s)" % (sum(modified), Shoe(modified).penetration_string()))
    if args.linear:
        report = sensitivity_report(rules, args.strategy, args.workers,
                                    evaluator=evaluator)
        estimate = report.linear_ev(deltas)
        print("%-28s %+.6f (first order from single card effects, error %+.6f)"
              % ("Linear estimate", estimate, estimate - changed))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Compare the fixed chart against composition dependent optimal play."""
    rules = rules_from_args(args)
    counts = base_counts(rules)
    result = evaluate_shoe(rules, counts, "optimal", collect_actions=True)
    strategy = BasicStrategy(rules)
    basic = evaluate_shoe(rules, counts, "basic")
    print("Rules: %s" % rules.describe())
    print(edge_line("Composition dependent optimal", result.ev))
    print(edge_line("Fixed basic strategy chart", basic.ev))
    print("%-28s %.4f%% of a bet per round"
          % ("Cost of the fixed chart", 100 * (result.ev - basic.ev)))
    print()
    print("Starting hands where the chart differs from the exact best action:")
    print()
    print("  hand         chart   best    cost of chart   frequency")
    print("  " + "-" * 56)
    rows = []
    for case in result.cases:
        if len(case.action_evs) < 2:
            continue
        first, second = case.cards
        pair_rank = first if first == second else None
        state = TWO_CARD_STATE[first][second]
        chart_action = strategy.decide(
            state, case.upcard, pair_rank,
            rules.may_double(state >> 1, bool(state & 1), 2, False),
            pair_rank is not None, rules.late_surrender)
        if chart_action == case.action:
            continue
        cost = case.action_evs[case.action] - case.action_evs[chart_action]
        rows.append((cost * case.probability, case.label,
                     ACTION_LETTERS[chart_action], ACTION_LETTERS[case.action],
                     cost, case.probability))
    for _, label, chart_letter, best_letter, cost, probability in sorted(rows, reverse=True):
        print("  %-12s %-7s %-7s %-15.5f %.5f"
              % (label, chart_letter, best_letter, cost, probability))
    if not rows:
        print("  (none)")
    print()
    print("  These are composition dependent deviations: the chart is total")
    print("  dependent, so a few exact two card hands prefer another action.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rules = rules_from_args(args)
    evaluator = Evaluator(rules, args.strategy, args.workers)
    report = sensitivity_report(rules, args.strategy, args.workers,
                               evaluator=evaluator)
    print("=" * 72)
    print("BLACKJACK HOUSE EDGE SENSITIVITY")
    print("=" * 72)
    print("Rules: %s" % rules.describe())
    print("Play : %s" % _play_label(args.strategy))
    print(edge_line("Baseline", report.baseline_ev))
    print()
    _print_eor_table(report)
    print()
    print("Betting correlation with standard count systems:")
    for name, tags in COUNT_SYSTEMS.items():
        print("  %-10s %+.4f" % (name, report.betting_correlation(tags)))
    print()
    print("-" * 72)
    print("HOW MANY CARDS DOES IT TAKE TO FLIP THE GAME")
    print("-" * 72)
    args.target = 0.0
    args._embedded = True
    cmd_thresholds(args, evaluator=evaluator, report=report)
    return 0


# -- argument plumbing ------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m engine.blackjack.cli",
        description="Exact blackjack engine and house edge sensitivity analysis.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--decks", type=int, default=6)
    common.add_argument("--h17", action="store_true",
                        help="dealer hits soft 17 (default: stands)")
    common.add_argument("--bj-pays", type=float, default=1.5,
                        help="blackjack payout, 1.5 for 3:2 or 1.2 for 6:5")
    common.add_argument("--double", choices=("any2", "9-11", "10-11", "none"),
                        default="any2")
    common.add_argument("--no-das", action="store_true",
                        help="forbid doubling after a split")
    common.add_argument("--no-resplit", action="store_true",
                        help="split to two hands only")
    common.add_argument("--rsa", action="store_true", help="allow resplitting aces")
    common.add_argument("--hit-split-aces", action="store_true")
    common.add_argument("--surrender", action="store_true",
                        help="allow late surrender")
    common.add_argument("--strategy", choices=("basic", "optimal"), default="basic",
                        help="fixed basic strategy chart, or composition "
                             "dependent optimal play")
    common.add_argument("--workers", type=int, default=1,
                        help="parallel processes for multi shoe analyses")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("edge", parents=[common],
                              help="house edge for one shoe composition")
    p.add_argument("--change", type=parse_change, action="append",
                   metavar="RANK:DELTA")
    p.set_defaults(func=cmd_edge)

    p = subparsers.add_parser("chart", parents=[common],
                              help="print the basic strategy chart")
    p.set_defaults(func=cmd_chart)

    p = subparsers.add_parser("eor", parents=[common],
                              help="effect of removing/adding one of each card")
    p.set_defaults(func=cmd_eor)

    p = subparsers.add_parser("thresholds", parents=[common],
                              help="how many cards of a rank flip the edge")
    p.add_argument("--target", type=float, default=0.0,
                   help="target player EV in percent (default 0, break even)")
    p.set_defaults(func=cmd_thresholds)

    p = subparsers.add_parser("whatif", parents=[common],
                              help="edge after an arbitrary composition change")
    p.add_argument("--change", type=parse_change, action="append",
                   metavar="RANK:DELTA", required=True)
    p.add_argument("--linear", action="store_true",
                   help="also show the first order estimate")
    p.set_defaults(func=cmd_whatif)

    p = subparsers.add_parser("sweep", parents=[common],
                              help="EV across a range of counts of one rank")
    p.add_argument("--rank", type=parse_rank, required=True)
    p.add_argument("--range", type=int, nargs=2, default=(-8, 8),
                   metavar=("LOW", "HIGH"))
    p.add_argument("--step", type=int, default=1)
    p.set_defaults(func=cmd_sweep)

    p = subparsers.add_parser("validate", parents=[common],
                              help="chart versus exact optimal play")
    p.set_defaults(func=cmd_validate)

    p = subparsers.add_parser("report", parents=[common],
                              help="the full sensitivity write up")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
