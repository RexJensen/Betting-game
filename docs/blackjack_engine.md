# Blackjack engine and house edge sensitivity

`engine/blackjack` is an exact combinatorial analyser for standard rules
blackjack. It exists to answer one family of questions:

> If a single card were added to or removed from the shoe, how much would the
> house edge move? How many aces have to be added before the game is +EV? How
> many sixes removed? What do extra nines actually do?

Every number below is computed, not simulated. The engine enumerates every
deal and every decision against the exact remaining shoe, so running it twice
gives the same answer to the last digit.

## The game it models

Defaults describe the most common US shoe game:

| rule | default | configurable |
|---|---|---|
| decks | 6 | `--decks` |
| dealer on soft 17 | stands | `--h17` |
| blackjack pays | 3:2 | `--bj-pays` |
| doubling | any two cards | `--double any2 \| 9-11 \| 10-11 \| none` |
| double after split | yes | `--no-das` |
| splitting | to four hands | `--no-resplit`, `--rsa` |
| split aces | one card each | `--hit-split-aces` |
| surrender | none | `--surrender` (late) |
| dealer hole card | peeked | fixed |

The player either follows a fixed basic strategy chart (the default, and the
right choice for measuring composition effects) or plays composition dependent
optimal — recomputing the best action against the exact shoe at every decision
(`--strategy optimal`).

## Running it

```
python -m engine.blackjack.cli edge                      # house edge for the rules
python -m engine.blackjack.cli chart                     # the basic strategy grid
python -m engine.blackjack.cli eor        --workers 4    # effect of every card
python -m engine.blackjack.cli thresholds --workers 4    # how many cards flip the game
python -m engine.blackjack.cli sweep --rank 9 --range -12 20 --step 4 --workers 4
python -m engine.blackjack.cli whatif --change A:+5 --change 6:-3 --linear
python -m engine.blackjack.cli validate                  # chart versus exact optimal
python -m engine.blackjack.cli report     --workers 4    # everything at once
```

A single exact evaluation of the whole game takes about four seconds for six
decks; `--workers` spreads the independent evaluations of a sweep across
processes.

## The answer

Six decks, S17, 3:2, double any two, DAS, split to four, no surrender. Basic
strategy is held fixed throughout: the edge moves because the cards moved, not
because the player started playing differently.

**Baseline house edge: 0.4099%** (player EV −0.004099 per unit wagered).

### What one card is worth

Percentage points of player EV, per card added to or removed from the 312 card
shoe:

| card | remove one | add one | rescaled to one deck |
|---|---:|---:|---:|
| A | −0.09545 | +0.09419 | −0.573 |
| 2 | +0.06314 | −0.06317 | +0.379 |
| 3 | +0.07174 | −0.07179 | +0.430 |
| 4 | +0.09446 | −0.09500 | +0.567 |
| 5 | **+0.11887** | −0.11914 | +0.713 |
| 6 | +0.07478 | −0.07466 | +0.449 |
| 7 | +0.04349 | −0.04337 | +0.261 |
| 8 | −0.00538 | +0.00562 | −0.032 |
| 9 | −0.03265 | +0.03244 | −0.196 |
| T | −0.08323 | +0.08148 | −0.499 |

Reading it: pulling one five out of a six deck shoe is worth **+0.119
percentage points** to the player — on its own, more than a quarter of the
house's whole 0.41% edge. Pulling one eight out is worth −0.005, which is
nothing. The last column rescales to a 52 card deck, the form Griffin's effect
of removal tables are quoted in. Running the engine on an actual single deck
reproduces that table directly — every rank within 0.027 of the published
value, with the same ordering:

| card | A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| engine, one deck | −0.585 | +0.370 | +0.413 | +0.529 | +0.669 | +0.435 | +0.269 | +0.005 | −0.161 | −0.485 |
| Griffin | −0.61 | +0.38 | +0.44 | +0.55 | +0.69 | +0.46 | +0.28 | +0.00 | −0.18 | −0.51 |

The shape is the familiar one: low cards (2–7) help the player when they
leave, high cards (T, A) hurt, the eight is the pivot, and the nine leans
slightly high. The ace is the second most valuable card by removal effect but
the most valuable to *have* in the shoe, because it is the one card that pays
3:2.

Correlating those effects with published counting systems gives exactly the
betting correlations those systems are known for, which is a good independent
check on the whole vector:

| system | betting correlation |
|---|---:|
| Hi-Lo | +0.967 |
| Zen Count | +0.962 |
| Omega II | +0.922 |
| Hi-Opt II | +0.913 |
| Hi-Opt I | +0.879 |

### How many cards it takes to flip the game

Smallest number of cards of one rank that has to be added or removed before
the player's EV reaches break even, everything else untouched:

| card | move | cards | resulting EV | shoe |
|---|---|---:|---:|---|
| A | add | **5** | +0.0485% | 317 |
| 2 | remove | 7 | +0.0313% | 305 |
| 3 | remove | 6 | +0.0203% | 306 |
| 4 | remove | 5 | +0.0569% | 307 |
| 5 | remove | **4** | +0.0639% | 308 |
| 6 | remove | **6** | +0.0410% | 306 |
| 7 | remove | 10 | +0.0318% | 302 |
| 8 | add | 40 | +0.0134% | 352 |
| 9 | add | **14** | +0.0291% | 326 |
| T | add | 6 | +0.0540% | 318 |

So: **five extra aces**, or **four fives removed**, or **six sixes removed**,
or **fourteen extra nines**. The eight needs forty — it is very nearly a dead
card, and moving eights is close to the worst way to try to beat the game.

### The curves

The effects are close to linear over the ranges that matter, with mild
diminishing returns as a rank gets far from its natural density.

Aces added (per ace: +0.094 points at the start, +0.081 by the twelfth):

| aces in shoe | 18 | 20 | 22 | **24** | 26 | 28 | 30 | 32 | 34 | 36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| player EV % | −1.001 | −0.799 | −0.602 | **−0.410** | −0.223 | −0.041 | +0.136 | +0.308 | +0.476 | +0.638 |

Sixes removed:

| sixes in shoe | 12 | 15 | 18 | 21 | **24** | 27 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| player EV % | +0.499 | +0.269 | +0.041 | −0.185 | **−0.410** | −0.634 | −0.856 |

Nines added — the point of the question. Nines are a weak high card, worth
about a third of an ace and two fifths of a ten:

| nines in shoe | 12 | 16 | 20 | **24** | 28 | 32 | 36 | 40 | 44 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| player EV % | −0.819 | −0.678 | −0.542 | **−0.410** | −0.281 | −0.156 | −0.032 | +0.090 | +0.210 |

Tens added:

| tens in shoe | 84 | 88 | 92 | **96** | 100 | 104 | 108 |
|---|---:|---:|---:|---:|---:|---:|---:|
| player EV % | −1.537 | −1.128 | −0.754 | **−0.410** | −0.094 | +0.196 | +0.463 |

### If the player also adapts

Everything above holds basic strategy fixed. A player who instead re-optimises
every decision against the actual composition (`--strategy optimal`) starts
from a 0.4068% house edge and needs slightly less help — but not much less,
because most of the gain from a distorted shoe is in the cards themselves, not
in the play:

| card | fixed basic strategy | re-optimising |
|---|---|---|
| aces added | 5 | 5 |
| sixes removed | 6 | 6 |
| nines added | 14 | 12 |
| tens added | 6 | 5 |

Adapting is worth more as the shoe gets stranger, which is why the gap opens up
for the ranks that need the biggest distortion.

### Rule variants, for scale

A card or two is worth about as much as a rule. Same shoe, basic strategy:

| rules | house edge |
|---|---:|
| 6 decks, S17 (baseline) | 0.4099% |
| 6 decks, H17 | 0.6226% |
| 6 decks, blackjack pays 6:5 | 1.7696% |
| 6 decks, no double after split | 0.5494% |
| 6 decks, double on 10–11 only | 0.6017% |
| 6 decks, no resplitting | 0.4580% |
| 6 decks, late surrender | 0.3374% |
| 6 decks, hit split aces | 0.2226% |
| 1 deck, S17 | −0.1237% |
| 2 decks, S17 | 0.1936% |
| 8 decks, S17 | 0.4372% |

Moving from 3:2 to 6:5 costs 1.36 points — the same as taking about fourteen
aces out of the shoe.

## Method

For every one of the 550 (player two cards × dealer upcard) starting positions,
weighted by its exact dealing probability without replacement:

* the dealer's final total distribution is computed by exhaustive recursion
  over every card the dealer could draw, with the shoe depleting as it goes,
  conditioned on the dealer having peeked and not having a natural;
* the player's stand, hit, double, split and surrender values are computed the
  same way, each against the shoe as it actually stands at that point;
* the action taken is either the chart's or the exact best one, and the round
  value is assembled from those.

Results are memoised on a packed integer key of the exact rank counts, which is
what makes the recursion finish: a six deck evaluation touches roughly half a
million distinct dealer states and reuses them across hands.

Two approximations remain, both standard for combinatorial analysers:

1. **Post-split hands are valued independently.** Each hand after a split sees
   the shoe minus the split cards and the upcard, not minus its sibling's
   draws, and the resplit budget is per hand (one resplit each, so at most four
   hands) rather than tracked globally.
2. **Peek conditioning is applied to the shoe as it stands when the player
   acts**, rather than to the shoe as it stood at the moment of the peek.

The second was measured directly against an exact hole card first computation
and is worth 0.00003 of a unit — irrelevant. The first is the one that matters,
and it is why the single deck figure above (−0.124%, i.e. a small player edge)
is more optimistic than published single deck analyses, which cluster nearer
break even. Splitting is worth 0.51 points in a single deck and 0.56 in six,
and the independence assumption inflates part of that; with six decks the
engine's house edge lands within about a hundredth of a point of published
combinatorial analyses, which bounds the error there.

For the sensitivity numbers this matters less than it looks. Removing the
split option entirely changes the effects of removal by at most 0.015 points
(for the five; less for everything else), so splitting contributes only a small
slice of each effect, and the error *inside* that slice is a fraction of it
again. Every comparison here is a difference between two runs that share the
same approximation.

## Validation

| quantity | engine | published |
|---|---:|---:|
| 6 deck S17 DAS house edge, basic strategy | 0.4099% | ≈0.40% |
| 8 deck S17 DAS house edge | 0.4372% | ≈0.43% |
| 2 deck S17 DAS house edge | 0.1936% | ≈0.19% |
| cost of H17 | 0.213 pts | ≈0.21 pts |
| cost of 6:5 blackjacks | 1.360 pts | ≈1.39 pts |
| cost of no DAS | 0.140 pts | ≈0.14 pts |
| value of late surrender | 0.073 pts | ≈0.08 pts |
| value of hitting split aces | 0.187 pts | ≈0.19 pts |
| insurance, full shoe | −7.40% | −7.40% |
| dealer bust rate, six up (no natural) | 42.28% | ≈42.3% |
| dealer bust rate, ten up (no natural) | 23.02% | ≈23.0% |
| Griffin's single deck effects of removal | max deviation 0.027 | published table |
| Hi-Lo betting correlation | 0.967 | 0.97 |
| gain from composition dependent play | 0.0031 pts | ≈0.003 pts |

`tests/test_blackjack.py` additionally checks the dealer distribution, hand
valuation, stand, hit and double values against a deliberately naive
reimplementation — plain card lists, no memoisation, aces counted as one and
promoted — which agrees to 1e-12.

`python -m engine.blackjack.cli validate` compares the shipped chart against
exact optimal play hand by hand. It finds exactly one starting hand where the
chart is not the best action for that specific composition: T,2 versus 4, where
hitting beats standing by 0.00075 of a unit. That is the well known composition
dependent exception, and its presence (and the absence of anything else) is a
strong check that the chart is right.

## Caveats worth stating

* This is the EV of **one round dealt off the top** of a shoe with the given
  composition. It is not a bet ramp, a counting simulation, or a bankroll
  model, and it says nothing about variance.
* "Adding a card" means physically adding it: the shoe grows to 313 cards. It
  is not the same as observing a card and adjusting a count, though the two
  give the same first order answer.
* The player is assumed not to take insurance. Insurance EV for the current
  shoe is reported separately by the `edge` command; it turns positive once
  the shoe is rich enough in tens.
* Basic strategy is held fixed at the standard chart even for wildly distorted
  shoes. A player who re-optimises does better, so the card counts above are
  the conservative answer to "how many cards make this beatable".
