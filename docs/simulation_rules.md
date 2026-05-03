# Simulation Rules (Launch Spec)

## 1) Bet types at launch

- **Launch scope (v1): Moneyline only** for pregame wagers.
- **Deferred scope (v1.1+):** point spread and totals (over/under) support will be added later.
- No live/in-play betting in v1.

## 2) Data cutoff timestamp per game

For each game, a strict **pregame cutoff timestamp** is enforced:

- `data_cutoff_ts = scheduled_first_pitch_local` in the home ballpark’s local timezone.
- Only market, team, and model features timestamped **at or before** `data_cutoff_ts` are visible to the strategy.
- Any updates after first pitch (line moves, injuries, weather shifts, live stats) are hidden until settlement.

Implementation note:
- Store both `scheduled_first_pitch_local` and `scheduled_first_pitch_utc` to avoid timezone ambiguity.

## 3) Bankroll rules

- **Starting bankroll:** `$10,000.00`.
- **Min bet size:** `$10.00`.
- **Max bet size:** `5.0%` of current bankroll per wager.
- **Sizing mode (v1 default):** flat bet of `$100` per qualified play, clamped to min/max limits.
- **Optional mode (v1 config flag):** unit sizing, where `1 unit = 1.0%` of current bankroll and allowed range is `0.5u` to `5.0u`.

Sizing precedence:
1. Compute requested stake via selected sizing mode.
2. Apply min/max clamps.
3. Round to nearest cent.

## 4) Settlement rules

- **Grading outcomes:** `win`, `loss`, `push`, `void`.
- **Push handling:** return full stake; P/L = `$0.00`.
- **Void handling (canceled game/listed-pitcher mismatch/book void):** return full stake; P/L = `$0.00`.
- **Odds normalization:** all odds normalized to **American odds** internally (`+150`, `-110`, etc.) with derived decimal odds for payout math.
- **Commission/Juice assumption:** odds are assumed to already include sportsbook vig; no extra commission is applied beyond quoted odds.

Payout math (American odds):
- If odds > 0: `profit = stake * (odds / 100)`.
- If odds < 0: `profit = stake * (100 / abs(odds))`.
- Total return on win = `stake + profit`; on loss = `0`.

## 5) Backtest line selection assumption

- **Default line used:** **closing pregame line** captured at `data_cutoff_ts` (first pitch).
- **Configurable alternatives:** opening line or best available pregame line (across tracked books), but these are disabled by default in v1.
- Backtest reports must log which line policy was used (`opening`, `closing`, `best_pregame`) for reproducibility.

---

## Concrete example (single game row)

### Pregame-visible fields (at/ before cutoff)

| field | example value | visible pregame? |
|---|---:|:---:|
| game_id | `MLB_2026-04-12_NYY_BOS` | ✅ |
| sport | `MLB` | ✅ |
| home_team | `BOS` | ✅ |
| away_team | `NYY` | ✅ |
| scheduled_first_pitch_local | `2026-04-12T19:10:00-04:00` | ✅ |
| scheduled_first_pitch_utc | `2026-04-12T23:10:00Z` | ✅ |
| data_cutoff_ts | `2026-04-12T19:10:00-04:00` | ✅ |
| moneyline_home_american | `-118` | ✅ |
| moneyline_away_american | `+108` | ✅ |
| line_source_policy | `closing` | ✅ |
| model_win_prob_home | `0.547` | ✅ |
| model_win_prob_away | `0.453` | ✅ |
| recommended_bet_side | `BOS` | ✅ |
| recommended_stake_usd | `100.00` | ✅ |

### Hidden until settlement

| field | example value | visible pregame? |
|---|---:|:---:|
| game_status_final | `FINAL` | ❌ |
| final_score_home | `4` | ❌ |
| final_score_away | `2` | ❌ |
| bet_outcome | `win` | ❌ |
| settled_odds_american | `-118` | ❌ |
| payout_profit_usd | `84.75` | ❌ |
| bankroll_post_bet_usd | `10084.75` | ❌ |
| settlement_ts_utc | `2026-04-13T02:31:00Z` | ❌ |
