# Competition scope (PTCG AI Battle Challenge)

Synthesized from the Kaggle pages + organizer coverage (June 2026). Confirm dates
on Kaggle before committing the calendar.

## Two tracks, same agent — then a second round

It's **one body of work** (an agent + a deck) entered into two tracks in Round 1:

| | Simulation | Strategy (where the prize is) |
|---|---|---|
| Deliverable | the agent (.tar.gz, `main.py`) | a **Kaggle Writeup ≤2000 words** + media gallery (+ optional code/notebooks) |
| Judged on | 24/7 ladder, Gaussian/Glicko-style live rating | **simulation results + report**: originality of **deck construction**, **algorithmic ingenuity**, **agent stability** |
| Deadline | **Aug 16, 2026** (sim submission) | **Sep 14, 2026** (report) |
| Reward | leaderboard standing | **Top 8 teams → $30,000 each**, advance to Round 2 |

**Round 2** (after 2026, streamed on YouTube): $50k winner, $30k second, $3k GCP
credits to all participants. So the path to the money is: rank well in Sim →
write a strong Strategy report → top 8 → Round 2.

## Game / deck format (verified from the engine)

- You build a **single 60-card deck from the organizer's legal card pool**
  (only listed cards are legal). The pool we cached (`src/ptcg/_data/cards.json`)
  is **1267 cards**: 1056 Pokémon (151 ex/megaEx), 77 Items, 61 Supporters,
  27 Tools, 26 Stadiums, 12 Special + 8 Basic Energy.
- **The deck is part of the agent**: it's returned on the first call
  (`obs["select"] is None`). You bring **one** deck. There is **no counter-pick**
  — at deck selection you have zero opponent info. No multi-deck system needed.
- **10 minutes per player per match**; running out the clock = immediate loss
  (matches the engine's ~600s overage pool). Budget search time accordingly.
- Rules are **official PTCG, "uniquely adjusted" for the tournament** — the
  divergences are decisive; pin them in `tests/test_sim_differences.py`.

## What this means for how we work (do it right from day 1)

1. **Deck construction is itself judged** (originality). Don't just copy a sample
   list — the domain expert should design/justify a deck from the legal pool. Our
   job: pilot *that* deck excellently.
2. **Agent stability is judged** → the crash-safe wrapper isn't just survival,
   it's score. Keep it.
3. **Keep a decision log from the start** — deck concept, why each algorithmic
   choice — it becomes the 2000-word report. The eval JSON + replays feed it.
4. **Respect the 10-min budget** in search/RL (MCTS already budgets by wall-clock).
5. One agent, one deck, piloted by deck-specific logic + search/RL.
