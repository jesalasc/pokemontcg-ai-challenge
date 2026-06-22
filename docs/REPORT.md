# Strategy Track report (working draft)

> Target: a Kaggle **Writeup ≤ 2,000 words** + media gallery, due **Sep 14, 2026**.
> Judged on **deck-construction originality**, **algorithmic ingenuity**, **agent
> stability**, and **simulation performance**. Build this up continuously — don't
> write it at the end. Pull numbers/figures from `artifacts/eval/*.json` and
> `artifacts/replays/*.html`. Keep prose tight; every section earns its words.

## Title / subtitle
_TBD — name the deck concept + the technical hook._

## 1. Deck concept & originality  _(the biggest judged lever)_
- The archetype and the *insight* behind it (win condition, prize math, tempo).
- What's **original** vs the obvious meta list — card choices, tech, ratios, and
  *why* (use `tools/card_explorer.py` to justify from the legal pool).
- Matchup intent: what it beats, how it handles its bad matchups.

## 2. Algorithmic approach & ingenuity
- The layered design: rule-based → determinized MCTS → RL self-play (cite
  `docs/ARCHITECTURE.md`).
- Imperfect-information handling: determinization over the engine forward model;
  belief modelling.
- What's novel in *our* method (reward shaping tied to prize/KO domain signals;
  distillation from the scripted policy; etc.).

## 3. Agent architecture & stability  _(judged)_
- Crash-safe contract (`make_safe`) — never an illegal move/timeout (= instant loss).
- 10-min/match time budgeting in search.
- Test coverage incl. the simulator-rule divergences (`tests/test_sim_differences.py`).

## 4. Results
- Win-rate vs baselines + ladder rating curve (figures from `artifacts/`).
- Per-matchup table; ablations (does MCTS/RL beat the baseline? by how much?).

## 5. Reproducibility
- One-command build/eval (Docker), pinned engine version, seeds.

---
See `docs/DECISION_LOG.md` — append a dated line every time we make a real choice;
those lines are the raw material for sections 1–2.
