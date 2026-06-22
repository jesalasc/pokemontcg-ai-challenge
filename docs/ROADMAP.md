# Improvement roadmap

Our agent improves in stages, each measured against the previous by the local
gate (`make eval`). **Advance only when a stage clearly beats the one before** on
our chosen deck. Each stage leaves a deployable agent.

> Rule of the road: nothing advances on vibes. A stage "wins" when it beats the
> prior best by a margin whose Wilson CI clears 50% (`harness/evaluate.py`).

### Stage 0 — Deck (prerequisite, your call)
Pick the 60-card deck from the legal pool. Tools: `make cards` (explore),
`tools/deck_check.py` (validate). Everything below is tuned *for this deck*.
→ Log the concept + card-choice reasoning in `docs/DECISION_LOG.md` (judged).

### Stage 1 — Rule-based  ✅ (have it)
Explicit, deck-specific lines. `agents/dragapult.py` (93.75% vs random) and the
generic `agents/rule_based.py`. Encodes "what a good player does" as code.
**Files:** `agents/*.py`, `engine_codes.py`.
**Exit:** stable, crash-safe, beats random convincingly on our deck.

### Stage 2 — Heuristics (richer evaluation)
Replace hardcoded lines with a strong *position evaluation* the agent maximizes:
prize race, KO math, board tempo, energy efficiency, ex/mega prize risk. This is
where your card knowledge becomes numbers.
**Files:** `heuristics.py` (`board_value` weights), `cards.py` (damage/prizes).
**Measure:** heuristic-greedy vs Stage-1 lines, head-to-head.
**Exit:** evaluation-driven play ≥ scripted lines, and generalizes to off-script spots.

### Stage 3 — General strategy (search)
Stop relying on hand-authored decisions; let determinized MCTS *plan* using the
heuristic as its leaf evaluation + rollout policy. Generalizes to situations no
rule covers. Respect the 10-min/match budget.
**Files:** `forward_model.py`, `agents/mcts.py` (wired; tune iterations/determinizations/belief).
**Measure:** `make eval A=mcts B=dragapult`.
**Exit:** search beats the best Stage-1/2 agent.

### Stage 4 — Reward functions (RL setup)
Define the learning signal. Terminal ±1 is too sparse over 60+ turns — shape it
with domain signals (prize taken, KO dealt/avoided, energy efficiency). **This is
the human lever; convergence lives or dies here.**
**Files:** `training/ppo.py` (`shape_reward`), `features.py`.
**Measure:** does a distilled net (`training/distill.py`) match the teacher? does
shaped reward improve learning curves?
**Exit:** a net policy that matches or beats the search agent's decisions.

### Stage 5 — Self-play (RL refinement)
PPO self-play from a distilled warm start; play vs frozen past snapshots to avoid
cycling. Heavy compute → GPU/Linux box.
**Files:** `training/selfplay.py`, `training/ppo.py`, `networks.py`.
**Measure:** RL agent vs search agent in the harness; then the ladder.
**Exit:** RL is the strongest agent we have → it's what we submit.

---
Contingency (from the brief): if a later stage stalls, the prior stage stays
deployable. We never go to the ladder without a working agent.
