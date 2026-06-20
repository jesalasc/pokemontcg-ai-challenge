# Architecture & approach

The state of the art for Pokémon agents is **hybrid** (heuristics + search + RL),
not any pure extreme. We build in layers so a deployable agent is always on the
ladder, and each layer is a sparring partner / fallback for the next.

```
Layer 0  random        sanity + self-play sparring
Layer 1  rule_based  ✅ heuristics (develop-first)         — the FLOOR
Layer 2  mcts        🟡 determinized MCTS over forward model — robust mid-ladder
Layer 3  rl          🟡 self-play policy/value net           — the CEILING
```

Sequence them — don't parallelize. Each leaves a shippable agent; if RL doesn't
converge in time, Layer-1/2 still competes.

## The decisive constraint: imperfect information + stochasticity

Hidden hands/decks, coin flips, 20–100-turn horizons. Two consequences shape
everything:

1. **Determinization for search.** The engine hands us `search_begin_input` each
   turn and (in the official build) can *sample a concrete world* consistent with
   public info. Layer 2 runs perfect-info MCTS on many such samples and averages
   — this is the ReBeL/Libratus "reason over a belief distribution, not one
   state" idea, made cheap because the engine does the sampling.
2. **Variance in evaluation.** A single game says little. The harness plays many
   games with **alternating seats** (first-player advantage is real) and reports
   a Wilson CI + implied Elo gap. Nothing ships without clearing the gate.

## Why the layers, concretely

- **Layer 1 (rule_based).** Encodes the one thing measurement proved decisive:
  *attacking ends the turn, so develop fully, then attack.* That alone takes it
  from 0% → ~87% vs random. The domain expert extends it via `engine_codes.py`
  (name the type/area codes), `heuristics.py` (prize math, board value), and
  `MAIN_TYPE_PRIORITY` (sequence develop actions: attach → evolve → supporter…).

- **Layer 2 (mcts).** `forward_model.py` abstracts the engine's
  `search_begin/step/end`; `mcts.py` is determinized UCT with the rule_based
  policy as rollout. Budget by wall-clock (~600s pool/game) with iterative
  deepening. Wire `EngineForwardModel` once `tools/fetch_engine.sh` provides the
  search API; until then `mcts` transparently plays the baseline.

- **Layer 3 (rl).** An **option-scoring** net (`networks.py`): it scores each
  legal option given the state, so it handles the dynamic action set and makes
  distillation natural. Path:
  1. `distill.py` — behavioral-clone the net from rule_based / MCTS (the brief's
     "Scripted Policy Distillation"). Fast warm start.
  2. `ppo.py` — self-play PPO from that warm start. **Reward shaping
     (`shape_reward`) is the human lever** — terminal ±1 is too sparse over 60+
     turns; add prize-taken / KO / energy-efficiency signals (domain expert).
  3. Self-play vs frozen past snapshots to avoid strategy cycling.

## Critical transversal tool: the local eval gate

5 submissions/day → the iterate→measure loop is the scarcest resource.
`harness/evaluate.py` is the mandatory gate: win-rate + CI + Elo gap + avg game
length, JSON-logged. Rule: nothing reaches the ladder without clearing it
locally first.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Simulator ≠ paper rules | `tests/test_sim_differences.py` — pin every divergence found |
| Per-game time budget kills deep search | budget by wall-clock; iterative deepening; the ~600s pool is generous |
| RL doesn't converge | Layer-1/2 always deployable; distill warm-start before PPO |
| Burning submissions on bad versions | the eval gate is mandatory |
| Hidden-info variance | many games, alternating seats, average over determinizations |
| Crash on the ladder = instant loss | `make_safe` wrapper + crash-safety tests; `main.py` degrades to a legal agent even if imports fail |

## References
`references/` (ReBeL `2007.13544`, Libratus `science.aao1733`) and the upstream
competition docs. The PBS / Nash-equilibrium framing motivates the
determinization-for-search and self-play choices above.
