# PTCG AI Battle Challenge — agent

Agent + training/eval scaffold for the Kaggle [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(cabt engine on `kaggle-environments`). Built layer-by-layer so there's always a
deployable agent on the ladder: **rule-based → MCTS → RL self-play**.

## Status (verified end-to-end)

| Layer | Agent | State | Result |
|---|---|---|---|
| 3 | `az` | ✅ pipeline working | **AlphaZero-style, deck-conditioned, learned from self-play** (no hand-coded strategy). Net-guided PUCT over the forward model; pipeline verified end-to-end. Train it, then it's our agent. See `docs/TRAINING.md`. |
| 1 | `dragapult` | ✅ working | deck-specific benchmark (vendored sample) — **93.75% vs random** on the Dragapult deck |
| 1 | `rule_based` | ✅ working | generic benchmark — 90% on a simple deck, ~25% on Dragapult (kept as a fixed opponent, not extended) |
| 2 | `mcts` | ✅ working | determinized Monte-Carlo over the forward model (uses the heuristic eval; a benchmark) |
| 0 | `random` | ✅ working | sparring / sanity baseline |

> **Direction:** strategy is **learned**, not hand-coded. The `az` agent is a single
> deck-conditioned network trained by population self-play across the deck roster in
> `decks/`; the rule-based/heuristic agents are kept only as fixed benchmarks. Train
> with `make az-train` (in the `ptcg-rl` image). Details: `docs/TRAINING.md`.

The engine, harness, agents, tests, card DB, forward model, and submission packaging all run and pass (verified in Docker).

> **Deck note (measured):** the *generic* `rule_based` is deck-agnostic — strong on a
> simple deck (90%) but weak piloting the complex Dragapult ex evolution deck (25%,
> long stalled games). Strong play on a real meta deck needs **deck-specific logic**
> (the `data/samples/` kernels are exactly that, one agent per archetype) and/or
> search/RL. `deck.csv` is the Dragapult ex meta list; next step is to port the
> matching sample agent or let MCTS/RL pilot it.

## ⚠️ Platform: the engine is Linux x86-64 only

`libcg.so` is a Linux binary — it **cannot run natively on macOS/Apple Silicon**.
Everything that touches the engine runs in the bundled `linux/amd64` Docker image.
Pure-logic editing and the non-engine tests work fine directly on your Mac.

## Quickstart

```bash
# 1. Build the engine image (once; Docker Desktop must be running)
make build

# 2. Sanity: baseline vs random (runs a real game loop in the container)
make baseline-check

# 3. Full test suite (engine tests included)
make test

# 4. Evaluate any matchup — THIS IS THE SUBMISSION GATE
make eval A=rule_based B=random N=100

# 5. Watch a game (interactive HTML replay -> artifacts/replays/)
make replay A=rule_based B=random

# 6. Build + smoke-test a submission (.tar.gz with main.py at root)
make submission A=rule_based

# 7. Submit (guards the 5/day quota; needs ~/.kaggle/kaggle.json)
python tools/submit.py --file 'submissions/*.tar.gz' -m "baseline v1"
```

`make help` lists everything.

## Repo layout

```
main.py              # submission entrypoint — crash-proof agent(obs) (root of the .tar.gz)
deck.csv             # the 60-card deck (swap freely; engine-default for now)
src/ptcg/
  protocol.py        # verified obs/select contract + legal fallback
  engine_codes.py    # cabt integer enums — ONE place to calibrate codes
  deck.py  heuristics.py  features.py  forward_model.py
  agents/  base.py random_agent.py rule_based.py mcts.py rl_agent.py
harness/             # run_match, evaluate (gate), replay, dump_obs, elo
training/            # selfplay, distill (BC), ppo, networks   (Layer 3, PyTorch)
tools/               # build_submission, submit, pull_samples.sh, fetch_engine.sh
tests/               # contract, crash-safety, deck, engine smoke, sim-differences
docker/              # linux/amd64 image + compose
docs/                # ENGINE_API.md (verified schema), ARCHITECTURE.md
```

## Who edits what

- **Domain expert:** `engine_codes.py` (type/area codes), `heuristics.py` (board
  value, prize math), `MAIN_TYPE_PRIORITY` in `rule_based.py`, `deck.csv`,
  `tests/test_sim_differences.py`.
- **Search engineer:** `forward_model.py`, `mcts.py`, `features.py`.
- **RL engineer:** `training/` (esp. `shape_reward` in `ppo.py`), `networks.py`.

See `ONBOARDING.md` to go from zero to a running game, and `docs/ENGINE_API.md`
for the verified engine contract.
