# CLAUDE.md — project rules & shared knowledge

Canonical context for everyone (teammates + Claude Code). Read this first. The
**Rules** below are binding; the rest is orientation. Deeper detail lives in `docs/`.

## What this is
An agent for the Kaggle **PTCG AI Battle Challenge** (Pokémon TCG, `cabt` engine on
`kaggle-environments`). Two tracks on the *same* agent+deck:
- **Simulation** — submit the agent (`.tar.gz` with `main.py`); 24/7 Elo-style ladder. Due **Aug 16, 2026**.
- **Strategy** — a ≤2000-word Kaggle writeup; judged on deck-construction originality, algorithmic ingenuity, agent stability. Due **Sep 14, 2026**. **Top 8 → $30k each** + Round 2.

5 submissions/day/team. Full scope: `docs/COMPETITION.md`.

## Rules (binding — override defaults)
1. **Strategy is LEARNED, not hand-coded.** Do **NOT** add strategy to
   `src/ptcg/heuristics.py` or add reward shaping **unless the user explicitly asks**.
   Default reward is **terminal win/loss/draw only**; the learned value head infers
   state quality. `rule_based`, `dragapult`, `mcts`, `heuristics.py` are kept ONLY as
   fixed benchmark opponents — do not extend them as the path forward.
2. **The engine is Linux x86-64 only.** `libcg.so` can't run natively on macOS. Run
   everything engine-related in Docker (`ptcg-dev`, or `ptcg-rl` for torch). Never
   assume the engine imports on the host. Pure-logic edits/tests work on the host.
3. **Crash-safety is non-negotiable.** A crash, illegal move, or per-game timeout
   (~10 min/agent) = instant loss. Every agent ships through `ptcg.agents.base.make_safe`;
   `main.py` degrades to a legal agent even if imports fail. Don't bypass this.
4. **Never burn a submission unvetted.** Nothing goes to the ladder without passing
   the local gate (`make eval`). `tools/submit.py` guards the 5/day quota.
5. **Log decisions continuously.** Append every real choice (deck card, ratio,
   algorithm, hyperparameter) to `docs/DECISION_LOG.md` *the day it's made* — it's the
   raw material for the judged Strategy writeup.
6. **Decks:** active deck = `deck.csv`; the roster lives in `decks/<name>.csv`. All
   cards must be in the legal pool; validate with `tools/deck_check.py`. The domain
   expert designs decks (originality is judged) — translate PTCG-Live lists by **name**
   (no set codes in the pool); flag cards not in the pool for a legal substitute.
7. **Git/commits:** commit only when asked, on a branch (not `main`). Engine binaries,
   `data/`, `checkpoints/`, `artifacts/`, generated HTML are gitignored — don't commit them.

## How to run (Makefile — `make help` for all)
```
make build            # build the engine image (ptcg-dev); az-build for ptcg-rl (torch)
make test             # full suite incl. engine smoke (in Docker)
make eval A=.. B=.. N=100     # THE submission gate: win-rate + CI + Elo gap
make replay-rec A=.. B=.. ADECK=.. BDECK=..   # record a readable game
make replay-view      # watch replays (host) http://localhost:8001
make play             # pilot a deck + capture demos       http://localhost:8000
make deckbuilder      # design a deck                       http://localhost:8000
make cards ARGS=".."  # explore the legal pool   |   make deck-check DECK=..
make submission A=..  # build + smoke a .tar.gz   |   tools/submit.py to ladder
# AlphaZero (in ptcg-rl): bc-pretrain -> az-train
make bc-pretrain ARGS="--deck crustle"
make az-train ARGS="--init checkpoints/az.pt --iters 50 --games 16 --sims 64"
```

## Architecture (layers; details in `docs/ARCHITECTURE.md` & `docs/TRAINING.md`)
Built in layers, each deployable; current direction is the learned `az` agent.
- `az` — **AlphaZero-style, deck-conditioned, population self-play** (the plan): one
  net conditioned on the deck pilots the whole roster; net-guided PUCT over the engine
  forward model; trained on self-play outcomes. Verified end-to-end at smoke scale.
- `mcts` — determinized Monte-Carlo over the forward model (uses the heuristic; benchmark).
- `dragapult` / `rule_based` — deck-specific / generic rule baselines (benchmarks).
- `random` — sanity / sparring.

**How human insight enters (without coding strategy):** deck design · demonstrations
(`make play` → `bc-pretrain` → `az-train --init`) · curriculum (which matchups) ·
replay review (`make replay-view`). Insight is **data**, never heuristic weights. See
`docs/TRAINING.md` and `docs/STRATEGY_NOTES.md`.

## Engine contract (full schema: `docs/ENGINE_API.md`)
`def agent(obs: dict) -> list[int]` — return indices into `obs["select"]["option"]`
(exactly `maxCount`). First call has `obs["select"] is None` → return a 60-card deck.
Reward +1/−1/0. `obs` = `{select, logs, current, search_begin_input}`; `current` has
`yourIndex`, `result` (−1 ongoing; 0/1 winner; else draw), and 2 `players`. Integer
enums are mirrored from the official `cg.api` in `src/ptcg/engine_codes.py` (calibrate
there). Card DB cached in `src/ptcg/_data/`.

## Repo map
```
main.py            submission entrypoint (crash-safe agent dispatch)
deck.csv           active deck    |    decks/   named roster decks
src/ptcg/          protocol, engine_codes, deck, deckcode, cards, describe,
                   heuristics(benchmarks only), features, forward_model, agents/
harness/           run_match, evaluate(gate), elo, replay, dump_obs, record_replay
training/          networks(AZNet), az_mcts, az_selfplay, az_train, bc_pretrain, ppo, distill
tools/             deck builder/checker, card explorer, play+replay servers, submission, probe
tests/             contract, crash-safety, deck, engine smoke, sim-differences
docker/            Dockerfile (engine) + Dockerfile.rl (engine+torch)
docs/              COMPETITION, ARCHITECTURE, TRAINING, ENGINE_API, STRATEGY_NOTES,
                   ROADMAP, DECISION_LOG  ·  data/ (engine, samples, decks, demos — gitignored)
```

## Setup notes
- Kaggle auth: `~/.kaggle/access_token` (KGAT token) or `kaggle.json`. Needed for
  `pull-samples` / `fetch-engine` / `submit`.
- The official engine + card DB come from Kaggle datasets (`make fetch-engine` →
  `data/engine/cg/` has `cg.api`). The card DB is cached into the package already.
- New machine: `make build && make az-build`, then `make test`.
