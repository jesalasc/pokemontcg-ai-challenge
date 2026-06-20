# Onboarding — zero to a running game

Goal: from a fresh clone to "I ran a battle and evaluated an agent" in one sitting.

## 0. Accept the rules (once, per person)
Accept rules and join the team on **both** competitions:
- Simulation: <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle>
- Strategy/Hackathon: <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy>

Then put your Kaggle API token at `~/.kaggle/kaggle.json` (Account → Create New
Token), `chmod 600` it. Needed for `pull-samples`, `fetch-engine`, `submit`.

## 1. Prereqs
- **Docker Desktop running** (the engine is Linux x86-64; we run it in a
  `linux/amd64` container — works on Apple Silicon via emulation).
- Python 3.11 on the host (for editing + non-engine tests). `pip install -r requirements.txt`.

## 2. Build the image and run a game
```bash
make build           # builds ptcg-dev (installs kaggle-environments==1.30.1 = the ladder engine)
make baseline-check  # rule_based vs random, 20 games, in the container
make test            # full suite incl. engine smoke tests
```
Expected: baseline wins clearly (~85%) and tests pass.

## 3. The daily loop
```bash
# change code (agents/heuristics/codes) ...
make eval A=rule_based B=random N=200      # the GATE: win-rate + CI + Elo gap
make replay A=rule_based B=random          # watch it: artifacts/replays/*.html
make submission A=rule_based               # build + smoke a .tar.gz
python tools/submit.py --file 'submissions/*.tar.gz' -m "msg"   # 5/day guarded
```
**Rule: nothing is submitted without passing the local gate first.** 5 subs/day.

## 4. Get the reference material (after creds)
```bash
make pull-samples    # the 3 sample kernels -> data/samples/
make fetch-engine    # official engine + card DB -> data/competition/
```
The sample kernels decode the engine's `type`/`area` integer codes — use them to
calibrate `src/ptcg/engine_codes.py` and to choose a real deck (`deck.csv`).

## 5. Where to dig in (by role)
- **Domain expert (highest leverage):** `src/ptcg/engine_codes.py`,
  `src/ptcg/heuristics.py`, `MAIN_TYPE_PRIORITY` in `agents/rule_based.py`,
  `deck.csv`, `tests/test_sim_differences.py`. Read replays, fix bad lines.
- **Search engineer:** `src/ptcg/forward_model.py` (wire the engine search API),
  `agents/mcts.py`, `src/ptcg/features.py`.
- **RL engineer:** `training/distill.py` then `training/ppo.py`
  (esp. `shape_reward`), `training/networks.py`. Use `requirements-rl.txt` + GPU.

## 6. How a turn works (mental model)
The engine calls `agent(obs)`; `obs["select"]["option"]` is the list of **legal**
moves; you return the indices you pick. First call (`select is None`) = return the
60-card deck. A crash/illegal/timeout = instant loss, so `make_safe` wraps every
agent. Full schema: `docs/ENGINE_API.md`.

## "I'm ready" checklist
- [ ] Rules accepted on both competitions; on the Kaggle team
- [ ] `~/.kaggle/kaggle.json` in place
- [ ] `make build` + `make test` green
- [ ] Read `docs/ENGINE_API.md` and `docs/ARCHITECTURE.md`
- [ ] Understand the 5/day limit and the eval gate
