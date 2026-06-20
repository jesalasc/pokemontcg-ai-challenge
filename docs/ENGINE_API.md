# cabt Engine API — verified reference

Ground truth extracted from `kaggle-environments==1.30.1`
(`envs/cabt/cabt.py`, `cg/game.py`, `cg/sim.py`) **and a live game** (see
`artifacts/schema/`). This is the contract our code targets. Upstream docs:
<https://matsuoinstitute.github.io/cabt/>.

## Platform constraint (important)

The engine is a native library `libcg.so` (**Linux x86-64 ELF**; there's a
Windows `cg.dll`, **no macOS build**). It is `dlopen`-ed at import and
`GameInitialize()` runs immediately, so simply importing the env **crashes on
Apple Silicon**. → All engine execution runs in the `linux/amd64` Docker image
(see `docker/`) or on a Linux box. Editing/tests of pure logic work on macOS.

## The agent contract

```python
def agent(obs: dict) -> list[int]:
    ...
```

Called every step. Returns **indices into `obs["select"]["option"]`**.

- **Deck selection** — the FIRST call has `obs["select"] is None`. Return a list
  of **60 card IDs** (the deck). This is how the deck is chosen — it ships in the
  agent. We do this in `ptcg.agents.base.make_safe` from `deck.csv`.
- **Play** — return exactly `obs["select"]["maxCount"]` distinct option indices.
- **Crash / illegal move / timeout ⇒ instant loss (reward −1).** Never raise.

## Observation schema

```
obs = {
  "remainingOverageTime": float,   # your thinking-time budget left (pool, ~600s/game)
  "step": int,
  "select": {                      # None during deck selection
    "type": int,
    "context": int,               # WHICH decision (see contexts below)
    "minCount": int,
    "maxCount": int,              # how many options to pick (usually 1)
    "remainDamageCounter": int,   # for damage-counter placement
    "remainEnergyCost": int,      # for paying energy
    "option": [ {...}, ... ],     # the legal choices (see option shapes)
    "deck": None, "contextCard": None, "effect": None   # context-dependent
  },
  "logs": [...],                  # event history
  "current": {                    # None during deck selection
    "turn": int, "turnActionCount": int,
    "yourIndex": int,             # which player (0/1) you are
    "firstPlayer": int,
    "supporterPlayed": bool, "stadiumPlayed": bool,
    "energyAttached": bool, "retreated": bool,   # once-per-turn flags
    "result": int,                # -1 ongoing; 0 -> P0 wins; 1 -> P1 wins; else draw
    "stadium": card|None, "looking": ...,
    "players": [ player0, player1 ]
  },
  "search_begin_input": str       # opaque blob -> seeds the engine forward model
}

player = {
  "active":  [card] | [],         # active Pokémon (list; usually 0 or 1)
  "bench":   [card, ...], "benchMax": int,
  "hand":    [card, ...],         # YOUR hand only; opponent's is hidden
  "handCount": int,               # both players expose a count
  "discard": [card, ...], "prize": [card, ...], "deckCount": int,
  "poisoned": bool, "burned": bool, "asleep": bool,
  "paralyzed": bool, "confused": bool
}

card = {
  "id": int,                      # card type id (maps to name/HP/attacks via card DB)
  "serial": int,                  # unique instance id
  "playerIndex": int,
  # in-play Pokémon also have:
  "hp": int, "maxHp": int,        # damage = maxHp - hp
  "energies": [...], "energyCards": [...],
  "tools": [...], "preEvolution": [...],   # evolution stack underneath
  "appearThisTurn": bool
}
```

> The card DB (`id` → name/HP/attacks) is NOT in the obs. Pull it from the
> official engine / sample kernels (`tools/fetch_engine.sh`, `pull_samples.sh`).

## Decision contexts (`select.context`) — measured

| context | meaning | option fields |
|---|---|---|
| `0` MAIN | main action menu | `type, area, index, attackId?, inPlayArea, inPlayIndex` |
| `1,2,3,4,7,8,22` | target / card selection | `type(=3), area, index, playerIndex` |
| `30` ENERGY | energy attach / distribute | `type, area, index, playerIndex, energyIndex, count` |
| `38` NUMBER | choose a number | `type, number` |
| `41` BINARY | simple A/B choice | `type` |

### MAIN-menu option types (measured)
- **`13` = attack** (exactly matches options carrying `attackId`).
- **`14` = end turn** (always present, always last).
- `{7,8,9,10,12}` = develop actions (play / attach / evolve / retreat / ability) —
  exact names TODO from the sample kernel.

**Key strategic fact (measured, not assumed):** attacking **ends the turn**, so a
strong agent does **all develop actions first and attacks last**. Forcing attacks
early loses *to random* (0/8); develop-first beats random ~87%. Codes live in
`src/ptcg/engine_codes.py` — calibrate there.

## Running a game (inside Docker / Linux)

```python
from kaggle_environments import make
env = make("cabt")
env.run([agent_a, agent_b])         # each agent returns its own 60-card deck first
print(env.state[0]["reward"], env.state[1]["reward"])   # +1 / -1 / 0, zero-sum
html = env.render(mode="html")      # interactive replay
```

## Config & limits (`cabt.json`)
- `episodeSteps: 10000`, `actTimeout: 0`, `runTimeout: 3000`,
  `remainingOverageTime: 600` → effectively a **~600s thinking pool per agent per
  game** (budget search by wall-clock; no hard per-move cap).
- Reward enum `[-1, 0, 1]`. 2 agents, zero-sum.

## Engine modules (bundled wheel)
- `cg.game`: `battle_start(deck0, deck1)`, `battle_select(list[int])`,
  `battle_finish()`, `visualize_data()`.
- `cg.sim`: `Battle` (holds `battle_ptr`, `obs`), ctypes bindings.
- **Not in the wheel:** `search_begin/step/end`, `all_card_data()` — these are in
  the official engine (`tools/fetch_engine.sh`) and power Layer-2 MCTS.
