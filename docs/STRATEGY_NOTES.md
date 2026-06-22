# Strategy design notes (principles to apply as we build)

Forward-looking design ideas. Each lists **where it plugs into our pipeline** so we
apply it at the right stage (see `docs/ROADMAP.md`).

## 1. Matchup-aware play (adapt to the opponent in front)

Every deck has counters and favorable matchups, so the agent should adapt its line
to the opponent it's actually facing.

**Key constraint (verified):** deck selection is *blind* — at the first call there's
no opponent info, so there's **no pre-game counter-pick**. Therefore matchup
adaptation happens **in-game**:

- **Opponent modelling** (new `opponent_model.py`): classify the opponent's deck /
  archetype from *revealed* information — their active/bench, discard pile, and the
  `logs` (cards played/attached/evolved). Confidence grows as the game reveals more.
- **Adapt the game plan** on the inferred matchup: favorable → press tempo, close
  fast; unfavorable → grind, play around their key answers (e.g. sequence to dodge
  their disruption, bait counters, manage prizes to deny multi-prize turns).
- **Feeds two places:**
  - `heuristics.py` — shift evaluation weights by inferred matchup (race vs grind).
  - `forward_model.py` — replace the placeholder opponent belief in determinization
    with the *predicted opponent deck* (much stronger MCTS).
- **Encode a matchup table** for our deck (favorable/even/unfavorable vs common
  archetypes) — also a great figure for the Strategy report.

## 2. Archetype core strategy (pilot to the deck's identity)

Decks have a core style — **aggressive / tempo / control / defensive / combo** — and
that identity should drive how the agent plays it.

- **A `style` profile** parameterizes the evaluation (`heuristics.py` weights):
  - *Aggressive:* maximize prize speed / KO output; spend resources freely.
  - *Tempo:* trade efficiently, keep board pressure, deny setup.
  - *Control / defensive:* value disruption, resource denial, longevity; patient
    prize race; protect key attackers.
  - *Combo:* value assembling the engine; tolerate slow starts.
- **Set the profile per deck.** Our flagship deck declares its style; the weights
  follow. If we ever field multiple deck-agents, each carries its own profile.
- **Also shapes RL reward** (`training/ppo.py:shape_reward`): an aggressive deck
  rewards prize tempo; a control deck rewards disruption/resource edge.

> Net: `style` sets the *baseline* plan; the inferred *matchup* modulates it
> (e.g. an aggressive deck still grinds patiently into an unfavorable control matchup).
