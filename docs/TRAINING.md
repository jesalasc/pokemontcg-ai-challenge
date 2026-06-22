# Training: AlphaZero-style population self-play (no hand-coded strategy)

Strategy is **learned**, not coded. Reward is **terminal only** (win +1 / loss −1 /
draw 0); a learned value network replaces all heuristics. (Per the project rule, we
do not add heuristics or reward shaping unless explicitly asked.)

## The idea
A single **deck-conditioned** network learns to pilot a whole roster of meta decks.
"The agent for deck X" = the net conditioned on X. We generate games by pairing
decks from `decks/` against each other; training across every matchup makes
matchup-aware play **emerge** rather than be programmed.

## Components
- `src/ptcg/deckcode.py` — card-pool index + `deck_vector(ids)`: the deck as a
  count vector over the legal pool, fed to the net as the conditioning input.
- `training/networks.py::AZNet` — inputs (state features, deck vector, per-option
  features) → **policy prior** over legal options + **value** in [−1,1].
- `training/az_mcts.py` — net-guided **PUCT** over the engine forward model
  (`search_begin/step`). Leaf value = the net's value (no rollouts, no heuristics).
  Pluggable `Evaluator`: `NetEvaluator` (torch) or `StubEvaluator` (uniform, for
  plumbing tests). Returns the visit-count policy π.
- `training/az_selfplay.py` — plays a game with both seats searching; records
  `(state, deck_vector, π, player)` per move, labels each with the terminal result.
- `training/az_train.py` — generate self-play → train (policy cross-entropy vs π +
  value MSE vs outcome) → checkpoint → repeat.
- `src/ptcg/agents/az_agent.py` — play/submission: loads the net, runs net-guided
  search conditioned on our deck. Falls back to a legal move if net/engine absent.

## Imperfect information
The forward model needs a guess of the opponent's hidden cards (determinization).
v1 uses legal placeholders; during self-play we know both decklists, so the upgrade
is to sample the opponent's hidden cards from their **known** deck (better belief →
better targets). At ladder time, infer the opponent's deck from revealed cards
(see `docs/STRATEGY_NOTES.md`). Marked as a TODO in `forward_model.py`.

## Where & when your domain insight enters (without hand-coding strategy)

The rule "no hand-coded strategy" means insight enters as **data the agent learns
from**, never as heuristic weights or reward shaping. Four injection points, at
different times — none of them touch `heuristics.py`:

1. **Deck design — now, ongoing.** The decklist encodes win condition, ratios, and
   tech. Biggest, cleanest insight channel; entirely yours.

2. **Demonstrations → imitation warm-start — *before* a self-play run.**
   This is literally "show the agents how to pilot each deck / matchup." You play
   (or script) example games; we pre-train the net to imitate your decisions
   (behavioral cloning), then self-play RL refines *from that warm start*
   (`az_train --init`). Matchup-specific demos work because the agent sees the
   opponent's revealed cards just like you did — "vs control, play patiently" is
   learnable from your in-matchup games. (The brief calls this Scripted Policy
   Distillation.) Re-inject whenever you spot a weakness.

3. **Curriculum / matchup emphasis — when configuring a run.** You choose which
   deck pairings to train on and how often (over-sample hard matchups). Insight
   about the meta shapes *what* the agent practices, not *how* it plays.

4. **Replay review → targeted retraining — *between* iterations.** Watch its games,
   flag bad lines/matchups; that feedback drives the next round (more demos there,
   more curriculum weight, more search). The human-in-the-loop "self-improver".

The dividing line: **insight-as-data** (decks, demonstrations, curriculum, replay
feedback) is allowed and learnable; **insight-as-code** (hand-tuned eval/reward) is
what we avoid unless you explicitly ask. Point 2 is the main "teach then learn"
mechanism — it needs a demonstration-capture + BC-pretrain step (to build).

## Compute
Self-play + net must run together → the `ptcg-rl` Docker image (CPU torch) for the
smoke; the Linux GPU box (CUDA torch) for real scale. Actors generate games, the
learner trains on them — decoupled, so it scales horizontally.
