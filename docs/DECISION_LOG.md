# Decision log & story

The narrative of *why* we did what we did — the raw material for the Strategy
writeup (`docs/REPORT.md`). Two parts: a running dated log (append newest on top),
and longer "decision records" for the big calls. Persisted in git, so it survives.

**Practice:** every real choice (deck card, ratio, heuristic, algorithm, reward,
hyperparameter) gets a one-line log entry the day we make it. Claude appends these
as we work; the team can too. Capture the *reasoning and the evidence*, not just
the what — that's what the judges score.

---

## Running log
Format: `YYYY-MM-DD — decision — why — evidence`

- 2026-06-20 — Built **human-play capture** (`make play`) → demos → `bc_pretrain` → `az-train --init`: the way domain insight enters the learned system (insight-as-DATA, not code). Verified end-to-end (piloted a full game, 55 decisions captured, BC imitation match 0.60→0.78). Four insight-injection points documented in `docs/TRAINING.md`.
- 2026-06-20 — Roster deck added: **Crustle / Mega Kangaskhan ex** (`decks/crustle.csv`, legal). Subs vs the pool: Pokémon Center Lady→Cook (#1212; Wally's Compassion #1229 a stronger Mega-synergy option), "Growing Grass Energy"→Grow Grass (#18), Crustle line = Ascension→Superb Scissors. Pattern: meta lists include cards newer than the competition's curated pool → some slots need legal subs.
- 2026-06-20 — Training approach: **AlphaZero-style, deck-conditioned, population self-play**; strategy is LEARNED (terminal reward only, learned value replaces heuristics) — Jesús's directive: no hand-coded strategy. One net conditioned on the deck pilots the whole roster; matchup play emerges from training across pairings. Pipeline verified end-to-end (228/228 per-move searches OK; self-play→train→checkpoint). Real scale runs on GPU via `make az-train`. See `docs/TRAINING.md`.
- 2026-06-20 — Flagship deck v1: **Dragapult ex / Dusknoir** control-spread (from Jesús's Pokémon Live list) — translated to competition IDs, saved `decks/dragapult-dusknoir.csv` (validated 60/60). NOTE: "Special Red Card" (CRI 82) is NOT in the legal pool; Jesús picked the 60th card himself.
- 2026-06-20 — Will pick our flagship deck from the legal pool ourselves (not ship a stock list) — deck-construction originality is explicitly judged in the Strategy track.
- 2026-06-20 — Ported the deck-specific Dragapult sample as the competitive Layer-1 (`dragapult`) — a generic heuristic can't pilot a Stage-2 deck — measured **93.75%** vs random (deck-specific) vs **25%** (generic) on the Dragapult deck.
- 2026-06-20 — Baseline rule: develop everything, attack LAST, end-turn only if forced — attacking ends the turn, so attacking early skips setup — generic agent went 0% → ~90% vs random on a simple deck after this fix.
- 2026-06-20 — Run the engine via a linux/amd64 Docker image — `libcg.so` is Linux-only, no macOS build — engine imports/plays cleanly in the container.
- 2026-06-20 — Pinned `kaggle-environments==1.30.1` — it bundles the exact ladder engine — `make('cabt')` runs identically locally.

---

## Decision records (the big calls)

Use this template for choices worth a paragraph in the report:

### ADR-NNN — <title>   (YYYY-MM-DD)
- **Context:** what forced the decision.
- **Options considered:** A / B / C.
- **Decision:** what we chose.
- **Why:** the reasoning (domain + engineering).
- **Evidence:** win-rate / Elo / replay link (`artifacts/...`).
- **Revisit if:** the condition that would change our mind.

### ADR-001 — Layered agent: rule-based → search → RL   (2026-06-20)
- **Context:** imperfect-info, stochastic, long-horizon game; 5 subs/day; report judges ingenuity + stability.
- **Decision:** build in layers, each independently deployable (`docs/ARCHITECTURE.md`).
- **Why:** always have a ladder-ready agent; each layer is a sparring partner / fallback for the next.
- **Evidence:** baseline 90% vs random (simple deck); deck-specific 93.75% (Dragapult); MCTS forward model verified.
- **Revisit if:** a single approach dominates in local eval.
