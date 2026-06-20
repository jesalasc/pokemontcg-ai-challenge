#!/usr/bin/env bash
# Download competition files (official engine + card database) from Kaggle.
# Needs ~/.kaggle/kaggle.json AND accepted competition rules.
#
# Note: the engine that runs games is already bundled in
# kaggle-environments==1.30.1. This fetches the OFFICIAL package which may ship
# the richer search API (search_begin/step/end) and all_card_data() that power
# Layer-2 MCTS and card-aware heuristics.
set -euo pipefail

COMP="pokemon-tcg-ai-battle"
DEST="data/competition"
mkdir -p "$DEST"

echo ">> downloading competition files for $COMP"
kaggle competitions download -c "$COMP" -p "$DEST"

echo ">> unzipping"
cd "$DEST"
for z in *.zip; do [ -f "$z" ] && unzip -o "$z"; done

echo "Done -> $DEST"
echo "If a richer engine ships here (card DB / search API), wire it into:"
echo "  - src/ptcg/forward_model.py  (EngineForwardModel.* for MCTS)"
echo "  - a card-database loader for heuristics (id -> name/hp/attacks)"
