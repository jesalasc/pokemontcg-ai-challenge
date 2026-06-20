#!/usr/bin/env bash
# Pull the official sample kernels for reference (needs ~/.kaggle/kaggle.json).
# These decode the engine's type/area codes and show rule-based + MCTS patterns.
set -euo pipefail

DEST="data/samples"
mkdir -p "$DEST"

KERNELS=(
  "kiyotah/reinforcement-learning-and-mcts-sample-code"
  "kiyotah/a-sample-rule-based-agent-dragapult-ex-deck"
  "kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck"
  "kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck"
  "kiyotah/a-sample-rule-based-agent-iono-s-deck"
  "kiyotah/how-to-output-local-battle-as-json-and-view"
)

for k in "${KERNELS[@]}"; do
  name="$(basename "$k")"
  echo ">> pulling $k -> $DEST/$name"
  kaggle kernels pull "$k" -p "$DEST/$name" -m
done

echo "Done. Reference them to CALIBRATE src/ptcg/engine_codes.py and the deck."
