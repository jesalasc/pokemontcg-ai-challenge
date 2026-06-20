"""ptcg — agent package for the PTCG AI Battle Challenge (cabt engine).

Layers (see docs/ARCHITECTURE.md):
  Layer 1  rule_based   hand-coded heuristics (domain-expert driven)
  Layer 2  mcts         determinized MCTS over the engine forward model
  Layer 3  rl           policy/value net trained via self-play
"""

__version__ = "0.1.0"
