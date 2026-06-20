"""Layer-3 training: self-play data, distillation, and PPO.

Heavy deps (torch) live here, kept out of the submission/baseline path. Run on
the Linux GPU box (or the Docker image with requirements-rl.txt installed).

Recommended order:
  1. distill.py  — behavioral-cloning a net from rule_based / MCTS (fast warm start)
  2. ppo.py      — self-play PPO refinement from that warm start
"""
