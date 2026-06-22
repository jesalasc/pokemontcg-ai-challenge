# PTCG AI Battle — dev commands. Everything that touches the engine runs in the
# linux/amd64 Docker image (the engine is Linux-only).

IMAGE := ptcg-dev
PLATFORM := linux/amd64
# data/engine on the path so engine-backed agents (mcts, dragapult) find cg.api.
RUN := docker run --rm --platform $(PLATFORM) \
       -e PYTHONPATH=/workspace:/workspace/src:/workspace/data/engine \
       -v "$(CURDIR)":/workspace -w /workspace $(IMAGE)
# Filter the noisy OpenSpiel banner from engine output.
QUIET := 2>&1 | grep -viE "open_spiel|INFO:" || true

A ?= rule_based
B ?= random
N ?= 100
ADECK ?= dragapult-ex
BDECK ?= dragapult-ex

.PHONY: help build shell test eval baseline-check replay dump-obs \
        submission distill ppo pull-samples fetch-engine cards deck-check deckbuilder probe \
        az-build az-train az-smoke play bc-pretrain replay-rec replay-view clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## Build the linux/amd64 engine image
	docker build --platform $(PLATFORM) -f docker/Dockerfile -t $(IMAGE) .

shell: ## Interactive shell in the container
	docker run --rm -it --platform $(PLATFORM) -e PYTHONPATH=/workspace:/workspace/src \
	  -v "$(CURDIR)":/workspace -w /workspace $(IMAGE) /bin/bash

test: ## Run the test suite (engine tests included)
	$(RUN) python -m pytest tests/ -q $(QUIET)

eval: ## Evaluate agents: make eval A=rule_based B=random N=100  (the submission gate)
	$(RUN) python harness/evaluate.py --a $(A) --b $(B) -n $(N) $(QUIET)

baseline-check: ## Quick: rule_based vs random, 20 games
	$(RUN) python harness/evaluate.py --a rule_based --b random -n 20 $(QUIET)

replay: ## Render an HTML replay: make replay A=rule_based B=random
	$(RUN) python harness/replay.py --a $(A) --b $(B) $(QUIET)

dump-obs: ## Capture real observations -> artifacts/schema/
	$(RUN) python harness/dump_obs.py --out artifacts/schema $(QUIET)

submission: ## Build + smoke-test a submission: make submission A=rule_based
	$(RUN) python tools/build_submission.py --agent $(A) --smoke $(QUIET)

distill: ## Behavioral-clone a net from the baseline (needs requirements-rl.txt)
	$(RUN) python training/distill.py --teacher rule_based --games 100 --epochs 10 $(QUIET)

ppo: ## Self-play PPO refinement (needs a warm-start checkpoint + GPU box)
	$(RUN) python training/ppo.py --init checkpoints/policy.pt $(QUIET)

pull-samples: ## kaggle kernels pull the 3 sample notebooks (needs ~/.kaggle/kaggle.json)
	bash tools/pull_samples.sh

fetch-engine: ## Download official engine + card DB from Kaggle
	bash tools/fetch_engine.sh

cards: ## Explore the legal card pool (host, no engine): make cards ARGS="--type pokemon --ex"
	python3 tools/card_explorer.py $(ARGS)

deck-check: ## Validate a deck's legality (host, no engine): make deck-check DECK=deck.csv
	python3 tools/deck_check.py $(DECK)

deckbuilder: ## Serve the local deck builder at http://localhost:8000 (host, no engine)
	python3 tools/serve_deckbuilder.py

probe: ## Probe engine rule behavior (coin fairness, end reasons): make probe N=50
	$(RUN) python tools/probe_rules.py -n $(N) $(QUIET)

# --- AlphaZero training (needs the ptcg-rl image: engine + torch) ---
RL_RUN := docker run --rm --platform $(PLATFORM) \
          -e PYTHONPATH=/workspace:/workspace/src:/workspace/data/engine \
          -v "$(CURDIR)":/workspace -w /workspace ptcg-rl

az-build: ## Build the RL image (engine + CPU torch)
	docker build --platform $(PLATFORM) -f docker/Dockerfile.rl -t ptcg-rl .

az-train: ## AlphaZero self-play training: make az-train ARGS="--iters 50 --games 16 --sims 64"
	$(RL_RUN) python training/az_train.py $(ARGS) $(QUIET)

az-smoke: ## Tiny end-to-end pipeline check (stub self-play + train)
	$(RL_RUN) python training/az_train.py --evaluator stub --iters 1 --games 1 --sims 2 --epochs 1 --out checkpoints/az_smoke.pt $(QUIET)

play: ## Interactive play + demo capture at http://localhost:8000 (you pilot a deck)
	docker run --rm -it --platform $(PLATFORM) \
	  -e PYTHONPATH=/workspace:/workspace/src:/workspace/data/engine -p 8000:8000 \
	  -v "$(CURDIR)":/workspace -w /workspace ptcg-rl python tools/serve_play.py

bc-pretrain: ## BC-pretrain the AZ net from your demos: make bc-pretrain ARGS="--deck crustle"
	$(RL_RUN) python training/bc_pretrain.py $(ARGS) $(QUIET)

replay-rec: ## Record a readable replay: make replay-rec A=dragapult B=rule_based ADECK=dragapult-dusknoir BDECK=crustle
	$(RL_RUN) python harness/record_replay.py --a $(A) --b $(B) --a-deck $(ADECK) --b-deck $(BDECK) $(QUIET)

replay-view: ## Watch recorded replays at http://localhost:8001 (host, no engine)
	python3 tools/serve_replay.py

clean: ## Remove regenerable artifacts
	rm -rf artifacts/*.py artifacts/replays artifacts/eval __pycache__ .pytest_cache
