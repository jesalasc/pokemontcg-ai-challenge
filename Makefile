# PTCG AI Battle — dev commands. Everything that touches the engine runs in the
# linux/amd64 Docker image (the engine is Linux-only).

IMAGE := ptcg-dev
PLATFORM := linux/amd64
RUN := docker run --rm --platform $(PLATFORM) -e PYTHONPATH=/workspace:/workspace/src \
       -v "$(CURDIR)":/workspace -w /workspace $(IMAGE)
# Filter the noisy OpenSpiel banner from engine output.
QUIET := 2>&1 | grep -viE "open_spiel|INFO:" || true

A ?= rule_based
B ?= random
N ?= 100

.PHONY: help build shell test eval baseline-check replay dump-obs \
        submission distill ppo pull-samples fetch-engine clean

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

clean: ## Remove regenerable artifacts
	rm -rf artifacts/*.py artifacts/replays artifacts/eval __pycache__ .pytest_cache
