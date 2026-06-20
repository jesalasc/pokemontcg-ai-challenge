# The cabt engine (libcg.so) is a Linux x86-64 ELF binary. It cannot run on
# Apple Silicon natively, so all engine execution happens in this container.
# Build/run with --platform=linux/amd64 (Docker Desktop emulates it on Mac).
FROM --platform=linux/amd64 python:3.11-slim

# libgomp1: the engine .so may link OpenMP; install defensively so dlopen works.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install deps first for layer caching.
COPY requirements.txt requirements-rl.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src

# Source is bind-mounted at runtime (see docker-compose.yml) so edits are live.
CMD ["python", "-c", "from kaggle_environments import make; make('cabt'); print('cabt engine OK')"]
