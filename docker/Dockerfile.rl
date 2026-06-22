# RL image = base engine image + CPU PyTorch. Used for self-play + training.
# On the GPU/Linux box, swap the torch install for the CUDA build instead.
FROM ptcg-dev

# CPU-only torch wheel (recent torch supports NumPy 2.x, so no downgrade needed).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir tqdm

CMD ["python", "-c", "import torch, kaggle_environments; print('rl image OK, torch', torch.__version__)"]
