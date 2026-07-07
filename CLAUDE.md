# Conventions

- No comments. Code must be self-explanatory through naming.
- Write the fewest lines that work. Prefer stdlib and one-liners over custom code.

# Running

No orchestrator yet (Dagster is Stage 6) — each stage is run by hand:

- Stage 1 ingest: `uv run --extra ingest python -m finance_mlops.ingest` — RSS → Postgres `news`. Needs Postgres at `DATABASE_URL` (default `postgresql:///finance_mlops`).
- Stage 2 label: `uv run --extra label python -m finance_mlops.label` — Ollama weak-labels `news` → `weak_labels`. Needs Ollama + `qwen3:14b`.
- Stage 3 baseline: `uv run --extra baseline python -m finance_mlops.baseline` — TF-IDF + logreg on `data/phrasebank/`, reports 3-class macro-F1, registers `finance-sentiment` v1 `@champion`.
- Stage 4 finetune: `uv run --extra finetune python -m finance_mlops.finetune` — full fine-tune of pinned FinBERT on the same `data/phrasebank/` split, registers v2 `@challenger`, prints the promotion gate. Needs a CUDA GPU + the MLflow stack.

Stage 5 platform (Postgres + MinIO + MLflow) runs as one Compose stack: `docker compose up -d --build`. DVC versions `data/phrasebank/` to MinIO — `uv run dvc pull` fetches it. MLflow UI at `http://localhost:5000`. See the README for details.
