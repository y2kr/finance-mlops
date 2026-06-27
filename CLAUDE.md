# Conventions

- No comments. Code must be self-explanatory through naming.
- Write the fewest lines that work. Prefer stdlib and one-liners over custom code.

# Running

No orchestrator yet (Dagster is Stage 6) — each stage is run by hand:

- Stage 1 ingest: `uv run --extra ingest python -m finance_mlops.ingest` — RSS → Postgres `news`. Needs Postgres at `DATABASE_URL` (default `postgresql:///finance_mlops`).
- Stage 2 label: `uv run --extra label python -m finance_mlops.label` — Ollama weak-labels `news` → `weak_labels`. Needs Ollama + `qwen3:14b`.
- Stage 3 baseline: `uv run --extra baseline python -m finance_mlops.baseline` — TF-IDF + logreg on `data/phrasebank/`, reports 3-class macro-F1.
