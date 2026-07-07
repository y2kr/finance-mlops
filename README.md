# finance-mlops

A self-hosted, end-to-end MLOps project: a small model classifies financial
news headlines as **bullish / bearish / neutral** per ticker. The model is
simple — the **operations loop around it is the real deliverable**
(tracking, versioning, orchestration, serving, monitoring, eval gates,
automated retraining). This is a LEARNING excerise to fill my brain with
ML advancements.

See [`design.md`](design.md) for the full architecture and the stage-by-stage
plan.

## Dev setup

```bash
# install uv: https://docs.astral.sh/uv/
uv sync --dev            # create .venv and install dev deps
uv run pre-commit install # enable lint/format on commit

uv run ruff check .       # lint
uv run ruff format .      # format
uv run pytest             # tests
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push and PR.

## Platform (Stage 5)

Postgres, MinIO (S3-compatible object store) and an MLflow tracking server run as
one Compose stack. MLflow uses a dedicated `mlflow` database on the same Postgres
(reused, not a second instance) and stores artifacts in MinIO via proxied access;
DVC versions datasets to the same MinIO.

```bash
docker compose up -d --build   # postgres + minio + mlflow + ingest
```

An `ingest` service runs Stage 1 on a loop (`INGEST_INTERVAL`, default hourly)
so the RSS stream accumulates in the background — drift and retraining (Stages
8/10) need weeks of banked history that RSS can't backfill. Dagster replaces it
at Stage 6. Weak-labelling (Stage 2) stays manual (Ollama/VRAM), run in batches.

- **Docker access:** if you're not in the `docker` group, prefix with `sudo` or
  `sudo usermod -aG docker $USER && newgrp docker`.
- MLflow UI → http://localhost:5000 · MinIO console → http://localhost:9001
  (`minioadmin`/`minioadmin`). Override any default via a gitignored `.env`
  (`POSTGRES_*`, `MINIO_ROOT_*`).

**Data versioning (DVC → MinIO `dvc` bucket):**

```bash
uv run dvc pull    # fetch data/phrasebank/ from MinIO (after the stack is up)
uv run dvc push    # publish a new dataset version
```

## Running the pipeline (manual)

No orchestrator yet — Dagster lands at Stage 6. Until then each stage is a module
you run by hand, in order.

**Prerequisites**

- The **Compose stack up** (Postgres for Stages 1–2, MLflow for Stage 3). Default
  `DATABASE_URL` = `postgresql://finance:finance@localhost:5432/finance_mlops`,
  `MLFLOW_TRACKING_URI` = `http://localhost:5000`; both overridable.
- **Ollama** running with the weak-label model pulled — `ollama pull qwen3:14b`.
  Needed by Stage 2 only.
- **Financial PhraseBank v1.0** in `data/phrasebank/` — `uv run dvc pull` fetches
  it from MinIO (or unzip manually). Needed by Stage 3.

```bash
# Stage 1 — ingest: per-ticker RSS -> Postgres `news`
uv run --extra ingest python -m finance_mlops.ingest

# Stage 2 — weak-label: Ollama labels unlabelled `news` -> `weak_labels`
uv run --extra label python -m finance_mlops.label

# Stage 3 — baseline: TF-IDF + logreg on PhraseBank; logs the run to MLflow and
# registers it as `finance-sentiment` v1 @champion
uv run --extra baseline python -m finance_mlops.baseline

# Stage 4 — finetune: full fine-tune of pinned FinBERT on the same split; registers
# `finance-sentiment` v2 @challenger and prints the promotion gate (needs a CUDA GPU)
uv run --extra finetune python -m finance_mlops.finetune
```

Stage 4 trains on the GPU and never moves `@champion` — a passing challenger is
promoted by hand (the run prints the one-liner). Don't re-run Stage 3 after
promoting v2, or its unconditional `@champion` set will yank the alias back.

Common overrides: `DATABASE_URL`, `TICKERS_FILE` (Stage 1); `WEAK_LABEL_MODEL`,
`WEAK_LABEL_BATCH` (Stage 2); `PHRASEBANK_DIR`, `MLFLOW_TRACKING_URI` (Stages 3–4).
