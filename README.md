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

## Running the pipeline (manual)

No orchestrator yet — Dagster lands at Stage 6. Until then each stage is a module
you run by hand, in order.

**Prerequisites**

- **Postgres** reachable at `postgresql:///finance_mlops` (`createdb finance_mlops`);
  override with `DATABASE_URL`. Needed by Stages 1–2.
- **Ollama** running with the weak-label model pulled — `ollama pull qwen3:14b`.
  Needed by Stage 2 only.
- **Financial PhraseBank v1.0** unzipped into `data/phrasebank/` (gitignored) —
  must contain `Sentences_AllAgree.txt` and `Sentences_75Agree.txt`. Needed by
  Stage 3 only.

```bash
# Stage 1 — ingest: per-ticker RSS -> Postgres `news`
uv run --extra ingest python -m finance_mlops.ingest

# Stage 2 — weak-label: Ollama labels unlabelled `news` -> `weak_labels`
uv run --extra label python -m finance_mlops.label

# Stage 3 — baseline: TF-IDF + logreg on PhraseBank, reports 3-class macro-F1
uv run --extra baseline python -m finance_mlops.baseline
```

Common overrides: `DATABASE_URL`, `TICKERS_FILE` (Stage 1); `WEAK_LABEL_MODEL`,
`WEAK_LABEL_BATCH` (Stage 2); `PHRASEBANK_DIR` (Stage 3).
