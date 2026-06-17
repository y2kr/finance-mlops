# finance-mlops

A self-hosted, end-to-end MLOps project: a small model classifies financial
news headlines as **bullish / bearish / neutral** per ticker. The model is
deliberately simple — the **operations loop around it is the real deliverable**
(tracking, versioning, orchestration, serving, monitoring, eval gates,
automated retraining).

See [`design.md`](design.md) for the full architecture and the stage-by-stage
plan.

## Stack

Python 3.12, managed with **uv**. Lint + format with **Ruff**. The wider
ML/MLOps stack (Polars, MLflow, DVC, Dagster, BentoML, Evidently, NannyML, …)
is added per stage as the code that needs it lands — see `design.md`.

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

## Status

Commit #1: project scaffold only. Stage 1 (data ingestion) is next.

## License

MIT — see [`LICENSE`](LICENSE).
