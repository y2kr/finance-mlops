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
