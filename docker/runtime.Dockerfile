FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends coreutils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra ingest --extra label --extra baseline --extra finetune --extra serve --extra orchestrate --extra monitor
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra ingest --extra label --extra baseline --extra finetune --extra serve --extra orchestrate --extra monitor
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
