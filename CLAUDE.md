# Conventions

- No comments. Code must be self-explanatory through naming.
- Write the fewest lines that work. Prefer stdlib and one-liners over custom code.

# Running

- Full local stack: `docker compose up -d --build`
- Demo: `./scripts/demo`; integration smoke: `./scripts/smoke`
- Checks: `./scripts/check && uv run pytest`
- Dagster automates ingest, weak labels, monitoring, retraining, and promotion.
- FinBERT remains manual: `uv run --extra finetune python -m finance_mlops.finetune`.
- See `README.md` for setup, local URLs, architecture, and module commands.

When reporting information to me, be extremely concise and sacrifice grammer for the sake of concision.
