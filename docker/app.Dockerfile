FROM python:3.12-slim
RUN pip install --no-cache-dir "feedparser>=6" "psycopg[binary]>=3.2"
ENV PYTHONPATH=/app/src
WORKDIR /app
CMD ["sh", "-c", "while true; do python -m finance_mlops.ingest; sleep ${INGEST_INTERVAL:-3600}; done"]
