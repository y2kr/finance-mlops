from __future__ import annotations

import os
import threading
import time
from contextlib import suppress

from finance_mlops.baseline import MODEL_NAME, TRACKING_URI
from finance_mlops.settings import DB_URL

ALIAS = os.environ.get("MODEL_ALIAS", "champion")
POLL_SECONDS = int(os.environ.get("SERVE_POLL_SECONDS", "60"))
LABELS = ("bearish", "bullish", "neutral")
SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id bigserial PRIMARY KEY,
    headline text NOT NULL,
    predicted_class text NOT NULL,
    confidence double precision NOT NULL,
    probabilities jsonb NOT NULL,
    model_version text NOT NULL,
    predicted_at timestamptz NOT NULL DEFAULT now()
)
"""
INSERT = """
INSERT INTO prediction_logs (headline, predicted_class, confidence, probabilities, model_version)
VALUES (%(headline)s, %(predicted_class)s, %(confidence)s, %(probabilities)s, %(model_version)s)
"""


try:
    import bentoml
except ImportError:
    bentoml = None


try:
    from prometheus_client import Counter

    REQUESTS = Counter("serve_predict_requests_total", "Prediction requests")
    PREDICTIONS = Counter("serve_predictions_total", "Predictions", ["sentiment"])
except ImportError:
    REQUESTS = PREDICTIONS = None


class Registry:
    def __init__(self):
        self._state = (None, None)
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            return self._state

    def refresh(self) -> bool:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(TRACKING_URI)
        version = str(MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)
        if version == self.state[1]:
            return False
        uri = f"models:/{MODEL_NAME}@{ALIAS}"
        with suppress(Exception):
            model = mlflow.sklearn.load_model(uri)
            with self._lock:
                self._state = (model, version)
            return True
        model = mlflow.pyfunc.load_model(uri)
        with self._lock:
            self._state = (model, version)
        return True

    def loop(self, seconds: int = POLL_SECONDS) -> None:
        while True:
            with suppress(Exception):
                self.refresh()
            time.sleep(seconds)


registry = Registry()


def parse_headlines(data) -> list[str]:
    headlines = data.get("headlines") if isinstance(data, dict) else data
    if (
        not isinstance(headlines, list)
        or not headlines
        or not all(isinstance(x, str) and x for x in headlines)
    ):
        raise ValueError("expected {'headlines': [non-empty strings]}")
    return headlines


def rows_for_output(raw, labels=LABELS) -> list[dict[str, float]]:
    if hasattr(raw, "to_dict"):
        return raw.to_dict("records")
    out = []
    for row in raw:
        if isinstance(row, dict):
            if "label" in row and "score" in row:
                out.append({k: (row["score"] if k == row["label"] else 0.0) for k in labels})
            else:
                out.append({k: float(row[k]) for k in row if k in labels})
        elif isinstance(row, str):
            out.append({k: (1.0 if k == row else 0.0) for k in labels})
        else:
            out.append(dict(zip(labels, map(float, row), strict=False)))
    return out


def probabilities(model, headlines: list[str]) -> list[dict[str, float]]:
    if hasattr(model, "predict_proba"):
        labels = tuple(getattr(model, "classes_", LABELS))
        return rows_for_output(model.predict_proba(headlines), labels)
    return rows_for_output(model.predict(headlines))


def predict_payload(data, state=None) -> dict:
    headlines = parse_headlines(data)
    model, version = state or registry.state
    if model is None:
        raise RuntimeError("model not loaded")
    rows = probabilities(model, headlines)
    predictions = []
    for headline, probs in zip(headlines, rows, strict=True):
        label, confidence = max(probs.items(), key=lambda x: x[1])
        predictions.append(
            {
                "headline": headline,
                "class": label,
                "confidence": confidence,
                "probabilities": probs,
                "model_version": version,
            }
        )
        if PREDICTIONS:
            PREDICTIONS.labels(label).inc()
    log_predictions(predictions)
    if REQUESTS:
        REQUESTS.inc()
    return {"predictions": predictions}


def log_predictions(predictions: list[dict]) -> None:
    import json

    import psycopg

    rows = [
        {
            "headline": p["headline"],
            "predicted_class": p["class"],
            "confidence": p["confidence"],
            "probabilities": json.dumps(p["probabilities"]),
            "model_version": p["model_version"],
        }
        for p in predictions
    ]
    with psycopg.connect(DB_URL) as conn:
        conn.execute(SCHEMA)
        with conn.cursor() as cursor:
            cursor.executemany(INSERT, rows)
        conn.commit()


def start() -> None:
    with suppress(Exception):
        registry.refresh()
    threading.Thread(target=registry.loop, daemon=True).start()


if bentoml:
    from fastapi import FastAPI

    health_app = FastAPI()

    @health_app.get("/health")
    def health() -> dict:
        return {"ok": registry.state[0] is not None, "model_version": registry.state[1]}

    @bentoml.service
    @bentoml.asgi_app(health_app)
    class FinanceSentiment:
        def __init__(self):
            start()

        @bentoml.api(route="/predict")
        def predict(self, headlines: list[str]) -> dict:
            return predict_payload({"headlines": headlines})
else:
    FinanceSentiment = None


if __name__ == "__main__":
    start()
