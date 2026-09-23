from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import psycopg
from mlflow import MlflowClient

from finance_mlops.baseline import (
    DATA_DIR,
    MODEL_NAME,
    TRACKING_URI,
    build_model,
    evaluate,
    train_eval_split,
)
from finance_mlops.ingest import SCHEMA as NEWS_SCHEMA
from finance_mlops.label import SCHEMA as LABEL_SCHEMA
from finance_mlops.settings import DB_URL

log = logging.getLogger("retrain")
MIN_CONFIDENCE = float(os.environ.get("MIN_WEAK_LABEL_CONFIDENCE", "0.8"))
MIN_NEW_LABELS = int(os.environ.get("MIN_NEW_WEAK_LABELS", "200"))
BEARISH_FLOOR = float(os.environ.get("BEARISH_F1_FLOOR", "0.525"))

STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_state (
    key text PRIMARY KEY,
    value bigint NOT NULL
)
"""
SELECT_LABELS = """
SELECT w.id, n.headline, w.label FROM weak_labels w
JOIN news n ON n.id = w.news_id
WHERE w.confidence >= %s AND w.label IN ('bullish', 'bearish', 'neutral')
ORDER BY w.id
"""


def weak_rows(conn, confidence: float = MIN_CONFIDENCE) -> list[tuple[int, str, str]]:
    return list(conn.execute(SELECT_LABELS, (confidence,)).fetchall())


def last_used_label(conn) -> int:
    conn.execute(STATE_SCHEMA)
    row = conn.execute(
        "SELECT value FROM pipeline_state WHERE key = 'last_weak_label_id'"
    ).fetchone()
    return row[0] if row else 0


def mark_labels_used(conn, label_id: int) -> None:
    conn.execute(
        """INSERT INTO pipeline_state (key, value) VALUES ('last_weak_label_id', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
        (label_id,),
    )
    conn.commit()


def training_rows(data_dir: Path, weak: list[tuple[str, str]]) -> tuple[list, list]:
    gold, evaluation = train_eval_split(data_dir)
    blocked = {headline for headline, _ in evaluation}
    return gold + [row for row in weak if row[0] not in blocked], evaluation


def should_promote(metrics: dict, champion_macro_f1: float) -> bool:
    return metrics["macro_f1"] >= champion_macro_f1 and metrics["bearish_f1"] >= BEARISH_FLOOR


def run(data_dir: Path = DATA_DIR, minimum: int = MIN_NEW_LABELS) -> bool:
    with psycopg.connect(DB_URL) as conn:
        locked = conn.execute(
            "SELECT pg_try_advisory_lock(hashtext('finance_mlops_retrain'))"
        ).fetchone()[0]
        if not locked:
            log.info("skip retraining: another run holds the lock")
            return False
        conn.execute(NEWS_SCHEMA)
        conn.execute(LABEL_SCHEMA)
        weak = weak_rows(conn)
        last_id = last_used_label(conn)
        new_count = sum(label_id > last_id for label_id, _, _ in weak)
        if new_count < minimum:
            log.info("skip retraining: %d new eligible weak labels, need %d", new_count, minimum)
            return False
        train, evaluation = training_rows(data_dir, [(text, label) for _, text, label in weak])
        model = build_model()
        headlines, labels = zip(*train, strict=True)
        model.fit(headlines, labels)
        predictions = model.predict([text for text, _ in evaluation])
        metrics = evaluate([label for _, label in evaluation], predictions)
        promoted = register(model, train, evaluation, metrics)
        mark_labels_used(conn, weak[-1][0])
    log.info("challenger macro-F1 %.3f, promoted=%s", metrics["macro_f1"], promoted)
    return promoted


def register(model, train, evaluation, metrics) -> bool:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champion_f1 = float(client.get_run(champion.run_id).data.metrics["macro_f1"])
    with mlflow.start_run(run_name="retrain-tfidf-logreg") as active:
        mlflow.log_params(
            {"n_train": len(train), "n_eval": len(evaluation), "weak_confidence": MIN_CONFIDENCE}
        )
        mlflow.log_metrics({"macro_f1": metrics["macro_f1"], "bearish_f1": metrics["bearish_f1"]})
        mlflow.log_text(metrics["report"], "classification_report.txt")
        mlflow.sklearn.log_model(model, name="model")
        uri = f"runs:/{active.info.run_id}/model"
    version = mlflow.register_model(uri, MODEL_NAME).version
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    promoted = should_promote(metrics, champion_f1)
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    return promoted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
