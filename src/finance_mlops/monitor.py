from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psycopg

from finance_mlops.baseline import ALL_AGREE, DATA_DIR, load_phrasebank
from finance_mlops.settings import DB_URL

LABELS = ("bearish", "bullish", "neutral")
LIMIT = int(os.environ.get("MONITOR_LIMIT", "200"))
DRIFT_THRESHOLD = float(os.environ.get("MONITOR_DRIFT_THRESHOLD", "0.2"))
PERFORMANCE_THRESHOLD = float(os.environ.get("MONITOR_PERFORMANCE_THRESHOLD", "0.65"))
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL")
os.environ.setdefault("NML_DISABLE_USAGE_LOGGING", "1")

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_runs (
    id bigserial PRIMARY KEY,
    drifted boolean NOT NULL,
    breached boolean NOT NULL,
    consecutive_breaches int NOT NULL,
    metrics jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""

SELECT_PREDICTIONS = """
SELECT headline, predicted_class, confidence, probabilities
FROM prediction_logs
ORDER BY predicted_at DESC
LIMIT %(limit)s
"""

INSERT_RUN = """
INSERT INTO monitoring_runs (drifted, breached, consecutive_breaches, metrics)
VALUES (%(drifted)s, %(breached)s, %(consecutive_breaches)s, %(metrics)s::jsonb)
"""

SELECT_STREAK = """
SELECT breached FROM monitoring_runs ORDER BY created_at DESC, id DESC LIMIT 1
"""


def reference_frame(data_dir: Path = DATA_DIR, model=None) -> pd.DataFrame:
    rows = load_phrasebank(data_dir / ALL_AGREE)
    if model is None:
        return frame((text, label, 1.0, {label: 1.0}, label) for text, label in rows)
    from finance_mlops.serve import probabilities

    scores = probabilities(model, [text for text, _ in rows])
    return frame(
        (
            text,
            max(probs, key=probs.get),
            max(probs.values()),
            probs,
            target,
        )
        for (text, target), probs in zip(rows, scores, strict=True)
    )


def current_frame(conn, limit: int = LIMIT) -> pd.DataFrame:
    return frame(conn.execute(SELECT_PREDICTIONS, {"limit": limit}).fetchall())


def frame(rows) -> pd.DataFrame:
    rows = list(rows)
    columns = (
        ["headline", "prediction", "confidence", "probabilities", "target"][: len(rows[0])]
        if rows
        else []
    )
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return pd.DataFrame(
            columns=["headline", "prediction", "confidence", "probabilities", "headline_length"]
        )
    df["headline_length"] = df["headline"].str.len()
    df["confidence"] = df["confidence"].astype(float)
    for label in LABELS:
        df[f"proba_{label}"] = df["probabilities"].map(
            lambda value, label=label: float(value.get(label, 0))
        )
    return df


def distribution_gap(reference: pd.Series, current: pd.Series) -> float:
    r, c = Counter(reference), Counter(current)
    keys = set(r) | set(c)
    return sum(abs(r[k] / len(reference) - c[k] / len(current)) for k in keys) / 2 if keys else 0.0


def fallback_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref_mean = reference["headline_length"].mean()
    cur_mean = current["headline_length"].mean()
    length_drift = abs(cur_mean - ref_mean) / max(ref_mean, 1)
    prediction_drift = distribution_gap(reference["prediction"], current["prediction"])
    return {
        "headline_length_drift": float(length_drift),
        "prediction_distribution_drift": float(prediction_drift),
        "drifted": bool(max(length_drift, prediction_drift) >= DRIFT_THRESHOLD),
    }


def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        schema = DataDefinition(
            categorical_columns=["prediction"], numerical_columns=["headline_length"]
        )
        result = Report([DataDriftPreset(columns=["headline_length", "prediction"])]).run(
            reference_data=Dataset.from_pandas(reference, data_definition=schema),
            current_data=Dataset.from_pandas(current, data_definition=schema),
        )
        raw = result.dict() if hasattr(result, "dict") else result
        text = json.dumps(raw, default=str).lower()
        out = fallback_drift(reference, current)
        drifted = out["drifted"] or 'drifted": true' in text or "drift detected" in text
        shares = [
            metric.get("value", {}).get("share", 0)
            for metric in raw.get("metrics", [])
            if "DriftedColumnsCount" in metric.get("metric_name", "")
        ]
        out.update({"drifted": drifted or any(share >= 0.5 for share in shares), "evidently": raw})
        return out
    except Exception as exc:
        out = fallback_drift(reference, current)
        out["evidently_error"] = type(exc).__name__
        return out


def confidence_performance(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    try:
        import nannyml as nml

        cbpe = nml.CBPE(
            y_pred_proba={label: f"proba_{label}" for label in LABELS},
            y_pred="prediction",
            y_true="target",
            metrics=["f1", "accuracy"],
            problem_type="classification_multiclass",
            chunk_size=max(len(current), 1),
        ).fit(reference)
        result = cbpe.estimate(current).to_df()
        scores = result.select_dtypes("number").iloc[-1].dropna()
        accuracy = next((float(v) for k, v in scores.items() if "accuracy" in str(k)), None)
        return {"estimated_accuracy": accuracy or float(current["confidence"].mean())}
    except Exception as exc:
        return {
            "estimated_accuracy": float(current["confidence"].mean()),
            "nannyml_error": type(exc).__name__,
        }


def previous_breached(conn) -> bool:
    row = conn.execute(SELECT_STREAK).fetchone()
    return bool(row and row[0])


def persist(conn, drifted: bool, metrics: dict) -> int:
    breached = (
        drifted or metrics["performance"].get("estimated_accuracy", 1.0) < PERFORMANCE_THRESHOLD
    )
    streak = 2 if breached and previous_breached(conn) else int(breached)
    conn.execute(
        INSERT_RUN,
        {
            "drifted": drifted,
            "breached": breached,
            "consecutive_breaches": streak,
            "metrics": json.dumps(metrics),
        },
    )
    conn.commit()
    return streak


def monitor(conn, data_dir: Path = DATA_DIR, limit: int = LIMIT, model=None) -> dict:
    from finance_mlops.serve import SCHEMA as PREDICTION_SCHEMA

    conn.execute(SCHEMA)
    conn.execute(PREDICTION_SCHEMA)
    current = current_frame(conn, limit)
    if len(current) < limit:
        raise ValueError(f"need {limit} predictions, found {len(current)}")
    if model is None:
        import mlflow

        from finance_mlops.baseline import MODEL_NAME, TRACKING_URI

        mlflow.set_tracking_uri(TRACKING_URI)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    reference = reference_frame(data_dir, model)
    drift = evidently_drift(reference, current)
    performance = confidence_performance(reference, current)
    metrics = {"drift": drift, "performance": performance, "rows": len(current)}
    streak = persist(conn, drift["drifted"], metrics)
    return {"drifted": drift["drifted"], "consecutive_breaches": streak, "metrics": metrics}


def publish(result: dict) -> None:
    if not PUSHGATEWAY_URL:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    Gauge("finance_model_drifted", "Latest drift status", registry=registry).set(result["drifted"])
    Gauge(
        "finance_monitor_breach_streak", "Consecutive monitoring breaches", registry=registry
    ).set(result["consecutive_breaches"])
    Gauge("finance_estimated_accuracy", "NannyML estimated accuracy", registry=registry).set(
        result["metrics"]["performance"]["estimated_accuracy"]
    )
    push_to_gateway(PUSHGATEWAY_URL, job="finance_monitor", registry=registry)


def run(data_dir: Path = DATA_DIR, db_url: str = DB_URL, limit: int = LIMIT) -> dict:
    with psycopg.connect(db_url) as conn:
        try:
            result = monitor(conn, data_dir, limit)
        except ValueError as exc:
            return {"skipped": str(exc)}
    publish(result)
    return result


def retrain_trigger(db_url: str = DB_URL) -> int | None:
    try:
        with psycopg.connect(db_url) as conn:
            row = conn.execute(
                "SELECT id, consecutive_breaches FROM monitoring_runs "
                "ORDER BY created_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row and row[1] >= 2 else None
    except psycopg.errors.UndefinedTable:
        return None


def should_retrain(db_url: str = DB_URL) -> bool:
    return retrain_trigger(db_url) is not None


def main() -> None:
    print(json.dumps({**run(), "checked_at": datetime.now(UTC).isoformat()}))


if __name__ == "__main__":
    main()
