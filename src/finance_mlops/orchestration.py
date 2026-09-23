from __future__ import annotations

import os
from contextlib import suppress
from importlib import import_module

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    RunRequest,
    ScheduleDefinition,
    SkipReason,
    asset,
    define_asset_job,
    sensor,
)


def _connect():
    ingest = import_module("finance_mlops.ingest")
    return import_module("psycopg").connect(ingest.DB_URL)


def _done(name: str, value):
    if gateway := os.environ.get("PUSHGATEWAY_URL"):
        with suppress(Exception):
            prometheus = import_module("prometheus_client")
            registry = prometheus.CollectorRegistry()
            prometheus.Gauge(
                "finance_pipeline_last_success_timestamp",
                "Last successful pipeline run",
                registry=registry,
            ).set_to_current_time()
            prometheus.push_to_gateway(
                gateway, job="finance_pipeline", grouping_key={"pipeline": name}, registry=registry
            )
    return value


@asset
def ingest_news() -> int:
    ingest = import_module("finance_mlops.ingest")
    with _connect() as conn:
        result = ingest.ingest(conn)
    return _done("ingest", result)


@asset
def weak_label_news() -> int:
    try:
        import_module("ollama").list()
    except Exception:
        return 0
    label = import_module("finance_mlops.label")
    with _connect() as conn:
        result = label.label_batch(conn)
    return _done("weak_label", result)


@asset
def bootstrap_baseline() -> float | None:
    return _done("bootstrap", import_module("finance_mlops.baseline").bootstrap())


@asset
def retrain_tfidf() -> bool:
    return _done("retrain", import_module("finance_mlops.retrain").run())


@asset
def monitor_model() -> dict:
    return _done("monitor", import_module("finance_mlops.monitor").run())


hourly_ingest_job = define_asset_job("hourly_ingest", selection=AssetSelection.assets(ingest_news))
daily_weak_label_job = define_asset_job(
    "daily_weak_label", selection=AssetSelection.assets(weak_label_news)
)
bootstrap_baseline_job = define_asset_job(
    "bootstrap_model", selection=AssetSelection.assets(bootstrap_baseline)
)
weekly_tfidf_retraining_job = define_asset_job(
    "weekly_tfidf_retraining", selection=AssetSelection.assets(retrain_tfidf)
)
monitoring_job = define_asset_job("monitoring", selection=AssetSelection.assets(monitor_model))

hourly_ingest_schedule = ScheduleDefinition(
    job=hourly_ingest_job,
    cron_schedule="0 * * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)
daily_weak_label_schedule = ScheduleDefinition(
    job=daily_weak_label_job,
    cron_schedule="0 2 * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)
weekly_tfidf_retraining_schedule = ScheduleDefinition(
    job=weekly_tfidf_retraining_job,
    cron_schedule="0 3 * * 0",
    default_status=DefaultScheduleStatus.RUNNING,
)
hourly_monitoring_schedule = ScheduleDefinition(
    job=monitoring_job,
    cron_schedule="30 * * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)


@sensor(
    job=weekly_tfidf_retraining_job,
    minimum_interval_seconds=3600,
    default_status=DefaultSensorStatus.RUNNING,
)
def monitoring_sensor():
    trigger = import_module("finance_mlops.monitor").retrain_trigger()
    if trigger:
        return RunRequest(run_key=str(trigger))
    return SkipReason("no sustained monitoring breach")


definitions = Definitions(
    assets=[ingest_news, weak_label_news, bootstrap_baseline, retrain_tfidf, monitor_model],
    jobs=[
        hourly_ingest_job,
        daily_weak_label_job,
        bootstrap_baseline_job,
        weekly_tfidf_retraining_job,
        monitoring_job,
    ],
    schedules=[
        hourly_ingest_schedule,
        daily_weak_label_schedule,
        weekly_tfidf_retraining_schedule,
        hourly_monitoring_schedule,
    ],
    sensors=[monitoring_sensor],
)
