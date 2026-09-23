from finance_mlops import orchestration


def test_jobs_and_schedules_are_defined():
    assert [
        orchestration.hourly_ingest_job.name,
        orchestration.daily_weak_label_job.name,
        orchestration.bootstrap_baseline_job.name,
        orchestration.weekly_tfidf_retraining_job.name,
        orchestration.monitoring_job.name,
    ] == [
        "hourly_ingest",
        "daily_weak_label",
        "bootstrap_model",
        "weekly_tfidf_retraining",
        "monitoring",
    ]
    assert [
        orchestration.hourly_ingest_schedule.cron_schedule,
        orchestration.daily_weak_label_schedule.cron_schedule,
        orchestration.weekly_tfidf_retraining_schedule.cron_schedule,
        orchestration.hourly_monitoring_schedule.cron_schedule,
    ] == ["0 * * * *", "0 2 * * *", "0 3 * * 0", "30 * * * *"]


def test_definitions_load():
    assert orchestration.definitions is not None
