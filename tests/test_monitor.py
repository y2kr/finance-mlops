import sys
from types import SimpleNamespace

import pytest

from finance_mlops import monitor


class Conn:
    def __init__(self, rows=(), last=None):
        self.rows = list(rows)
        self.last = last
        self.sql = []
        self.params = []
        self.commits = 0

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self.params.append(params)
        if sql == monitor.SELECT_PREDICTIONS:
            return SimpleNamespace(fetchall=lambda: self.rows)
        if sql == monitor.SELECT_STREAK:
            return SimpleNamespace(fetchone=lambda: self.last)
        return SimpleNamespace(fetchall=lambda: [], fetchone=lambda: None, rowcount=1)

    def commit(self):
        self.commits += 1


def phrasebank(tmp_path):
    (tmp_path / monitor.ALL_AGREE).write_text(
        "Profit rose@positive\nLoss widened@negative\nBoard met@neutral\n", encoding="latin-1"
    )
    return tmp_path


def test_current_frame_reads_latest_predictions():
    conn = Conn([("Apple rises", "bullish", 0.8, {"bullish": 0.8})])
    df = monitor.current_frame(conn, 1)
    assert df.iloc[0][["headline", "prediction", "confidence"]].to_dict() == {
        "headline": "Apple rises",
        "prediction": "bullish",
        "confidence": 0.8,
    }
    assert df.iloc[0]["proba_bullish"] == 0.8
    assert conn.params[-1] == {"limit": 1}


def test_fallback_drift_flags_prediction_shift(tmp_path):
    reference = monitor.reference_frame(phrasebank(tmp_path))
    current = monitor.frame(
        [("A" * 20, "bullish", 0.9, {"bullish": 0.9}), ("B" * 20, "bullish", 0.8, {"bullish": 0.8})]
    )
    out = monitor.fallback_drift(reference, current)
    assert out["drifted"] is True
    assert out["prediction_distribution_drift"] > 0


def test_monitor_persists_second_consecutive_breach(tmp_path, monkeypatch):
    conn = Conn([("x", "bullish", 0.9, {"bullish": 0.9})], last=(True,))
    monkeypatch.setattr(monitor, "reference_frame", lambda data_dir, model: monitor.frame([]))
    monkeypatch.setattr(monitor, "evidently_drift", lambda reference, current: {"drifted": True})
    monkeypatch.setattr(
        monitor,
        "confidence_performance",
        lambda reference, current: {"estimated_accuracy": 0.9},
    )
    out = monitor.monitor(conn, phrasebank(tmp_path), 1, object())
    assert out["drifted"] is True
    assert out["consecutive_breaches"] == 2
    assert any(sql == monitor.INSERT_RUN for sql in conn.sql)
    assert conn.commits == 1


def test_run_opens_psycopg_connection(tmp_path, monkeypatch):
    conn = Conn([("x", "bullish", 0.9)])

    class Context:
        def __enter__(self):
            return conn

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(monitor.psycopg, "connect", lambda url: Context())
    monkeypatch.setattr(monitor, "monitor", lambda c, data_dir, limit: {"ok": (c, data_dir, limit)})
    assert monitor.run(phrasebank(tmp_path), "db", 7)["ok"] == (conn, phrasebank(tmp_path), 7)


def test_confidence_performance_uses_nannyml_when_available(monkeypatch):
    calls = {}

    class Result:
        def to_df(self):
            import pandas as pd

            return pd.DataFrame({"estimated_accuracy": [0.88]})

    class CBPE:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit(self, data):
            return self

        def estimate(self, data):
            return Result()

    monkeypatch.setitem(sys.modules, "nannyml", SimpleNamespace(CBPE=CBPE))
    data = monitor.frame([("x", "bullish", 0.9, {"bullish": 0.9}, "bullish")])
    out = monitor.confidence_performance(data, data.drop(columns="target"))
    assert out["estimated_accuracy"] == 0.88
    assert calls["problem_type"] == "classification_multiclass"


def test_empty_current_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "reference_frame", lambda data_dir, model: monitor.frame([]))
    with pytest.raises(ValueError):
        monitor.monitor(Conn(), phrasebank(tmp_path), 1, object())
