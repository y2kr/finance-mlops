import sys
import types

import pytest

from finance_mlops import serve


class ProbaModel:
    classes_ = ("bearish", "bullish", "neutral")

    def predict_proba(self, headlines):
        return [[0.1, 0.7, 0.2] for _ in headlines]


def test_predict_payload_shapes_batch_and_logs(monkeypatch):
    logged = []
    monkeypatch.setattr(serve, "log_predictions", logged.extend)
    out = serve.predict_payload({"headlines": ["profit jumps", "shares fall"]}, (ProbaModel(), "7"))
    assert out == {
        "predictions": [
            {
                "headline": "profit jumps",
                "class": "bullish",
                "confidence": 0.7,
                "probabilities": {"bearish": 0.1, "bullish": 0.7, "neutral": 0.2},
                "model_version": "7",
            },
            {
                "headline": "shares fall",
                "class": "bullish",
                "confidence": 0.7,
                "probabilities": {"bearish": 0.1, "bullish": 0.7, "neutral": 0.2},
                "model_version": "7",
            },
        ]
    }
    assert logged == out["predictions"]


@pytest.mark.parametrize("bad", [{}, {"headlines": []}, {"headlines": [""]}, {"headlines": "x"}])
def test_parse_headlines_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        serve.parse_headlines(bad)


def test_registry_refresh_loads_new_alias_atomically(monkeypatch):
    loaded = []

    class Client:
        def get_model_version_by_alias(self, name, alias):
            return types.SimpleNamespace(version=len(loaded) + 1)

    def load_model(uri):
        loaded.append(uri)
        return f"model-{len(loaded)}"

    mlflow = types.SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        pyfunc=types.SimpleNamespace(load_model=load_model),
        MlflowClient=Client,
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    registry = serve.Registry()
    assert registry.refresh() is True
    assert registry.state == ("model-1", "1")
    assert loaded == ["models:/finance-sentiment@champion"]
