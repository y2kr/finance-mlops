from finance_mlops import retrain


def test_training_rows_excludes_evaluation_leakage(tmp_path):
    (tmp_path / "Sentences_AllAgree.txt").write_text("Held@positive\n", encoding="latin-1")
    (tmp_path / "Sentences_75Agree.txt").write_text(
        "Held@positive\nGold@negative\n", encoding="latin-1"
    )
    train, evaluation = retrain.training_rows(tmp_path, [("Held", "bearish"), ("Fresh", "neutral")])
    assert train == [("Gold", "bearish"), ("Fresh", "neutral")]
    assert evaluation == [("Held", "bullish")]


def test_promotion_requires_both_floors(monkeypatch):
    monkeypatch.setattr(retrain, "BEARISH_FLOOR", 0.525)
    assert retrain.should_promote({"macro_f1": 0.7, "bearish_f1": 0.6}, 0.69)
    assert not retrain.should_promote({"macro_f1": 0.68, "bearish_f1": 0.6}, 0.69)
    assert not retrain.should_promote({"macro_f1": 0.7, "bearish_f1": 0.5}, 0.69)
