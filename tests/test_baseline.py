from finance_mlops import baseline


def test_load_phrasebank_maps_polarity(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text(
        "Profit rose@positive\nLoss widened@negative\nBoard met@neutral\n", encoding="latin-1"
    )
    assert baseline.load_phrasebank(f) == [
        ("Profit rose", "bullish"),
        ("Loss widened", "bearish"),
        ("Board met", "neutral"),
    ]


def test_train_eval_split_holds_out_allagree(tmp_path):
    (tmp_path / baseline.ALL_AGREE).write_text("A@positive\nB@neutral\n", encoding="latin-1")
    (tmp_path / baseline.SEVENTYFIVE_AGREE).write_text(
        "A@positive\nB@neutral\nC@negative\n", encoding="latin-1"
    )
    train_rows, eval_rows = baseline.train_eval_split(tmp_path)
    assert {t for t, _ in train_rows} == {"C"}
    assert {t for t, _ in eval_rows} == {"A", "B"}
