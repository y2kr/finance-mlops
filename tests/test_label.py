import json

import pytest

from finance_mlops import label


def test_load_rubric_returns_text_and_version(tmp_path):
    f = tmp_path / "LABELING.md"
    f.write_text("# Rubric\n\n`rubric_version: 3`\n\n- bullish\n")
    text, version = label.load_rubric(f)
    assert version == 3
    assert "bullish" in text


def test_build_messages_quotes_rubric_and_headline():
    msgs = label.build_messages("Apple soars", "RUBRIC TEXT")
    assert msgs[0]["role"] == "system" and "RUBRIC TEXT" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "Apple soars"}


def test_parse_coerces_confidence_and_keeps_rationale():
    rec = label.parse(json.dumps({"label": "bullish", "confidence": 1, "rationale": "up"}))
    assert rec == {"label": "bullish", "confidence": 1.0, "rationale": "up"}


def test_parse_rejects_unknown_label():
    with pytest.raises(ValueError):
        label.parse(json.dumps({"label": "moon", "confidence": 0.9, "rationale": "x"}))


def test_parse_rejects_non_json():
    with pytest.raises(json.JSONDecodeError):
        label.parse("not json")
