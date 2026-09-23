from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import psycopg

from finance_mlops.settings import DB_URL

log = logging.getLogger("label")

LABELS = ("bullish", "bearish", "neutral", "irrelevant")
MODEL_TAG = os.environ.get("WEAK_LABEL_MODEL", "qwen3:14b")
PROMPT_VERSION = 1
BATCH = int(os.environ.get("WEAK_LABEL_BATCH", "200"))
LABELING_FILE = Path(os.environ.get("LABELING_FILE", "LABELING.md"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS weak_labels (
    id             bigserial PRIMARY KEY,
    news_id        bigint NOT NULL REFERENCES news(id),
    label          text   NOT NULL,
    confidence     real   NOT NULL,
    rationale      text,
    model_tag      text   NOT NULL,
    prompt_version int    NOT NULL,
    rubric_version int    NOT NULL,
    labelled_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (news_id, model_tag, prompt_version, rubric_version)
);
"""

SELECT_UNLABELED = """
SELECT n.id, n.headline FROM news n
WHERE NOT EXISTS (
    SELECT 1 FROM weak_labels w
    WHERE w.news_id = n.id
      AND w.model_tag = %(model_tag)s
      AND w.prompt_version = %(prompt_version)s
      AND w.rubric_version = %(rubric_version)s
)
LIMIT %(limit)s
"""

INSERT = """
INSERT INTO weak_labels
    (news_id, label, confidence, rationale, model_tag, prompt_version, rubric_version)
VALUES
    (%(news_id)s, %(label)s, %(confidence)s, %(rationale)s, %(model_tag)s,
     %(prompt_version)s, %(rubric_version)s)
ON CONFLICT (news_id, model_tag, prompt_version, rubric_version) DO NOTHING
"""

FORMAT = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "rationale"],
}


def load_rubric(path: Path) -> tuple[str, int]:
    text = path.read_text()
    version = int(re.search(r"rubric_version:\s*(\d+)", text).group(1))
    return text, version


def build_messages(headline: str, rubric: str) -> list[dict]:
    system = (
        f"{rubric}\n\nApply the rubric above to the headline. Reply only with JSON: "
        '{"label", "confidence" (0-1), "rationale"}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": headline}]


def parse(content: str) -> dict:
    data = json.loads(content)
    if data["label"] not in LABELS:
        raise ValueError(f"unknown label {data['label']!r}")
    return {
        "label": data["label"],
        "confidence": float(data["confidence"]),
        "rationale": data.get("rationale"),
    }


def chat(headline: str, rubric: str) -> str:
    import ollama

    resp = ollama.chat(
        model=MODEL_TAG,
        messages=build_messages(headline, rubric),
        format=FORMAT,
        options={"temperature": 0},
        think=False,
    )
    return resp["message"]["content"]


def label_batch(conn) -> int:
    conn.execute(SCHEMA)
    rubric, rubric_version = load_rubric(LABELING_FILE)
    keys = {
        "model_tag": MODEL_TAG,
        "prompt_version": PROMPT_VERSION,
        "rubric_version": rubric_version,
    }
    inserted = 0
    for news_id, headline in conn.execute(SELECT_UNLABELED, {**keys, "limit": BATCH}).fetchall():
        try:
            rec = parse(chat(headline, rubric))
        except Exception:
            log.warning("skip news_id=%s: weak-label failed", news_id)
            continue
        inserted += conn.execute(INSERT, {"news_id": news_id, **rec, **keys}).rowcount
    conn.commit()
    return inserted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with psycopg.connect(DB_URL) as conn:
        n = label_batch(conn)
    log.info("wrote %d weak labels", n)


if __name__ == "__main__":
    main()
