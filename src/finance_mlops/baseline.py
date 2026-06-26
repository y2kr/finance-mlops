from __future__ import annotations

import logging
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

log = logging.getLogger("baseline")

POLARITY = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
DATA_DIR = Path(os.environ.get("PHRASEBANK_DIR", "data/phrasebank"))
ALL_AGREE = "Sentences_AllAgree.txt"
SEVENTYFIVE_AGREE = "Sentences_75Agree.txt"


def load_phrasebank(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        text, sep, polarity = line.rpartition("@")
        if sep and text.strip() and polarity in POLARITY:
            rows.append((text.strip(), POLARITY[polarity]))
    return rows


def train_eval_split(data_dir: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    eval_rows = load_phrasebank(data_dir / ALL_AGREE)
    held_out = {t for t, _ in eval_rows}
    train_rows = [r for r in load_phrasebank(data_dir / SEVENTYFIVE_AGREE) if r[0] not in held_out]
    return train_rows, eval_rows


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def run(data_dir: Path = DATA_DIR) -> float:
    train_rows, eval_rows = train_eval_split(data_dir)
    model = build_model()
    model.fit([t for t, _ in train_rows], [y for _, y in train_rows])
    pred = model.predict([t for t, _ in eval_rows])
    true = [y for _, y in eval_rows]
    macro_f1 = f1_score(true, pred, average="macro")
    log.info("trained on %d, evaluated on %d", len(train_rows), len(eval_rows))
    log.info("\n%s", classification_report(true, pred, digits=3))
    log.info("3-class macro-F1 on AllAgree: %.3f", macro_f1)
    return macro_f1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
