from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

log = logging.getLogger("baseline")

POLARITY = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
DATA_DIR = Path(os.environ.get("PHRASEBANK_DIR", "data/phrasebank"))
ALL_AGREE = "Sentences_AllAgree.txt"
SEVENTYFIVE_AGREE = "Sentences_75Agree.txt"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "finance-sentiment"
REPORT_FILE = Path("stage3-baseline-report.md")


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


def evaluate(y_true, y_pred) -> dict:
    scores = classification_report(y_true, y_pred, output_dict=True)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "bearish_f1": scores["bearish"]["f1-score"],
        "report": classification_report(y_true, y_pred, digits=3),
        "scores": scores,
    }


def run(data_dir: Path = DATA_DIR) -> float:
    train_rows, eval_rows = train_eval_split(data_dir)
    model = build_model()
    model.fit([t for t, _ in train_rows], [y for _, y in train_rows])
    pred = model.predict([t for t, _ in eval_rows])
    metrics = evaluate([y for _, y in eval_rows], pred)
    log.info("trained on %d, evaluated on %d", len(train_rows), len(eval_rows))
    log.info("\n%s", metrics["report"])
    log.info("3-class macro-F1 on AllAgree: %.3f", metrics["macro_f1"])
    track(model, train_rows, eval_rows, metrics)
    return metrics["macro_f1"]


def track(model, train_rows, eval_rows, metrics) -> None:
    tfidf = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name="baseline-tfidf-logreg") as active:
        mlflow.log_params(
            {
                "ngram_range": tfidf.ngram_range,
                "min_df": tfidf.min_df,
                "sublinear_tf": tfidf.sublinear_tf,
                "max_iter": clf.max_iter,
                "class_weight": clf.class_weight,
                "n_train": len(train_rows),
                "n_eval": len(eval_rows),
            }
        )
        scores = metrics["scores"]
        mlflow.log_metrics(
            {
                "macro_f1": metrics["macro_f1"],
                "bearish_f1": metrics["bearish_f1"],
                "bullish_f1": scores["bullish"]["f1-score"],
                "neutral_f1": scores["neutral"]["f1-score"],
                "accuracy": scores["accuracy"],
            }
        )
        mlflow.log_text(metrics["report"], "classification_report.txt")
        if REPORT_FILE.exists():
            mlflow.log_artifact(str(REPORT_FILE))
        mlflow.sklearn.log_model(model, name="model")
        uri = f"runs:/{active.info.run_id}/model"
    version = mlflow.register_model(uri, MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    log.info("registered %s v%s as @champion", MODEL_NAME, version)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
