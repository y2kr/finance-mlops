from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.transformers
import torch
import torch.nn.functional as F
import transformers
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from finance_mlops.baseline import DATA_DIR, MODEL_NAME, TRACKING_URI, evaluate, train_eval_split

log = logging.getLogger("finetune")

BASE_MODEL = "yiyanghkust/finbert-pretrain"
REVISION = "88ab954a39ea6d3ce2b62cff086dd5ad1172c664"
MAX_LEN = 128
SEED = 42
GATE_MACRO = 0.692
GATE_BEARISH = 0.525
LABELS = ("bearish", "bullish", "neutral")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


class HeadlineDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = [LABEL2ID[label] for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def class_weights(labels: list[str]) -> torch.Tensor:
    counts = [labels.count(label) for label in LABELS]
    total = len(labels)
    return torch.tensor([total / (len(LABELS) * c) for c in counts], dtype=torch.float)


def loss_fn(weights: torch.Tensor):
    def compute(outputs, labels, num_items_in_batch=None):
        return F.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))

    return compute


def metrics_fn(eval_pred) -> dict:
    preds = eval_pred.predictions.argmax(-1)
    y_pred = [LABELS[i] for i in preds]
    y_true = [LABELS[i] for i in eval_pred.label_ids]
    scores = evaluate(y_true, y_pred)
    return {"macro_f1": scores["macro_f1"], "bearish_f1": scores["bearish_f1"]}


def run(data_dir: Path = DATA_DIR) -> dict:
    assert torch.cuda.is_available(), "no CUDA device — refusing to fine-tune on CPU"
    set_seed(SEED)
    train_rows, eval_rows = train_eval_split(data_dir)
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        [t for t, _ in train_rows],
        [y for _, y in train_rows],
        test_size=0.1,
        stratify=[y for _, y in train_rows],
        random_state=SEED,
    )
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    train_ds = HeadlineDataset(tr_texts, tr_labels, tokenizer)
    val_ds = HeadlineDataset(va_texts, va_labels, tokenizer)
    eval_ds = HeadlineDataset([t for t, _ in eval_rows], [y for _, y in eval_rows], tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=REVISION,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    weights = class_weights(tr_labels)

    with tempfile.TemporaryDirectory() as out_dir:
        args = TrainingArguments(
            output_dir=out_dir,
            bf16=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            num_train_epochs=4,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=SEED,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_loss_func=loss_fn(weights),
            compute_metrics=metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        preds = trainer.predict(eval_ds).predictions.argmax(-1)

    y_pred = [LABELS[i] for i in preds]
    y_true = [y for _, y in eval_rows]
    metrics = evaluate(y_true, y_pred)
    log.info("\n%s", metrics["report"])
    log.info("AllAgree macro-F1 %.3f bearish-F1 %.3f", metrics["macro_f1"], metrics["bearish_f1"])

    track(model, tokenizer, tr_labels, va_labels, eval_rows, weights, metrics)
    gate(metrics)
    return metrics


def track(model, tokenizer, tr_labels, va_labels, eval_rows, weights, metrics) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name="finetune-finbert") as active:
        mlflow.log_params(
            {
                "base_model": BASE_MODEL,
                "revision": REVISION,
                "max_seq_len": MAX_LEN,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "epochs": 4,
                "seed": SEED,
                "class_weights": dict(zip(LABELS, weights.tolist(), strict=True)),
                "n_train": len(tr_labels),
                "n_val": len(va_labels),
                "n_eval": len(eval_rows),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
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
        mlflow.transformers.log_model(
            {"model": model, "tokenizer": tokenizer},
            name="model",
            task="text-classification",
            pip_requirements=["torch", "transformers"],
        )
        uri = f"runs:/{active.info.run_id}/model"
    version = mlflow.register_model(uri, MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger", version)
    log.info("registered %s v%s as @challenger", MODEL_NAME, version)


def gate(metrics: dict) -> bool:
    passed = metrics["macro_f1"] >= GATE_MACRO and metrics["bearish_f1"] >= GATE_BEARISH
    log.info(
        "GATE %s — macro-F1 %.3f (>=%.3f), bearish-F1 %.3f (>=%.3f)",
        "PASS" if passed else "FAIL",
        metrics["macro_f1"],
        GATE_MACRO,
        metrics["bearish_f1"],
        GATE_BEARISH,
    )
    if passed:
        log.info(
            "promote by hand: MlflowClient().set_registered_model_alias(%r, 'champion', <v>)",
            MODEL_NAME,
        )
    return passed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run()


if __name__ == "__main__":
    main()
