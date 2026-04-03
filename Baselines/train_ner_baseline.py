import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class SentenceRecord:
    tokens: List[str]
    labels: List[str]
    rows: List[Tuple[str, Union[List[str], str]]]


def read_iob2_file(path: str, use_gold_labels: bool = True) -> List[SentenceRecord]:
    sentences: List[SentenceRecord] = []
    tokens: List[str] = []
    labels: List[str] = []
    rows: List[Tuple[str, Union[List[str], str]]] = []

    def flush_sentence() -> None:
        nonlocal tokens, labels, rows
        if tokens:
            sentences.append(SentenceRecord(tokens=tokens, labels=labels, rows=rows))
        tokens = []
        labels = []
        rows = []

    with open(path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped == "":
                flush_sentence()
                continue

            if stripped.startswith("#") and "\t" not in stripped:
                rows.append(("comment", line))
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            token = parts[1]
            if use_gold_labels and len(parts) >= 3:
                label = parts[2]
            else:
                label = "O"

            tokens.append(token)
            labels.append(label)
            rows.append(("token", parts))

    flush_sentence()
    return sentences


def build_label_list(*datasets: Sequence[SentenceRecord]) -> List[str]:
    labels = set()
    for data in datasets:
        for sent in data:
            labels.update(sent.labels)

    non_o = sorted(label for label in labels if label != "O")
    return ["O"] + non_o


def to_hf_dataset(data: Sequence[SentenceRecord], label_to_id: Dict[str, int]) -> Dataset:
    return Dataset.from_dict(
        {
            "tokens": [sent.tokens for sent in data],
            "ner_tags": [[label_to_id[tag] for tag in sent.labels] for sent in data],
        }
    )


def tokenize_and_align_labels(examples, tokenizer, max_length: int):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels = []
    for i, sentence_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(sentence_labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def compute_metrics(eval_preds, id_to_label: Dict[int, str]):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        gold_tags = []
        for pred_id, gold_id in zip(pred_seq, label_seq):
            if gold_id == -100:
                continue
            pred_tags.append(id_to_label[int(pred_id)])
            gold_tags.append(id_to_label[int(gold_id)])
        true_predictions.append(pred_tags)
        true_labels.append(gold_tags)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def decode_predictions(predictions: np.ndarray, label_ids: np.ndarray, id_to_label: Dict[int, str]) -> List[List[str]]:
    pred_ids = np.argmax(predictions, axis=2)
    decoded = []
    for pred_seq, label_seq in zip(pred_ids, label_ids):
        tags = []
        for pred_id, gold_id in zip(pred_seq, label_seq):
            if gold_id == -100:
                continue
            tags.append(id_to_label[int(pred_id)])
        decoded.append(tags)
    return decoded


def write_predictions_iob2(
    source_sentences: Sequence[SentenceRecord],
    predicted_labels: Sequence[Sequence[str]],
    output_path: str,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sent, pred in zip(source_sentences, predicted_labels):
            token_count = len(sent.tokens)
            pred_list = list(pred)

            if len(pred_list) < token_count:
                pred_list.extend(["O"] * (token_count - len(pred_list)))
            elif len(pred_list) > token_count:
                pred_list = pred_list[:token_count]

            token_idx = 0
            for row_type, row_value in sent.rows:
                if row_type == "comment":
                    outfile.write(str(row_value) + "\n")
                    continue

                parts = list(row_value)
                while len(parts) < 3:
                    parts.append("-")

                parts[2] = pred_list[token_idx]
                token_idx += 1
                outfile.write("\t".join(parts) + "\n")

            outfile.write("\n")


def parse_args(defaults: Optional[Dict[str, Union[str, int, float]]] = None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Monolingual BERT NER baseline for IOB2 datasets")
    parser.add_argument("--train_file", default=defaults.get("train_file", "Datasets/English/en_ewt-ud-train.iob2"))
    parser.add_argument("--dev_file", default=defaults.get("dev_file", "Datasets/English/en_ewt-ud-dev.iob2"))
    parser.add_argument("--test_file", default=defaults.get("test_file", "Datasets/English/en_ewt-ud-test-masked.iob2"))
    parser.add_argument("--model_name", default=defaults.get("model_name", "bert-base-cased"))
    parser.add_argument("--output_dir", default=defaults.get("output_dir", "outputs/english_bert_baseline"))
    parser.add_argument("--max_length", type=int, default=int(defaults.get("max_length", 256)))
    parser.add_argument("--learning_rate", type=float, default=float(defaults.get("learning_rate", 2e-5)))
    parser.add_argument("--train_batch_size", type=int, default=int(defaults.get("train_batch_size", 8)))
    parser.add_argument("--eval_batch_size", type=int, default=int(defaults.get("eval_batch_size", 16)))
    parser.add_argument("--gradient_accumulation_steps", type=int, default=int(defaults.get("gradient_accumulation_steps", 1)))
    parser.add_argument("--num_train_epochs", type=float, default=float(defaults.get("num_train_epochs", 4.0)))
    parser.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 0.01)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--label_report_path", default=defaults.get("label_report_path", ""))
    return parser.parse_args()


def main(defaults: Optional[Dict[str, Union[str, int, float]]] = None):
    args = parse_args(defaults)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_data = read_iob2_file(args.train_file, use_gold_labels=True)
    dev_data = read_iob2_file(args.dev_file, use_gold_labels=True)
    test_data = read_iob2_file(args.test_file, use_gold_labels=False)

    label_list = build_label_list(train_data, dev_data)
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    if args.label_report_path:
        report_dir = os.path.dirname(args.label_report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.label_report_path, "w", encoding="utf-8") as out:
            json.dump({"labels": label_list, "num_labels": len(label_list)}, out, indent=2)

    train_ds = to_hf_dataset(train_data, label_to_id)
    dev_ds = to_hf_dataset(dev_data, label_to_id)
    test_ds = to_hf_dataset(test_data, label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = train_ds.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, args.max_length),
        batched=True,
    )
    tokenized_dev = dev_ds.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, args.max_length),
        batched=True,
    )
    tokenized_test = test_ds.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, args.max_length),
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(preds, id_to_label),
    )

    trainer.train()
    dev_metrics = trainer.evaluate()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "dev_metrics.json"), "w", encoding="utf-8") as out_json:
        json.dump(dev_metrics, out_json, indent=2)

    dev_pred = trainer.predict(tokenized_dev)
    decoded_dev = decode_predictions(dev_pred.predictions, dev_pred.label_ids, id_to_label)
    dev_pred_path = os.path.join(args.output_dir, "dev_predictions.iob2")
    write_predictions_iob2(dev_data, decoded_dev, dev_pred_path)

    test_pred = trainer.predict(tokenized_test)
    decoded_test = decode_predictions(test_pred.predictions, test_pred.label_ids, id_to_label)
    test_pred_path = os.path.join(args.output_dir, "test_predictions.iob2")
    write_predictions_iob2(test_data, decoded_test, test_pred_path)

    print("Training complete.")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Dev metrics written to: {os.path.join(args.output_dir, 'dev_metrics.json')}")
    print(f"Dev predictions written to: {dev_pred_path}")
    print(f"Test predictions written to: {test_pred_path}")
    print(f"Evaluate dev file with: python span_f1.py {args.dev_file} <dev_predictions.iob2>")


if __name__ == "__main__":
    main()