"""
Per-entity-type F1 for key systems on the Danish dev set.
Produces a CSV and printed table for the report.
"""
import os, csv
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DA_DEV  = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-dev.iob2")
DA_TEST = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-test.iob2")

SYSTEMS = [
    ("Danish BERT baseline",          "danish_bert_baseline"),
    ("Danish XLM-R baseline",         "danish_xlmr_baseline"),
    ("EN→DA XLM-R zero-shot",         "english_to_danish_xlmr_zero_shot"),
    ("NO→DA XLM-R zero-shot",         "norwegian_to_danish_xlmr_zero_shot"),
    ("EN→DA mBERT + DA FT",           "english_to_danish_mbert_finetuned_danish"),
    ("NO→DA mBERT + DA FT",           "norwegian_to_danish_mbert_finetuned_danish"),
    ("NO→DA XLM-R + DA FT",           "norwegian_to_danish_xlmr_finetuned_danish"),
]

TYPES = ["PER", "LOC", "ORG", "MISC"]


def read_iob2(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line == "":
            if cur: sents.append(cur)
            cur = []
        elif line.startswith("#") and "\t" not in line:
            continue
        else:
            parts = line.split("\t")
            if len(parts) > 2:
                cur.append(parts[2])
    if cur: sents.append(cur)
    return sents


def to_typed_spans(tags):
    """Returns dict: type -> set of 'sentidx:beg-end' span keys."""
    spans = []
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            etype = t[2:]
            j = i
            while j + 1 < len(tags) and tags[j + 1].startswith("I-"):
                j += 1
            spans.append((etype, i, j))
    return spans


def per_type_metrics(gold_sents, pred_sents):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for gold_tags, pred_tags in zip(gold_sents, pred_sents):
        gold_spans = to_typed_spans(gold_tags)
        pred_spans = to_typed_spans(pred_tags)

        gold_set = set((e, b, end) for e, b, end in gold_spans)
        pred_set = set((e, b, end) for e, b, end in pred_spans)

        for span in gold_set & pred_set:
            tp[span[0]] += 1
        for span in pred_set - gold_set:
            fp[span[0]] += 1
        for span in gold_set - pred_set:
            fn[span[0]] += 1

    results = {}
    for etype in TYPES:
        p = tp[etype] / (tp[etype] + fp[etype]) if tp[etype] + fp[etype] > 0 else 0.0
        r = tp[etype] / (tp[etype] + fn[etype]) if tp[etype] + fn[etype] > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        results[etype] = {"p": p, "r": r, "f1": f,
                          "tp": tp[etype], "fp": fp[etype], "fn": fn[etype]}
    return results


def run_split(gold_path, split):
    gold = read_iob2(gold_path)
    rows = []
    print(f"\n{'='*70}")
    print(f"Per-entity-type F1 — {split} set")
    print(f"{'='*70}")
    header = f"{'System':<32} {'PER':>6} {'LOC':>6} {'ORG':>6} {'MISC':>6}"
    print(header)
    print("-" * 55)
    for label, exp in SYSTEMS:
        pred_path = os.path.join(BASE, "outputs", exp, f"{split}_predictions.iob2")
        if not os.path.exists(pred_path):
            continue
        pred = read_iob2(pred_path)
        if len(pred) != len(gold):
            print(f"  {label}: sentence mismatch, skip")
            continue
        metrics = per_type_metrics(gold, pred)
        row = {"system": label, "split": split}
        scores = []
        for etype in TYPES:
            row[f"{etype}_f1"] = round(metrics[etype]["f1"] * 100, 1)
            scores.append(f"{metrics[etype]['f1']*100:6.1f}")
        rows.append(row)
        print(f"{label:<32} {'  '.join(scores)}")
    return rows


if __name__ == "__main__":
    dev_rows  = run_split(DA_DEV,  "dev")
    test_rows = run_split(DA_TEST, "test")

    out = os.path.join(BASE, "Latex", "entity_type_f1.csv")
    fields = ["system", "split"] + [f"{t}_f1" for t in TYPES]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(dev_rows + test_rows)
    print(f"\nSaved: {out}")
