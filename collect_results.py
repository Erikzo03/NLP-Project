"""
Run span_f1 on all dev and test predictions, collect strict/unlabeled/loose
metrics, and write results_summary_dev.csv and results_summary_test.csv.
"""
import os
import csv
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(BASE, "outputs")

# Gold files
DA_DEV  = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-dev.iob2")
DA_TEST = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-test.iob2")
NO_DEV  = os.path.join(BASE, "Datasets/Norwegian/nno_norne-ud-dev.iob2")
NO_TEST = os.path.join(BASE, "Datasets/Norwegian/nno_norne-ud-test.iob2")
EN_DEV  = os.path.join(BASE, "Datasets/English/en_ewt-ud-dev.iob2")

EXPERIMENTS = {
    "danish_bert_baseline":                 (DA_DEV,  DA_TEST),
    "danish_xlmr_baseline":                 (DA_DEV,  DA_TEST),
    "danish_xlmr_baseline_10pct":           (DA_DEV,  DA_TEST),
    "danish_xlmr_baseline_25pct":           (DA_DEV,  DA_TEST),
    "danish_xlmr_baseline_50pct":           (DA_DEV,  DA_TEST),
    "english_bert_baseline":                (EN_DEV,  None),
    "english_xlmr_baseline":                (EN_DEV,  None),
    "english_to_danish_zero_shot":          (DA_DEV,  DA_TEST),
    "english_to_danish_xlmr_zero_shot":     (DA_DEV,  DA_TEST),
    "english_to_danish_mbert_finetuned_danish":  (DA_DEV, DA_TEST),
    "english_to_danish_xlmr_finetuned_danish":   (DA_DEV, DA_TEST),
    "norwegian_bert_baseline":              (NO_DEV,  NO_TEST),
    "norwegian_xlmr_baseline":              (NO_DEV,  NO_TEST),
    "norwegian_to_danish_mbert_zero_shot":  (DA_DEV,  DA_TEST),
    "norwegian_to_danish_mbert_finetuned_danish": (DA_DEV, DA_TEST),
    "norwegian_to_danish_xlmr_zero_shot":   (DA_DEV,  DA_TEST),
    "norwegian_to_danish_xlmr_finetuned_danish":  (DA_DEV, DA_TEST),
}


def read_iob2(path):
    annotations, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line == "":
            if cur:
                annotations.append(cur)
            cur = []
        elif line.startswith("#") and len(line.split("\t")) == 1:
            continue
        else:
            parts = line.split("\t")
            if len(parts) > 2:
                cur.append(parts[2])
    if cur:
        annotations.append(cur)
    return annotations


def to_spans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == "B":
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[end][0] != "I":
                    break
            spans.add(f"{beg}-{end}:{tags[beg][2:]}")
    return spans


def get_beg_end(span):
    return [int(x) for x in span.split(":")[0].split("-")]


def loose_overlap(spans1, spans2):
    found = 0
    for span in spans1:
        beg, end = get_beg_end(span)
        label = span.split(":")[1]
        match = False
        for span2 in spans2:
            b2, e2 = get_beg_end(span2)
            l2 = span2.split(":")[1]
            if label == l2:
                if b2 >= beg and b2 <= end:
                    match = True
                if e2 <= end and e2 >= beg:
                    match = True
        if match:
            found += 1
    return found


def unlabeled_overlap(spans1, spans2):
    s1 = set(x.split(":")[0] for x in spans1)
    s2 = set(x.split(":")[0] for x in spans2)
    return len(s1 & s2)


def compute_metrics(gold_path, pred_path):
    gold_ners = read_iob2(gold_path)
    pred_ners = read_iob2(pred_path)

    if len(gold_ners) != len(pred_ners):
        print(f"  WARNING: sentence count mismatch {len(gold_ners)} vs {len(pred_ners)}")
        n = min(len(gold_ners), len(pred_ners))
        gold_ners = gold_ners[:n]
        pred_ners = pred_ners[:n]

    tp = fp = fn = 0
    tp_ul = fp_ul = fn_ul = 0
    rl_tp = rl_fn = pl_tp = pl_fp = 0

    for gold, pred in zip(gold_ners, pred_ners):
        gs = to_spans(gold)
        ps = to_spans(pred)

        ov = len(gs & ps)
        tp += ov; fp += len(ps) - ov; fn += len(gs) - ov

        ov_ul = unlabeled_overlap(gs, ps)
        tp_ul += ov_ul; fp_ul += len(ps) - ov_ul; fn_ul += len(gs) - ov_ul

        ov_rl = loose_overlap(gs, ps)
        rl_tp += ov_rl; rl_fn += len(gs) - ov_rl
        ov_pl = loose_overlap(ps, gs)
        pl_tp += ov_pl; pl_fp += len(ps) - ov_pl

    def f1(p, r):
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec  = 0.0 if tp + fn == 0 else tp / (tp + fn)
    strict_f1 = f1(prec, rec)

    prec_ul = 0.0 if tp_ul + fp_ul == 0 else tp_ul / (tp_ul + fp_ul)
    rec_ul  = 0.0 if tp_ul + fn_ul == 0 else tp_ul / (tp_ul + fn_ul)
    ul_f1 = f1(prec_ul, rec_ul)

    prec_l = 0.0 if pl_tp + pl_fp == 0 else pl_tp / (pl_tp + pl_fp)
    rec_l  = 0.0 if rl_tp + rl_fn == 0 else rl_tp / (rl_tp + rl_fn)
    loose_f1 = f1(prec_l, rec_l)

    return {
        "strict_precision": prec,
        "strict_recall": rec,
        "strict_f1": strict_f1,
        "unlabeled_f1": ul_f1,
        "loose_f1": loose_f1,
    }


FIELDS = ["experiment", "strict_precision", "strict_recall", "strict_f1", "unlabeled_f1", "loose_f1"]


def collect(split):
    rows = []
    for exp, (dev_gold, test_gold) in EXPERIMENTS.items():
        gold = dev_gold if split == "dev" else test_gold
        if gold is None:
            print(f"[{split}] {exp}: no gold file, skipping")
            continue
        pred_file = os.path.join(OUTPUTS, exp, f"{split}_predictions.iob2")
        if not os.path.exists(pred_file):
            print(f"[{split}] {exp}: predictions file missing, skipping")
            continue
        print(f"[{split}] {exp} ...", end=" ", flush=True)
        try:
            m = compute_metrics(gold, pred_file)
            m["experiment"] = exp
            rows.append(m)
            print(f"strict={m['strict_f1']:.4f} ul={m['unlabeled_f1']:.4f} loose={m['loose_f1']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    print(f"Written: {path}")


if __name__ == "__main__":
    dev_rows  = collect("dev")
    test_rows = collect("test")

    latex_dir = os.path.join(BASE, "Latex")
    write_csv(dev_rows,  os.path.join(latex_dir, "results_span_f1_dev.csv"))
    write_csv(test_rows, os.path.join(latex_dir, "results_span_f1_test.csv"))
    print("\nDone.")
