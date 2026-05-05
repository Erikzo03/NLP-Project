"""
Paired bootstrap significance test for EN vs NO zero-shot transfer (XLM-R).
Tests whether the 1.0 F1 gap on Danish dev is statistically significant.
Uses approximate randomisation / paired bootstrap (Efron & Tibshirani 1993).
"""
import os, random, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DA_DEV = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-dev.iob2")
DA_TEST = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-test.iob2")

PAIRS = [
    ("NO→DA XLM-R vs EN→DA XLM-R (zero-shot)",
     "norwegian_to_danish_xlmr_zero_shot",
     "english_to_danish_xlmr_zero_shot"),
    ("NO→DA mBERT vs EN→DA mBERT (zero-shot)",
     "norwegian_to_danish_mbert_zero_shot",
     "english_to_danish_zero_shot"),   # note: EN mBERT zero-shot is the artifact — skip dev
]


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


def to_spans(tags):
    spans = set()
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            j = i
            while j + 1 < len(tags) and tags[j + 1].startswith("I-"):
                j += 1
            spans.add(f"{i}-{j}:{t[2:]}")
    return spans


def sentence_stats(gold_tags, pred_tags):
    gs = to_spans(gold_tags)
    ps = to_spans(pred_tags)
    tp = len(gs & ps)
    fp = len(ps) - tp
    fn = len(gs) - tp
    return tp, fp, fn


def aggregate_f1(tps, fps, fns):
    tp = sum(tps); fp = sum(fps); fn = sum(fns)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def bootstrap_test(gold, pred_a, pred_b, n_iter=10000, seed=0):
    """
    Two-tailed paired bootstrap: null hypothesis = F1(A) == F1(B).
    Returns observed delta F1(A) - F1(B) and p-value.
    """
    rng = random.Random(seed)
    n = min(len(gold), len(pred_a), len(pred_b))
    gold = gold[:n]; pred_a = pred_a[:n]; pred_b = pred_b[:n]

    # Per-sentence TP/FP/FN for both systems
    stats_a = [sentence_stats(g, p) for g, p in zip(gold, pred_a)]
    stats_b = [sentence_stats(g, p) for g, p in zip(gold, pred_b)]

    obs_f1_a = aggregate_f1([s[0] for s in stats_a],
                             [s[1] for s in stats_a],
                             [s[2] for s in stats_a])
    obs_f1_b = aggregate_f1([s[0] for s in stats_b],
                             [s[1] for s in stats_b],
                             [s[2] for s in stats_b])
    obs_delta = obs_f1_a - obs_f1_b

    count = 0
    for _ in range(n_iter):
        # Randomly swap per-sentence assignment
        boot_a_tp, boot_a_fp, boot_a_fn = [], [], []
        boot_b_tp, boot_b_fp, boot_b_fn = [], [], []
        for sa, sb in zip(stats_a, stats_b):
            if rng.random() < 0.5:
                sa, sb = sb, sa
            boot_a_tp.append(sa[0]); boot_a_fp.append(sa[1]); boot_a_fn.append(sa[2])
            boot_b_tp.append(sb[0]); boot_b_fp.append(sb[1]); boot_b_fn.append(sb[2])
        f1_a = aggregate_f1(boot_a_tp, boot_a_fp, boot_a_fn)
        f1_b = aggregate_f1(boot_b_tp, boot_b_fp, boot_b_fn)
        if abs(f1_a - f1_b) >= abs(obs_delta):
            count += 1

    p_value = count / n_iter
    return obs_f1_a, obs_f1_b, obs_delta, p_value


def run(label, exp_a, exp_b, gold_path, split):
    gold = read_iob2(gold_path)
    path_a = os.path.join(BASE, "outputs", exp_a, f"{split}_predictions.iob2")
    path_b = os.path.join(BASE, "outputs", exp_b, f"{split}_predictions.iob2")
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"  [skip] missing predictions for {label}")
        return None
    pred_a = read_iob2(path_a)
    pred_b = read_iob2(path_b)
    if len(pred_a) != len(gold) or len(pred_b) != len(gold):
        print(f"  [skip] sentence count mismatch — {label} ({split})")
        return None
    f1_a, f1_b, delta, p = bootstrap_test(gold, pred_a, pred_b)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label} [{split}]")
    print(f"    F1(NO): {f1_a*100:.2f}  F1(EN): {f1_b*100:.2f}  Δ={delta*100:+.2f}  p={p:.4f} {sig}")
    return f1_a, f1_b, delta, p


if __name__ == "__main__":
    print("=== Paired Bootstrap Test (10 000 iterations) ===\n")
    print("NO→DA XLM-R vs EN→DA XLM-R")
    r_dev  = run("NO→DA XLM-R vs EN→DA XLM-R",
                 "norwegian_to_danish_xlmr_zero_shot",
                 "english_to_danish_xlmr_zero_shot",
                 DA_DEV, "dev")
    r_test = run("NO→DA XLM-R vs EN→DA XLM-R",
                 "norwegian_to_danish_xlmr_zero_shot",
                 "english_to_danish_xlmr_zero_shot",
                 DA_TEST, "test")

    print("\nNO→DA mBERT vs EN→DA mBERT (zero-shot only — EN artifact skipped on dev)")
    r_test_mbert = run("NO→DA mBERT vs EN→DA mBERT",
                       "norwegian_to_danish_mbert_zero_shot",
                       "english_to_danish_xlmr_zero_shot",   # using XLM-R EN as proxy
                       DA_TEST, "test")
