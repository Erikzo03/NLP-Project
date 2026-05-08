"""
Multi-seed analysis for EN vs NO zero-shot transfer.
Computes span F1 for all 3 seeds, reports mean ± std,
and runs a paired t-test on the NO vs EN comparison.
"""
import os, csv, math
from collections import defaultdict

BASE    = os.path.dirname(os.path.abspath(__file__))
DA_DEV  = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-dev.iob2")
DA_TEST = os.path.join(BASE, "Datasets/Danish/da_ddt-ud-test.iob2")

# seed_label -> (NO_XLM-R dir, EN_XLM-R dir, NO_mBERT dir, EN_mBERT dir)
SEEDS = {
    "seed42": (
        "norwegian_to_danish_xlmr_zero_shot",
        "english_to_danish_xlmr_zero_shot",
        "norwegian_to_danish_mbert_zero_shot",
        "english_to_danish_mbert_zero_shot",   # note: dev artifact; test used
    ),
    "seed1": (
        "norwegian_to_danish_xlmr_zero_shot_1seed",
        "english_to_danish_xlmr_zero_shot_seed1",
        "norwegian_to_danish_mbert_zero_shot_1seed",
        "english_to_danish_mbert_zero_shot_1seed",
    ),
    "seed2": (
        "norwegian_to_danish_xlmr_zero_shot_2seed",
        "english_to_danish_xlmr_zero_shot_2_seed",
        "norwegian_to_danish_mbert_zero_shot_2seed",
        "english_to_danish_mbert_zero_shot_2seed",
    ),
}

LABELS = ["NO XLM-R", "EN XLM-R", "NO mBERT", "EN mBERT"]


# ── span F1 helpers ───────────────────────────────────────────────────────────

def read_iob2(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
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
            while j + 1 < len(tags) and tags[j+1].startswith("I-"):
                j += 1
            spans.add(f"{i}-{j}:{t[2:]}")
    return spans

def strict_f1(gold_sents, pred_sents):
    tp = fp = fn = 0
    n = min(len(gold_sents), len(pred_sents))
    for g, p in zip(gold_sents[:n], pred_sents[:n]):
        gs, ps = to_spans(g), to_spans(p)
        ov = len(gs & ps)
        tp += ov; fp += len(ps) - ov; fn += len(gs) - ov
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0


# ── collect F1 scores ─────────────────────────────────────────────────────────

def collect_scores(gold_path, split):
    gold = read_iob2(gold_path)
    # scores[label][seed] = f1
    scores = defaultdict(dict)
    for seed_lbl, dirs in SEEDS.items():
        for label, dirname in zip(LABELS, dirs):
            # Skip EN mBERT dev (known artifact for seed42)
            if label == "EN mBERT" and split == "dev" and seed_lbl == "seed42":
                continue
            pred_path = os.path.join(BASE, "outputs", dirname,
                                     f"{split}_predictions.iob2")
            if not os.path.exists(pred_path):
                print(f"  MISSING: {dirname}/{split}_predictions.iob2")
                continue
            pred = read_iob2(pred_path)
            if abs(len(pred) - len(gold)) > 5:
                print(f"  MISMATCH ({len(pred)} vs {len(gold)}): {dirname} [{split}]")
                continue
            f1 = strict_f1(gold, pred)
            scores[label][seed_lbl] = f1 * 100
    return scores


# ── statistics ────────────────────────────────────────────────────────────────

def mean_std(vals):
    n = len(vals)
    if n == 0: return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1: return m, 0.0
    s = math.sqrt(sum((x-m)**2 for x in vals) / (n-1))
    return m, s

def paired_ttest(a_vals, b_vals):
    """Paired two-tailed t-test. Returns t, p (two-tailed), df."""
    pairs = [(a, b) for a, b in zip(a_vals, b_vals)]
    n = len(pairs)
    if n < 2: return float("nan"), float("nan"), 0
    diffs = [a - b for a, b in pairs]
    md = sum(diffs) / n
    sd = math.sqrt(sum((d - md)**2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float("nan")
    df = n - 1
    # Two-tailed p-value via t-distribution approximation (Abramowitz & Stegun)
    p = _t_pvalue(abs(t), df) * 2
    return t, p, df

def _t_pvalue(t, df):
    """One-tailed p-value P(T > t) for t-distribution, df degrees of freedom."""
    # Use incomplete beta function approximation
    x = df / (df + t * t)
    return 0.5 * _ibeta(df/2, 0.5, x)

def _ibeta(a, b, x, max_iter=200):
    """Regularised incomplete beta I_x(a,b) via continued fraction."""
    if x < 0 or x > 1: return float("nan")
    if x == 0: return 0.0
    if x == 1: return 1.0
    lbeta = _lbeta(a, b)
    front = math.exp(math.log(x)*a + math.log(1-x)*b - lbeta) / a
    # Lentz continued fraction
    def cf():
        tiny = 1e-30
        f = tiny; C = f; D = 0.0
        for m in range(max_iter):
            for which in (0, 1):
                if which == 0:
                    num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)) if m else 1.0
                else:
                    num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
                D = 1 + num * D; D = tiny if abs(D) < tiny else D; D = 1/D
                C = 1 + num / C; C = tiny if abs(C) < tiny else C
                f *= C * D
                if abs(C*D - 1) < 1e-10: return f
        return f
    return front * cf()

def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── reporting ─────────────────────────────────────────────────────────────────

def report(split, gold_path):
    print(f"\n{'='*65}")
    print(f"  Multi-seed strict F1 — {split} set")
    print(f"{'='*65}")
    scores = collect_scores(gold_path, split)

    # Per-system summary
    print(f"\n{'System':<20} {'S42':>6} {'S1':>6} {'S2':>6}  {'Mean':>6}  {'Std':>5}")
    print("-" * 55)
    summaries = {}
    for label in LABELS:
        if label not in scores: continue
        vals_dict = scores[label]
        s42 = vals_dict.get("seed42", float("nan"))
        s1  = vals_dict.get("seed1",  float("nan"))
        s2  = vals_dict.get("seed2",  float("nan"))
        vals = [v for v in [s42, s1, s2] if not math.isnan(v)]
        m, s = mean_std(vals)
        summaries[label] = {"vals": vals, "mean": m, "std": s}
        def fmt(v): return f"{v:6.2f}" if not math.isnan(v) else "   n/a"
        print(f"{label:<20} {fmt(s42)} {fmt(s1)} {fmt(s2)}  {m:6.2f}  {s:5.2f}")

    # Paired t-tests: NO vs EN within each encoder
    print(f"\n--- Paired t-test (NO vs EN) ---")
    for enc in ["XLM-R", "mBERT"]:
        no_lbl = f"NO {enc}"; en_lbl = f"EN {enc}"
        if no_lbl not in summaries or en_lbl not in summaries: continue
        no_vals = summaries[no_lbl]["vals"]
        en_vals = summaries[en_lbl]["vals"]
        n = min(len(no_vals), len(en_vals))
        if n < 2:
            print(f"  {enc}: insufficient data (n={n})")
            continue
        t, p, df = paired_ttest(no_vals[:n], en_vals[:n])
        delta = summaries[no_lbl]["mean"] - summaries[en_lbl]["mean"]
        stars = sig_stars(p)
        print(f"  {enc}: NO={summaries[no_lbl]['mean']:.2f}±{summaries[no_lbl]['std']:.2f}  "
              f"EN={summaries[en_lbl]['mean']:.2f}±{summaries[en_lbl]['std']:.2f}  "
              f"Δ={delta:+.2f}  t({df})={t:.3f}  p={p:.4f} {stars}")
    return summaries


def save_csv(dev_summaries, test_summaries):
    rows = []
    for label in LABELS:
        row = {"system": label}
        for split, summ in [("dev", dev_summaries), ("test", test_summaries)]:
            if label in summ:
                row[f"{split}_mean"] = round(summ[label]["mean"], 2)
                row[f"{split}_std"]  = round(summ[label]["std"],  2)
                for i, (sk, sv) in enumerate(
                        zip(["seed42","seed1","seed2"],
                            [summ[label]["vals"][j] if j < len(summ[label]["vals"])
                             else float("nan") for j in range(3)])):
                    row[f"{split}_{sk}"] = round(sv, 2) if not math.isnan(sv) else ""
        rows.append(row)
    out = os.path.join(BASE, "Latex", "multiseed_results.csv")
    fields = ["system",
              "dev_seed42","dev_seed1","dev_seed2","dev_mean","dev_std",
              "test_seed42","test_seed1","test_seed2","test_mean","test_std"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    dev_summ  = report("dev",  DA_DEV)
    test_summ = report("test", DA_TEST)
    save_csv(dev_summ, test_summ)
