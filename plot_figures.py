"""Regenerate improved Figure 2 (learning curve) and Figure 3 (scatter plot)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

BASE     = os.path.dirname(os.path.abspath(__file__))
DEV_CSV  = os.path.join(BASE, "Latex/results_span_f1_dev.csv")
FIG_DIR  = os.path.join(BASE, "Latex/figures")

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["experiment"]: r for r in csv.DictReader(f)}

dev = load_csv(DEV_CSV)


# ── Figure 2: Danish learning curve with proper zero-shot reference lines ──────

def make_learning_curve():
    pct_keys = [
        ("danish_xlmr_baseline_10pct", 10),
        ("danish_xlmr_baseline_25pct", 25),
        ("danish_xlmr_baseline_50pct", 50),
        ("danish_xlmr_baseline",       100),
    ]
    xs = [p for _, p in pct_keys]
    ys = [float(dev[k]["strict_f1"]) * 100 for k, _ in pct_keys]

    en_zero = float(dev["english_to_danish_xlmr_zero_shot"]["strict_f1"]) * 100
    no_zero = float(dev["norwegian_to_danish_xlmr_zero_shot"]["strict_f1"]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2, markersize=7,
            label="Danish XLM-R baseline", zorder=3)

    ax.axhline(en_zero, linestyle="--", color="#ff7f0e", linewidth=1.8,
               label=f"EN→DA XLM-R zero-shot ({en_zero:.1f})", zorder=2)
    ax.axhline(no_zero, linestyle="--", color="#2ca02c", linewidth=1.8,
               label=f"NO→DA XLM-R zero-shot ({no_zero:.1f})", zorder=2)

    ax.set_xlabel("Danish supervision budget (%)", fontsize=12)
    ax.set_ylabel("Strict span F1 (%)", fontsize=12)
    ax.set_title("Danish learning curve and zero-shot transfer", fontsize=13)
    ax.set_xlim(-2, 105)
    ax.set_xticks([10, 25, 50, 100])
    ax.set_ylim(65, 92)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.35)

    out = os.path.join(FIG_DIR, "danish_learning_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Scatter with non-overlapping labels ─────────────────────────────

LABEL_MAP = {
    "danish_bert_baseline":                    "DA BERT",
    "danish_xlmr_baseline":                    "DA XLM-R",
    "danish_xlmr_baseline_10pct":              "DA XLM-R 10%",
    "danish_xlmr_baseline_25pct":              "DA XLM-R 25%",
    "danish_xlmr_baseline_50pct":              "DA XLM-R 50%",
    "english_bert_baseline":                   "EN BERT",
    "english_xlmr_baseline":                   "EN XLM-R",
    "english_to_danish_zero_shot":             "EN→DA 0-shot (artifact)",
    "english_to_danish_xlmr_zero_shot":        "EN→DA XLM-R 0-shot",
    "english_to_danish_mbert_finetuned_danish":"EN→DA mBERT FT",
    "english_to_danish_xlmr_finetuned_danish": "EN→DA XLM-R FT",
    "norwegian_bert_baseline":                 "NO BERT",
    "norwegian_xlmr_baseline":                 "NO XLM-R",
    "norwegian_to_danish_mbert_zero_shot":     "NO→DA mBERT 0-shot",
    "norwegian_to_danish_mbert_finetuned_danish":"NO→DA mBERT FT",
    "norwegian_to_danish_xlmr_zero_shot":      "NO→DA XLM-R 0-shot",
    "norwegian_to_danish_xlmr_finetuned_danish":"NO→DA XLM-R FT",
}

# Manual label offsets (dx, dy) in data units to avoid overlap
OFFSETS = {
    # bottom-left isolated points
    "DA XLM-R 10%":           ( 0.5, -1.5),
    "EN BERT":                ( 0.5, -1.5),
    "EN XLM-R":               ( 0.5,  1.0),
    "EN→DA 0-shot (artifact)":(-1.5,  1.5),
    # zero-shot cluster (x~83-87)
    "EN→DA XLM-R 0-shot":    (-9.0, -2.5),
    "NO→DA mBERT 0-shot":    (-9.5,  1.5),
    "NO→DA XLM-R 0-shot":    (-9.5, -4.5),
    "DA XLM-R 25%":           (-9.5,  4.5),
    "DA XLM-R 50%":           ( 0.5, -2.5),
    # dense top-right cluster (x~88-93)
    "DA BERT":                (-8.0, -4.5),
    "DA XLM-R":               (-8.0, -2.0),
    "EN→DA mBERT FT":         ( 0.5,  3.5),
    "EN→DA XLM-R FT":         ( 0.5, -2.5),
    "NO→DA mBERT FT":         ( 0.5,  1.0),
    "NO→DA XLM-R FT":         (-8.5,  2.5),
    "NO BERT":                (-8.0,  4.5),
    "NO XLM-R":               ( 0.5,  1.5),
}

# Load eval_f1 from the original results_summary.csv which has eval scores
OLD_CSV = os.path.join(BASE, "Latex/results_summary.csv")
with open(OLD_CSV, newline="", encoding="utf-8") as f:
    old = {r["experiment"]: r for r in csv.DictReader(f)}


def make_scatter():
    points = []
    for exp, label in LABEL_MAP.items():
        if exp not in dev or exp not in old:
            continue
        x = float(dev[exp]["strict_f1"]) * 100
        y = float(old[exp]["eval_f1"]) * 100
        points.append((x, y, label))

    fig, ax = plt.subplots(figsize=(8, 7))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.scatter(xs, ys, s=60, color="#1f77b4", zorder=3)

    # Diagonal reference line
    lims = [0, 100]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, alpha=0.7, zorder=1)

    for x, y, label in points:
        dx, dy = OFFSETS.get(label, (0.3, 0.5))
        ax.annotate(
            label, (x, y),
            xytext=(x + dx, y + dy),
            fontsize=7.5,
            ha="left" if dx >= 0 else "right",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
            if abs(dx) > 1 or abs(dy) > 0.5 else None,
        )

    ax.set_xlabel("Strict span F1 (%)", fontsize=12)
    ax.set_ylabel("Saved evaluation F1 (%)", fontsize=12)
    ax.set_title("Strict F1 vs saved evaluation F1", fontsize=13)
    ax.set_xlim(-5, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIG_DIR, "strict_vs_eval_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Also regenerate Figure 1 with cleaner labels ─────────────────────────────

def make_bar_chart():
    order = [
        ("english_bert_baseline",                 "EN BERT",          "#4878cf"),
        ("english_xlmr_baseline",                 "EN XLM-R",         "#4878cf"),
        ("norwegian_bert_baseline",               "NO BERT",          "#6acc65"),
        ("norwegian_xlmr_baseline",               "NO XLM-R",         "#6acc65"),
        ("danish_bert_baseline",                  "DA BERT",          "#d65f5f"),
        ("danish_xlmr_baseline",                  "DA XLM-R",         "#d65f5f"),
        ("english_to_danish_xlmr_zero_shot",      "EN→DA XLM-R 0-shot","#b47cc7"),
        ("norwegian_to_danish_mbert_zero_shot",   "NO→DA mBERT 0-shot","#b47cc7"),
        ("norwegian_to_danish_xlmr_zero_shot",    "NO→DA XLM-R 0-shot","#b47cc7"),
        ("english_to_danish_mbert_finetuned_danish","EN→DA mBERT FT", "#c4ad66"),
        ("english_to_danish_xlmr_finetuned_danish", "EN→DA XLM-R FT", "#c4ad66"),
        ("norwegian_to_danish_mbert_finetuned_danish","NO→DA mBERT FT","#c4ad66"),
        ("norwegian_to_danish_xlmr_finetuned_danish", "NO→DA XLM-R FT","#c4ad66"),
        ("danish_xlmr_baseline_10pct",            "DA XLM-R 10%",     "#77bedb"),
        ("danish_xlmr_baseline_25pct",            "DA XLM-R 25%",     "#77bedb"),
        ("danish_xlmr_baseline_50pct",            "DA XLM-R 50%",     "#77bedb"),
    ]
    labels = [l for _, l, _ in order]
    vals   = [float(dev[k]["strict_f1"]) * 100 for k, _, _ in order if k in dev]
    colors = [c for _, _, c in order if _[0] in dev or True]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Strict span F1 (%)", fontsize=11)
    ax.set_title("Strict span F1 across all evaluated runs", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "strict_f1_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_learning_curve()
    make_scatter()
    make_bar_chart()
    print("All figures saved.")
