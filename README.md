# Cross-Lingual NER Transfer to Danish: English vs Norwegian

**Repository:** [github.com/Erikzo03/NLP-Project](https://github.com/Erikzo03/NLP-Project)

This repository contains all code, data, results, and the LaTeX report for the project *"Cross-Lingual NER Transfer to Danish: English vs Norwegian with Linguistic Distance Analysis"*, submitted as the final project for the NLP and Deep Learning course at ITU Copenhagen.

---

## Important: HPC Required

Training the transformer models (mBERT, XLM-R) is **computationally heavy** and was run on a High-Performance Computing (HPC) cluster using SLURM. Running the training scripts on a standard laptop or desktop is not recommended and will be very slow.

To reproduce the training results you will need:

- Access to an HPC cluster with SLURM job scheduling
- At least one GPU (the experiments used a single GPU per job)
- Your own SLURM job files (see the [Job File Template](#job-file-template) section below)

The analysis scripts (`collect_results.py`, `plot_figures.py`, etc.) are lightweight and can be run locally once the training outputs exist.

---

## Project Structure

```text
NLP-Project/
├── Baselines/
│   ├── train_ner_baseline.py              # Core shared training logic
│   ├── train_english_baseline.py          # English mBERT monolingual
│   ├── train_english_xlmr_baseline.py     # English XLM-R monolingual
│   ├── train_english_bert_baseline.py     # English BERT (single-language)
│   ├── train_danish_baseline.py           # Danish mBERT monolingual
│   ├── train_danish_xlmr_baseline.py      # Danish XLM-R monolingual
│   ├── train_norwegian_baseline.py        # Norwegian mBERT monolingual
│   ├── train_norwegian_xlmr_baseline.py   # Norwegian XLM-R monolingual
│   ├── train_english_to_danish_zero_shot_baseline.py
│   ├── train_english_to_danish_xlmr_zero_shot_baseline.py
│   ├── train_norwegian_to_danish_zero_shot_baseline.py
│   ├── train_norwegian_to_danish_xlmr_zero_shot_baseline.py
│   ├── train_english_to_danish_mbert_finetune.py
│   ├── train_english_to_danish_xlmr_finetune.py
│   ├── train_norwegian_to_danish_mbert_finetune.py
│   ├── train_norwegian_to_danish_xlmr_finetune.py
│   ├── span_f1.py                         # Span-level F1 evaluation script
│   ├── check_label_consistency.py         # Verify label sets across languages
│   └── language_distance_eval.py          # WALS typological distance computation
├── Datasets/
│   ├── English/                           # en_ewt-ud-{train,dev,test}.iob2
│   ├── Norwegian/                         # nor_nynorsk or bokmål IOB2 files
│   └── Danish/                            # da_ddt-ud-{train,dev,test}.iob2
├── outputs/                               # Saved metrics and predictions per run
├── Baseline_predictions/                  # Reference prediction artifacts
├── logs/                                  # SLURM .out/.err logs from HPC runs
├── Latex/                                 # LaTeX report source
│   ├── main.tex
│   ├── chapters/
│   ├── figures/
│   └── main.pdf                           # Compiled report
├── collect_results.py                     # Aggregate outputs → summary CSVs
├── multiseed_analysis.py                  # Multi-seed means, std, paired t-tests
├── bootstrap_test.py                      # Paired bootstrap resampling (10k iters)
├── entity_type_analysis.py                # Per-entity-type F1 breakdown
├── plot_figures.py                        # Regenerate all report figures
└── Requirements.txt
```

---

## Dependencies

**Python version:** 3.9+

Install all dependencies with:

```bash
pip install -r Requirements.txt
```

### Full dependency list

| Package | Purpose |
| --- | --- |
| `torch` | PyTorch — deep learning backend for all transformer models |
| `transformers` | Hugging Face Transformers — mBERT, XLM-R, tokenizers, Trainer API |
| `datasets` | Hugging Face Datasets — data loading utilities |
| `seqeval` | Sequence labelling evaluation (token-level F1, precision, recall) |
| `accelerate` | Hugging Face Accelerate — multi-GPU / mixed-precision training support |
| `numpy` | Numerical operations |
| `pandas` | Results aggregation and CSV handling |
| `matplotlib` | Figure generation (learning curves, scatter plots, bar charts) |
| `scipy` | Statistical tests (paired t-tests) |
| `lang2vec` | WALS-based typological feature vectors for linguistic distance computation |

All packages are also listed in `Requirements.txt`.

---

## Reproducing the Results

### Step 1 — Install dependencies

```bash
pip install -r Requirements.txt
```

### Step 2 — Verify label consistency (optional sanity check)

```bash
python Baselines/check_label_consistency.py
```

### Step 3 — Run training on HPC

Each experiment is launched by its own script in `Baselines/`. Because training is computationally heavy, **submit each script as a SLURM job** (see [Job File Template](#job-file-template) below).

**Monolingual baselines:**

```bash
python Baselines/train_english_xlmr_baseline.py
python Baselines/train_english_baseline.py
python Baselines/train_norwegian_xlmr_baseline.py
python Baselines/train_norwegian_baseline.py
python Baselines/train_danish_xlmr_baseline.py
python Baselines/train_danish_baseline.py
```

**Zero-shot transfer to Danish:**

```bash
python Baselines/train_english_to_danish_xlmr_zero_shot_baseline.py
python Baselines/train_english_to_danish_zero_shot_baseline.py
python Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py
python Baselines/train_norwegian_to_danish_zero_shot_baseline.py
```

**Transfer + Danish fine-tuning:**

```bash
python Baselines/train_english_to_danish_xlmr_finetune.py
python Baselines/train_english_to_danish_mbert_finetune.py
python Baselines/train_norwegian_to_danish_xlmr_finetune.py
python Baselines/train_norwegian_to_danish_mbert_finetune.py
```

Each script writes metrics and predictions to `outputs/<experiment_name>/`.

### Step 4 — Run analysis and regenerate figures

Once all training runs are complete:

```bash
python collect_results.py        # Aggregate metrics into Latex/*.csv
python multiseed_analysis.py     # Multi-seed comparison, t-tests
python bootstrap_test.py         # Paired bootstrap resampling
python entity_type_analysis.py   # Per-entity-type F1 breakdown
python plot_figures.py           # Regenerate all figures in Latex/figures/
```

### Step 5 — Evaluate predictions manually (optional)

```bash
python Baselines/span_f1.py \
    Datasets/Danish/da_ddt-ud-dev.iob2 \
    outputs/norwegian_to_danish_xlmr_zero_shot/dev_predictions.iob2
```

---

## Job File Template

Since training was run on a SLURM-based HPC cluster, you need to create your own job files. Below is a template — adjust the partition name, time limit, and paths to match your cluster's configuration.

```bash
#!/bin/bash
#SBATCH --job-name=nlp_ner
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu          # Change to your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load your environment (adjust to your cluster's module system)
module load python/3.9          # or: source activate <your_conda_env>
source /path/to/your/venv/bin/activate

# Run the desired training script
cd /path/to/NLP-Project
python Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py
```

Save this as e.g. `jobs/run_no2da_xlmr.sh` and submit with:

```bash
sbatch jobs/run_no2da_xlmr.sh
```

> **Note:** The `logs/` directory must exist before submitting (`mkdir -p logs`). The actual job files used in our experiments are not included in this repository; the template above covers all the settings we used.

---

## Building the Report

The report is written in LaTeX (ACL style). Build from the `Latex/` directory:

```bash
cd Latex
latexmk -pdf main.tex
```

Requires a working TeX installation (TeX Live 2020+ or MacTeX). The compiled PDF is already included as `Latex/main.pdf`.

---

## Linguistic Distance Analysis

The WALS-based typological distance computation is self-contained:

```bash
python Baselines/language_distance_eval.py
```

This uses `lang2vec` with the `syntax_wals` feature set and prints pairwise cosine distances for English, Danish, Norwegian, Swedish, and German.
