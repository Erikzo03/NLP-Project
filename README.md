# NLP-Project

This repository contains the code, data, results, and report for the Danish NER transfer project. The current report is organized in ACL style under `Latex/`, with chapter files in `Latex/chapters/` and the final entry point in `Latex/main.tex`.

## Project Structure

- `Baselines/`: shared training code and launchers for monolingual, zero-shot, and fine-tuning experiments.
- `Datasets/`: language-specific IOB2 data files.
- `Latex/`: report source, figures, bibliography, and the final PDF.
- `outputs/`: saved metrics and predictions from completed experiments.
- `Baseline_predictions/`: reference prediction artifacts.

## Report Build

The report uses the ACL style and is split into chapter files for readability. Build it from the `Latex/` directory:

```bash
cd Latex
latexmk -pdf main.tex
```

The report depends on a working TeX installation such as TeX Live or MacTeX. Packages like `graphicx`, `booktabs`, and `hyperref` come from the TeX distribution and are already resolved in the compiled report; they are not Python dependencies.

## Training Scripts

Shared training logic lives in:

- `Baselines/train_ner_baseline.py`

Main baseline launchers:

- `Baselines/train_english_baseline.py`
- `Baselines/train_danish_baseline.py`
- `Baselines/train_norwegian_baseline.py`
- `Baselines/train_english_xlmr_baseline.py`
- `Baselines/train_danish_xlmr_baseline.py`
- `Baselines/train_norwegian_xlmr_baseline.py`

Cross-lingual launchers:

- `Baselines/train_english_to_danish_zero_shot_baseline.py`
- `Baselines/train_norwegian_to_danish_zero_shot_baseline.py`
- `Baselines/train_english_to_danish_xlmr_zero_shot_baseline.py`
- `Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py`

Example run:

```bash
python Baselines/train_english_xlmr_baseline.py
```

Run all mBERT baselines:

```bash
bash Baselines/run_all_baselines.sh
```

Run all XLM-R baselines:

```bash
bash Baselines/run_all_xlmr_baselines.sh
```

Run the combined mBERT and zero-shot suite:

```bash
bash Baselines/run_all_mbert_and_zero_shot_to_danish.sh --quick
```

## Validation Utilities

Check that the label inventories match across English, Danish, and Norwegian:

```bash
python Baselines/check_label_consistency.py
```

Evaluate predictions with the project scorer:

```bash
python Baselines/span_f1.py Datasets/en_ewt-ud-dev.iob2 outputs/english_bert_baseline/dev_predictions.iob2
```

## Python Dependencies

Install the Python packages with:

```bash
pip install -r Requirements.txt
```

The report plots and distance analysis use `matplotlib`, `pandas`, and `scipy` in addition to the core training stack.
