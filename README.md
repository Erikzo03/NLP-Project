# NLP-Project

**Repository:** [github.com/Erikzo03/NLP-Project](https://github.com/Erikzo03/NLP-Project)

This repository contains the code, data, results, and report for the Danish NER transfer project. The current report is organized in ACL style under `Latex/`, with chapter files in `Latex/chapters/` and the final entry point in `Latex/main.tex`.

## Project Structure

- `Baselines/`: shared training code and launchers for monolingual, zero-shot, and fine-tuning experiments.
- `Datasets/`: language-specific IOB2 data files (`Danish/`, `English/`, `Norwegian/`).
- `Latex/`: report source, figures, bibliography, and the final PDF.
- `outputs/`: saved metrics and predictions from completed experiments.
- `Baseline_predictions/`: reference prediction artifacts for the English-to-Danish zero-shot run.

## Report Build

The report uses the ACL style and is split into chapter files for readability. Build it from the `Latex/` directory:

```bash
cd Latex
latexmk -pdf main.tex
```

The report depends on a working TeX installation such as TeX Live or MacTeX. Packages like `graphicx`, `booktabs`, and `hyperref` come from the TeX distribution and are already resolved in the compiled report; they are not Python dependencies.

## Python Dependencies

Install the Python packages with:

```bash
pip install -r Requirements.txt
```

The requirements cover the full stack: `torch`, `transformers`, `datasets`, and `seqeval` for model training; `lang2vec` and `scipy` for the typological-distance analysis; `matplotlib`, `pandas`, and `numpy` for results collection and figure generation.

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

## Analysis Scripts

These root-level scripts collect results, run statistical tests, and regenerate figures:

- `collect_results.py` — aggregates metrics from `outputs/` into summary CSVs under `Latex/`.
- `multiseed_analysis.py` — computes multi-seed means, standard deviations, and paired *t*-tests.
- `bootstrap_test.py` — runs paired bootstrap resampling tests (10,000 iterations) on saved prediction files.
- `entity_type_analysis.py` — computes per-entity-type F1 breakdowns.
- `plot_figures.py` — regenerates all report figures (learning curve, scatter plot, bar chart) from the summary CSVs.

Run them in order after training is complete:

```bash
python collect_results.py
python multiseed_analysis.py
python bootstrap_test.py
python entity_type_analysis.py
python plot_figures.py
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
