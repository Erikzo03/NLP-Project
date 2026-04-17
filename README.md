# NLP-Project

## English-only BERT baseline (EWT NER)

This repo now includes a baseline training script:

- `train_english_bert_baseline.py`

## Baselines folder (English, Danish, Norwegian)

All baseline training entrypoints are now grouped in:

- `Baselines/train_english_baseline.py`
- `Baselines/train_danish_baseline.py`
- `Baselines/train_norwegian_baseline.py`
- `Baselines/train_english_xlmr_baseline.py`
- `Baselines/train_danish_xlmr_baseline.py`
- `Baselines/train_norwegian_xlmr_baseline.py`

Shared training logic is in:

- `Baselines/train_ner_baseline.py`

Each baseline writes output to its own folder under `outputs/`.

Run English baseline:

```bash
python Baselines/train_english_baseline.py
```

Model default in this launcher: `bert-base-multilingual-cased`.

Run Danish baseline:

```bash
python Baselines/train_danish_baseline.py
```

Run Norwegian baseline:

```bash
python Baselines/train_norwegian_baseline.py
```

Run English baseline (XLM-R):

```bash
python Baselines/train_english_xlmr_baseline.py
```

Run Danish baseline (XLM-R):

```bash
python Baselines/train_danish_xlmr_baseline.py
```

Run Norwegian baseline (XLM-R):

```bash
python Baselines/train_norwegian_xlmr_baseline.py
```

Run all mBERT baselines:

```bash
bash Baselines/run_all_baselines.sh
```

Run all XLM-R baselines:

```bash
bash Baselines/run_all_xlmr_baselines.sh
```

Run all requested experiments in one command (mBERT monolingual baselines + zero-shot to Danish with mBERT and XLM-R):

```bash
bash Baselines/run_all_mbert_and_zero_shot_to_danish.sh
```

If you run on HPC with modules, load your modules first, then run the script.

This unified script runs, in order:

- `Baselines/train_english_baseline.py`
- `Baselines/train_danish_baseline.py`
- `Baselines/train_norwegian_baseline.py`
- `Baselines/train_english_to_danish_zero_shot_baseline.py`
- `Baselines/train_norwegian_to_danish_zero_shot_baseline.py`
- `Baselines/train_english_to_danish_xlmr_zero_shot_baseline.py`
- `Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py`

You can pass shared overrides to all runs, for example:

```bash
bash Baselines/run_all_mbert_and_zero_shot_to_danish.sh \
  --num_train_epochs 4 \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --fp16
```

Useful runner flags:

- `--quick`: adds fast defaults for smoke testing (`--num_train_epochs 1 --train_batch_size 4 --eval_batch_size 8`).
- `--skip-mbert-baselines`: skip English/Danish/Norwegian mBERT monolingual runs.
- `--skip-mbert-zero-shot`: skip mBERT zero-shot runs.
- `--skip-xlmr-zero-shot`: skip XLM-R zero-shot runs.

Example smoke run:

```bash
bash Baselines/run_all_mbert_and_zero_shot_to_danish.sh --quick
```

Expected output folders from this unified run:

- `outputs/english_bert_baseline`
- `outputs/danish_bert_baseline`
- `outputs/norwegian_bert_baseline`
- `outputs/english_to_danish_mbert_zero_shot`
- `outputs/norwegian_to_danish_mbert_zero_shot`
- `outputs/english_to_danish_xlmr_zero_shot`
- `outputs/norwegian_to_danish_xlmr_zero_shot`

To verify if labels are consistent across English, Danish, and Norwegian datasets:

```bash
python Baselines/check_label_consistency.py
```

This produces:

- `Baselines/reports/label_consistency_report.json`

It trains a token-classification BERT model on:

- `Datasets/en_ewt-ud-train.iob2`

and evaluates on:

- `Datasets/en_ewt-ud-dev.iob2`

then exports predictions for:

- `Datasets/en_ewt-ud-test-masked.iob2`

in the same CoNLL-like format required by `span_f1.py`.

## Install

```bash
pip install torch transformers datasets seqeval accelerate
```

## Run baseline (local)

```bash
python train_english_bert_baseline.py \
  --model_name bert-base-cased \
  --output_dir outputs/english_bert_baseline \
  --num_train_epochs 4 \
  --train_batch_size 8 \
  --eval_batch_size 16
```

## Evaluate dev predictions

```bash
python span_f1.py Datasets/en_ewt-ud-dev.iob2 outputs/english_bert_baseline/dev_predictions.iob2
```

## If you use university/HPC servers

Use the same script and increase efficiency settings, for example:

```bash
python train_english_bert_baseline.py \
  --model_name bert-base-cased \
  --output_dir outputs/english_bert_hpc \
  --num_train_epochs 5 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --fp16
```

If you later switch to cross-lingual transfer, replace `--model_name` with `bert-base-multilingual-cased`.
