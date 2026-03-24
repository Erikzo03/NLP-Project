# NLP-Project

## English-only BERT baseline (EWT NER)

This repo now includes a baseline training script:

- `train_english_bert_baseline.py`

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
