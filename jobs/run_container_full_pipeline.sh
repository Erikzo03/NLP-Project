#!/bin/bash
set -euo pipefail

cd /workspace
mkdir -p logs outputs

echo "=== Container environment check ==="
python --version
python - <<'PY'
import torch, transformers, datasets, seqeval, accelerate, numpy
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("seqeval:", seqeval.__version__)
print("accelerate:", accelerate.__version__)
print("numpy:", numpy.__version__)
PY

echo "=== Running monolingual baselines ==="
python -u Baselines/train_english_baseline.py
python -u Baselines/train_danish_baseline.py
python -u Baselines/train_norwegian_baseline.py
python -u Baselines/train_english_xlmr_baseline.py
python -u Baselines/train_danish_xlmr_baseline.py
python -u Baselines/train_norwegian_xlmr_baseline.py

echo "=== Running zero-shot transfer to Danish ==="
python -u Baselines/train_english_to_danish_zero_shot_baseline.py
python -u Baselines/train_norwegian_to_danish_zero_shot_baseline.py
python -u Baselines/train_english_to_danish_xlmr_zero_shot_baseline.py
python -u Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py

echo "=== Running full fine-tuning on Danish ==="
python -u Baselines/train_english_to_danish_mbert_finetune.py
python -u Baselines/train_norwegian_to_danish_mbert_finetune.py
python -u Baselines/train_english_to_danish_xlmr_finetune.py
python -u Baselines/train_norwegian_to_danish_xlmr_finetune.py

echo "=== Running Danish data-budget experiments (mBERT) ==="
python -u Baselines/train_danish_baseline.py --train_fraction 0.10 --output_dir outputs/danish_bert_baseline_10pct
python -u Baselines/train_danish_baseline.py --train_fraction 0.25 --output_dir outputs/danish_bert_baseline_25pct
python -u Baselines/train_danish_baseline.py --train_fraction 0.50 --output_dir outputs/danish_bert_baseline_50pct

echo "=== Running transfer fine-tuning budgets (mBERT) ==="
python -u Baselines/train_english_to_danish_mbert_finetune.py --train_fraction 0.10 --output_dir outputs/english_to_danish_mbert_finetuned_danish_10pct
python -u Baselines/train_english_to_danish_mbert_finetune.py --train_fraction 0.25 --output_dir outputs/english_to_danish_mbert_finetuned_danish_25pct
python -u Baselines/train_english_to_danish_mbert_finetune.py --train_fraction 0.50 --output_dir outputs/english_to_danish_mbert_finetuned_danish_50pct
python -u Baselines/train_norwegian_to_danish_mbert_finetune.py --train_fraction 0.10 --output_dir outputs/norwegian_to_danish_mbert_finetuned_danish_10pct
python -u Baselines/train_norwegian_to_danish_mbert_finetune.py --train_fraction 0.25 --output_dir outputs/norwegian_to_danish_mbert_finetuned_danish_25pct
python -u Baselines/train_norwegian_to_danish_mbert_finetune.py --train_fraction 0.50 --output_dir outputs/norwegian_to_danish_mbert_finetuned_danish_50pct

echo "=== Running transfer fine-tuning budgets (XLM-R) ==="
python -u Baselines/train_english_to_danish_xlmr_finetune.py --train_fraction 0.10 --output_dir outputs/english_to_danish_xlmr_finetuned_danish_10pct
python -u Baselines/train_english_to_danish_xlmr_finetune.py --train_fraction 0.25 --output_dir outputs/english_to_danish_xlmr_finetuned_danish_25pct
python -u Baselines/train_english_to_danish_xlmr_finetune.py --train_fraction 0.50 --output_dir outputs/english_to_danish_xlmr_finetuned_danish_50pct
python -u Baselines/train_norwegian_to_danish_xlmr_finetune.py --train_fraction 0.10 --output_dir outputs/norwegian_to_danish_xlmr_finetuned_danish_10pct
python -u Baselines/train_norwegian_to_danish_xlmr_finetune.py --train_fraction 0.25 --output_dir outputs/norwegian_to_danish_xlmr_finetuned_danish_25pct
python -u Baselines/train_norwegian_to_danish_xlmr_finetune.py --train_fraction 0.50 --output_dir outputs/norwegian_to_danish_xlmr_finetuned_danish_50pct

echo "=== All container runs completed ==="
