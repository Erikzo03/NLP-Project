#!/bin/bash
#SBATCH --job-name=no2da-xlmr-zs
#SBATCH --output=logs/no2da-xlmr-zs-%j.out
#SBATCH --error=logs/no2da-xlmr-zs-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "$HOME/NLP-Project"
mkdir -p logs outputs

source ~/ner_env/bin/activate

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Python: $(which python)"
python --version
nvidia-smi || true

python - <<'PY'
import torch, transformers, datasets, seqeval
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("seqeval:", seqeval.__version__)
PY

python -u Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py
