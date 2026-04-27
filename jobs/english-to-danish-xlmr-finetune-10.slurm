#!/bin/bash
#SBATCH --job-name=en2da-xft-10
#SBATCH --output=logs/en2da-xft-10-%j.out
#SBATCH --error=logs/en2da-xft-10-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "$HOME/NLP-Project"
mkdir -p logs outputs

module purge
module load Anaconda3/2024.06-1

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

python -u Baselines/train_english_to_danish_xlmr_finetune.py --train_fraction 0.10 --output_dir outputs/english_to_danish_xlmr_finetuned_danish_10pct
