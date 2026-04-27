#!/bin/bash
#SBATCH --job-name=en-da-xlmr
#SBATCH --output=logs/en-da-xlmr-%j.out
#SBATCH --error=logs/en-da-xlmr-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/onok/NLP-Project
source ~/ner_env/bin/activate

python Baselines/train_english_to_danish_xlmr_zero_shot_baseline.py
