#!/bin/bash
#SBATCH --job-name=da-baseline
#SBATCH --output=logs/danish-baseline-%j.out
#SBATCH --error=logs/danish-baseline-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/onok/NLP-Project
source ~/ner_env/bin/activate

python Baselines/train_danish_baseline.py
