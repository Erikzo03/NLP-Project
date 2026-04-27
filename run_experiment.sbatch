#!/bin/bash
#SBATCH --job-name=english_to_danish_xlmr_zero_shot
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

cd /home/onok/NLP-Project
source ~/ner_env/bin/activate

python "$SCRIPT"
