#!/bin/bash
#SBATCH --job-name=nor-cn3
#SBATCH --output=logs/norwegian-cn3-%j.out
#SBATCH --error=logs/norwegian-cn3-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/onok/NLP-Project

source ~/ner_env/bin/activate

python train_danish_baseline.py


