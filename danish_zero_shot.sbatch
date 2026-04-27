#!/bin/bash
#SBATCH --job-name=en-to-da-zero-shot
#SBATCH --output=logs/english-to-danish-%j.out
#SBATCH --error=logs/english-to-danish-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/onok/NLP-Project

source ~/ner_env/bin/activate

# ENGLISH BASELINE (train English → zero-shot Danish test)
python Baselines/train_english_baseline.py \
    --train_file /home/onok/NLP-Project/Datasets/English/en_ewt-ud-train.iob2 \
    --dev_file /home/onok/NLP-Project/Datasets/English/en_ewt-ud-dev.iob2 \
    --test_file /home/onok/NLP-Project/Datasets/Danish/da_ddt-ud-test.iob2 \
    --model_name bert-base-multilingual-cased \
    --output_dir /home/onok/NLP-Project/outputs/english_to_danish_zero_shot \
    --label_report_path /home/onok/NLP-Project/Baselines/reports/english_to_danish_labels.json
