#!/bin/bash
#SBATCH --job-name=nor_ner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=nor_ner_%j.out
#SBATCH --error=nor_ner_%j.err

module purge
module load python/3.10
conda activate ner_env

cd /home/onok/NLP-Project/NLP-Project/Baselines/
python train_norwegian_baseline.py \
    --train_file Datasets/Norwegian/nno_norne-ud-train.iob2 \
    --dev_file Datasets/Norwegian/nno_norne-ud-dev.iob2 \
    --test_file Datasets/Danish/da_ddt-ud-test.iob2 \
    --model_name bert-base-multilingual-cased \
    --output_dir outputs/norwegian_to_danish_mbert
