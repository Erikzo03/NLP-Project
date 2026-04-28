#!/bin/bash
#SBATCH --job-name=submit-pipeline
#SBATCH --output=logs/submit-pipeline-%j.out
#SBATCH --error=logs/submit-pipeline-%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn3
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

cd "$HOME/NLP-Project"
mkdir -p logs

source ~/ner_env/bin/activate

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Python: $(which python)"

bash jobs/submit_all_pipeline.sh