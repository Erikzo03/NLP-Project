#!/bin/bash
#SBATCH --job-name=nlp-container-pipeline
#SBATCH --output=logs/nlp-container-pipeline-%j.out
#SBATCH --error=logs/nlp-container-pipeline-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "$HOME/NLP-Project"
mkdir -p logs outputs containers/tmp

module purge

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available on this host."
  exit 1
fi

IMAGE="$HOME/NLP-Project/containers/nlp_project.sif"
DEF_FILE="$HOME/NLP-Project/containers/nlp_project.def"

if [[ ! -f "$IMAGE" ]]; then
  echo "Container image not found: $IMAGE"
  echo "Build it first with:"
  echo "  cd ~/NLP-Project"
  echo "  apptainer build --fakeroot containers/nlp_project.sif containers/nlp_project.def"
  echo "or, if remote build is enabled:"
  echo "  apptainer build --remote containers/nlp_project.sif containers/nlp_project.def"
  exit 1
fi

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Container: ${IMAGE}"
echo "Apptainer binary: ${APPTAINER_BIN}"
${APPTAINER_BIN} exec --nv \
  --bind "$HOME/NLP-Project:/workspace" \
  --env TMPDIR=/tmp \
  "$IMAGE" \
  bash /workspace/jobs/run_container_full_pipeline.sh
