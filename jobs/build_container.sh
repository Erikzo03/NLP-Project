#!/bin/bash
set -euo pipefail

cd "$HOME/NLP-Project"

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN=singularity
else
  echo "Neither apptainer nor singularity is available."
  exit 1
fi

mkdir -p containers

echo "Building Apptainer image..."
${APPTAINER_BIN} build --fakeroot containers/nlp_project.sif containers/nlp_project.def

echo "Build complete: containers/nlp_project.sif"
