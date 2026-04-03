#!/usr/bin/env bash
set -euo pipefail

# Run English, Danish, and Norwegian baselines with shared defaults.
python Baselines/train_english_baseline.py "$@"
python Baselines/train_danish_baseline.py "$@"
python Baselines/train_norwegian_baseline.py "$@"

echo "All baseline runs completed."
