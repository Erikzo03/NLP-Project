#!/usr/bin/env bash
set -euo pipefail

# Run English, Danish, and Norwegian XLM-R baselines with shared CLI overrides.
python Baselines/train_english_xlmr_baseline.py "$@"
python Baselines/train_danish_xlmr_baseline.py "$@"
python Baselines/train_norwegian_xlmr_baseline.py "$@"

echo "All XLM-R baseline runs completed."