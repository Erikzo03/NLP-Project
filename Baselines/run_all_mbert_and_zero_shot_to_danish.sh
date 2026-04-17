#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log_step() {
  local message="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message"
}

SKIP_MBERT_BASELINES=0
SKIP_MBERT_ZERO_SHOT=0
SKIP_XLMR_ZERO_SHOT=0
QUICK_MODE=0
PASSTHROUGH_ARGS=()

while (($#)); do
  case "$1" in
    --skip-mbert-baselines)
      SKIP_MBERT_BASELINES=1
      shift
      ;;
    --skip-mbert-zero-shot)
      SKIP_MBERT_ZERO_SHOT=1
      shift
      ;;
    --skip-xlmr-zero-shot)
      SKIP_XLMR_ZERO_SHOT=1
      shift
      ;;
    --quick)
      QUICK_MODE=1
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

COMMON_ARGS=("${PASSTHROUGH_ARGS[@]}")
if [[ ${QUICK_MODE} -eq 1 ]]; then
  log_step "Quick mode enabled: adding fast defaults (1 epoch, small batch sizes)."
  COMMON_ARGS+=(--num_train_epochs 1 --train_batch_size 4 --eval_batch_size 8)
fi

if [[ ${SKIP_MBERT_BASELINES} -eq 0 ]]; then
  log_step "Starting monolingual mBERT baselines (English, Danish, Norwegian)."
  python Baselines/train_english_baseline.py "${COMMON_ARGS[@]}"
  python Baselines/train_danish_baseline.py "${COMMON_ARGS[@]}"
  python Baselines/train_norwegian_baseline.py "${COMMON_ARGS[@]}"
else
  log_step "Skipping monolingual mBERT baselines."
fi

if [[ ${SKIP_MBERT_ZERO_SHOT} -eq 0 ]]; then
  log_step "Starting zero-shot mBERT transfer runs to Danish."
  python Baselines/train_english_to_danish_zero_shot_baseline.py "${COMMON_ARGS[@]}"
  python Baselines/train_norwegian_to_danish_zero_shot_baseline.py "${COMMON_ARGS[@]}"
else
  log_step "Skipping zero-shot mBERT transfer runs."
fi

if [[ ${SKIP_XLMR_ZERO_SHOT} -eq 0 ]]; then
  log_step "Starting zero-shot XLM-R transfer runs to Danish."
  python Baselines/train_english_to_danish_xlmr_zero_shot_baseline.py "${COMMON_ARGS[@]}"
  python Baselines/train_norwegian_to_danish_xlmr_zero_shot_baseline.py "${COMMON_ARGS[@]}"
else
  log_step "Skipping zero-shot XLM-R transfer runs."
fi

log_step "All requested baseline and zero-shot runs completed successfully."
