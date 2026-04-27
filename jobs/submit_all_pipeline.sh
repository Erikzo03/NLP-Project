#!/bin/bash
#
# Submit all NLP-Project SLURM jobs in dependency order.
# Usage: bash submit_all_pipeline.sh
#
# Job dependency rules:
# 1. Zero-shot jobs have no dependencies (independent)
# 2. Fine-tuning jobs depend on their matching zero-shot job
# 3. Danish budget baselines are independent
#
# To cancel all jobs from this pipeline, store the job IDs and use:
#   scancel <jobid1> <jobid2> ...

set -euo pipefail

cd "$HOME/NLP-Project" || exit 1

echo "============================================"
echo "NLP-Project SLURM Pipeline Submission"
echo "============================================"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo ""

# Ensure logs directory exists
mkdir -p logs

# ============ PHASE 1: Submit zero-shot baseline jobs (independent) ============
echo "[PHASE 1] Submitting zero-shot transfer jobs (no dependencies)..."
echo ""

J_EN_MBERT_ZS=$(sbatch jobs/english-to-danish-mbert-zero-shot.slurm | awk '{print $4}')
echo "✓ English→Danish mBERT zero-shot:         Job ID ${J_EN_MBERT_ZS}"

J_NO_MBERT_ZS=$(sbatch jobs/norwegian-to-danish-mbert-zero-shot.slurm | awk '{print $4}')
echo "✓ Norwegian→Danish mBERT zero-shot:       Job ID ${J_NO_MBERT_ZS}"

J_EN_XLMR_ZS=$(sbatch jobs/english-to-danish-xlmr-zero-shot.slurm | awk '{print $4}')
echo "✓ English→Danish XLM-R zero-shot:         Job ID ${J_EN_XLMR_ZS}"

J_NO_XLMR_ZS=$(sbatch jobs/norwegian-to-danish-xlmr-zero-shot.slurm | awk '{print $4}')
echo "✓ Norwegian→Danish XLM-R zero-shot:       Job ID ${J_NO_XLMR_ZS}"

echo ""
echo "[PHASE 1] Complete. Waiting for zero-shot jobs to complete before proceeding..."
echo ""

# ============ PHASE 2: Full fine-tune jobs (depend on matching zero-shot) ============
echo "[PHASE 2] Submitting full fine-tuning jobs (depend on zero-shot)..."
echo ""

FT_EN_MBERT=$(sbatch --dependency=afterok:${J_EN_MBERT_ZS} jobs/english-to-danish-mbert-finetune.slurm | awk '{print $4}')
echo "✓ English→Danish mBERT fine-tune (full):  Job ID ${FT_EN_MBERT} (depends on ${J_EN_MBERT_ZS})"

FT_NO_MBERT=$(sbatch --dependency=afterok:${J_NO_MBERT_ZS} jobs/norwegian-to-danish-mbert-finetune.slurm | awk '{print $4}')
echo "✓ Norwegian→Danish mBERT fine-tune (full): Job ID ${FT_NO_MBERT} (depends on ${J_NO_MBERT_ZS})"

FT_EN_XLMR=$(sbatch --dependency=afterok:${J_EN_XLMR_ZS} jobs/english-to-danish-xlmr-finetune.slurm | awk '{print $4}')
echo "✓ English→Danish XLM-R fine-tune (full):  Job ID ${FT_EN_XLMR} (depends on ${J_EN_XLMR_ZS})"

FT_NO_XLMR=$(sbatch --dependency=afterok:${J_NO_XLMR_ZS} jobs/norwegian-to-danish-xlmr-finetune.slurm | awk '{print $4}')
echo "✓ Norwegian→Danish XLM-R fine-tune (full): Job ID ${FT_NO_XLMR} (depends on ${J_NO_XLMR_ZS})"

echo ""

# ============ PHASE 3: Data-budget fine-tuning jobs (10%, 25%, 50%) ============
echo "[PHASE 3] Submitting data-budget fine-tuning jobs (10%, 25%, 50%)..."
echo ""

for PERCENT in 10 25 50; do
  echo "  Submitting ${PERCENT}% budget variants..."
  
  FT_EN_MBERT_P=$(sbatch --dependency=afterok:${J_EN_MBERT_ZS} jobs/english-to-danish-mbert-finetune-${PERCENT}.slurm | awk '{print $4}')
  echo "  ✓ English→Danish mBERT fine-tune (${PERCENT}%):   Job ID ${FT_EN_MBERT_P}"
  
  FT_NO_MBERT_P=$(sbatch --dependency=afterok:${J_NO_MBERT_ZS} jobs/norwegian-to-danish-mbert-finetune-${PERCENT}.slurm | awk '{print $4}')
  echo "  ✓ Norwegian→Danish mBERT fine-tune (${PERCENT}%):  Job ID ${FT_NO_MBERT_P}"
  
  FT_EN_XLMR_P=$(sbatch --dependency=afterok:${J_EN_XLMR_ZS} jobs/english-to-danish-xlmr-finetune-${PERCENT}.slurm | awk '{print $4}')
  echo "  ✓ English→Danish XLM-R fine-tune (${PERCENT}%):   Job ID ${FT_EN_XLMR_P}"
  
  FT_NO_XLMR_P=$(sbatch --dependency=afterok:${J_NO_XLMR_ZS} jobs/norwegian-to-danish-xlmr-finetune-${PERCENT}.slurm | awk '{print $4}')
  echo "  ✓ Norwegian→Danish XLM-R fine-tune (${PERCENT}%):  Job ID ${FT_NO_XLMR_P}"
done

echo ""

# ============ PHASE 4: Independent Danish-only baselines ============
echo "[PHASE 4] Submitting independent Danish-only baseline jobs..."
echo ""

DA_MBERT_10=$(sbatch jobs/danish-mbert-10.slurm | awk '{print $4}')
echo "✓ Danish mBERT (10%):  Job ID ${DA_MBERT_10}"

DA_MBERT_25=$(sbatch jobs/danish-mbert-25.slurm | awk '{print $4}')
echo "✓ Danish mBERT (25%):  Job ID ${DA_MBERT_25}"

DA_MBERT_50=$(sbatch jobs/danish-mbert-50.slurm | awk '{print $4}')
echo "✓ Danish mBERT (50%):  Job ID ${DA_MBERT_50}"

echo ""
echo "============================================"
echo "All jobs submitted successfully!"
echo "============================================"
echo ""
echo "To check queue status:"
echo "  squeue -u eray"
echo ""
echo "To check completed jobs:"
echo "  sacct -u eray --format=JobID,JobName,State,ExitCode,Elapsed -S now-1day"
echo ""
echo "To follow a specific job's output (replace JOBID):"
echo "  tail -f logs/<jobname>-JOBID.out"
echo ""
echo "To cancel all jobs from this pipeline:"
echo "  scancel ${J_EN_MBERT_ZS} ${J_NO_MBERT_ZS} ${J_EN_XLMR_ZS} ${J_NO_XLMR_ZS} ${FT_EN_MBERT} ${FT_NO_MBERT} ${FT_EN_XLMR} ${FT_NO_XLMR}"
echo ""
