#!/bin/bash
#
# Quick reference for NLP-Project HPC commands
# Usage: source hpc_commands.sh, then use the functions
#

# Check Python environment
check_env() {
  echo "=== Python Environment ==="
  echo "Current shell: $SHELL"
  echo "Python path: $(which python)"
  python --version
  echo ""
  echo "=== Required packages ==="
  python - <<'PY'
try:
    import torch
    print(f"✓ torch: {torch.__version__}")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers: {e}")

try:
    import datasets
    print(f"✓ datasets: {datasets.__version__}")
except ImportError as e:
    print(f"✗ datasets: {e}")

try:
    import seqeval
    print(f"✓ seqeval: {seqeval.__version__}")
except ImportError as e:
    print(f"✗ seqeval: {e}")
PY
}

# View job queue
view_queue() {
  echo "=== Job Queue (squeue) ==="
  squeue -u eray --format="%.18i %.20j %.8T %.10M %.10l %.10L %.6D %C"
  echo ""
}

# View completed jobs
view_history() {
  echo "=== Recent Job History (last 24h) ==="
  sacct -u eray --format=JobID,JobName%20,State,ExitCode,Elapsed -S now-1day
  echo ""
}

# Monitor a specific job by name (e.g., mon_job en2da-mbert-zs)
mon_job() {
  local job_name="$1"
  if [[ -z "$job_name" ]]; then
    echo "Usage: mon_job <job_name>"
    echo "Example: mon_job en2da-mbert-zs"
    return 1
  fi
  
  local log_file=$(ls logs/${job_name}-*.out 2>/dev/null | tail -1)
  if [[ -f "$log_file" ]]; then
    tail -f "$log_file"
  else
    echo "Log file not found for job: $job_name"
    ls -lht logs/ | head -10
  fi
}

# Check errors in logs
check_errors() {
  echo "=== Recent Errors in logs/ ==="
  if grep -l "Traceback\|Error\|FAILED\|bad_request" logs/*.err 2>/dev/null | head -5; then
    echo ""
    echo "Use 'view_error <logfile>' to inspect full error."
  else
    echo "No errors found in logs."
  fi
}

# View a specific error log
view_error() {
  local file="$1"
  if [[ -z "$file" ]]; then
    echo "Usage: view_error <error_logfile>"
    echo "Example: view_error logs/en2da-mbert-zs-12345.err"
    return 1
  fi
  
  tail -50 "$file"
}

# Git sync
git_sync() {
  echo "=== Git Status ==="
  git status
  echo ""
  echo "=== Git Pull ==="
  git pull --rebase origin main || (git stash && git pull --rebase origin main && git stash pop)
}

# List all SLURM jobs available
list_jobs() {
  echo "=== Available SLURM Jobs ==="
  ls -1 jobs/*.slurm | sed 's|jobs/||' | sed 's|.slurm||'
}

# Print help
hpc_help() {
  cat <<'EOF'
NLP-Project HPC Command Reference
==================================

Utility Functions:
  check_env           - Verify Python environment and required packages
  view_queue          - Show current SLURM job queue
  view_history        - Show completed jobs (last 24h)
  mon_job <name>      - Monitor a specific job (tail output)
  check_errors        - Find errors in recent logs
  view_error <file>   - View full error log
  git_sync            - Pull latest from GitHub
  list_jobs           - List available SLURM job scripts
  hpc_help            - Show this help message

Quick Workflow:
  1. Check environment:        check_env
  2. Sync with GitHub:         git_sync
  3. Submit all jobs:          cd jobs && bash submit_all_pipeline.sh
  4. Monitor queue:            view_queue
  5. Check recent jobs:        view_history
  6. Monitor specific job:      mon_job <name>
  7. If errors:                check_errors
  8. View error details:       view_error logs/<file>.err

Example Job Monitoring:
  mon_job en2da-mbert-zs      # Follow English→Danish mBERT zero-shot
  mon_job en2da-mbert-ft      # Follow English→Danish mBERT fine-tune

Useful SLURM commands:
  squeue -u eray              # View your jobs
  scancel <jobid>             # Cancel a job
  scancel -u eray -n "<name>" # Cancel by job name
  sinfo                       # View cluster info
  module avail                # List available modules
EOF
}

echo "✓ HPC commands loaded. Type 'hpc_help' for available functions."
