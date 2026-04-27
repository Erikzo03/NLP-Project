# HPC First-Time Setup & Quick Start

## 1. SSH into HPC

```bash
ssh eray@hpc3.itu.dk
```

## 2. Navigate to project

```bash
cd NLP-Project
```

## 3. Initial setup (first time only)

### Load the Anaconda module
```bash
module purge
module load Anaconda3/2024.06-1
```

### Verify Python environment
```bash
python --version
which python
```

### Check required packages
```bash
python - <<'PY'
import torch, transformers, datasets, seqeval
print("✓ All packages OK")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"seqeval: {seqeval.__version__}")
PY
```

If any package is missing, ask your HPC admin to install it, or ask Akos/O'neal for the environment setup details.

### Update repo from GitHub
```bash
git status
git pull --rebase origin main
```

If you have local changes:
```bash
git stash
git pull --rebase origin main
git stash pop
```

## 4. Running the job pipeline

### Simple one-command submission
```bash
cd ~/NLP-Project/jobs
bash submit_all_pipeline.sh
```

This will:
1. Submit all 4 zero-shot jobs (parallel, independent)
2. Wait for them to complete, then submit full fine-tuning jobs
3. Submit all 12 data-budget fine-tuning variants (10%, 25%, 50%)
4. Submit 3 independent Danish-only baseline jobs
5. Show you the job IDs and monitoring commands

### After submission, check your queue
```bash
squeue -u eray
```

## 5. Monitoring jobs

### Load helper commands (optional but convenient)
```bash
source ~/NLP-Project/jobs/hpc_commands.sh
```

Then you can use:
```bash
check_env              # Verify environment
view_queue             # Show your job queue
view_history           # Recent jobs (last 24h)
mon_job en2da-mbert-zs # Follow a job's output
check_errors           # Find errors in logs
```

### Manual monitoring
```bash
# See your queue
squeue -u eray

# See completed jobs
sacct -u eray --format=JobID,JobName,State,ExitCode,Elapsed -S now-1day

# Follow a job's output (use JobID from squeue)
tail -f logs/en2da-mbert-zs-<JOBID>.out

# Check for errors
tail -50 logs/en2da-mbert-zs-<JOBID>.err

# All errors in one scan
grep "Traceback\|Error\|ModuleNotFoundError" logs/*.err
```

## 6. If a job fails

1. Find the job ID in `squeue -u eray` or `view_history`
2. Look at the error log:
   ```bash
   tail -100 logs/<jobname>-<jobid>.err
   ```
3. Common issues:
   - **ModuleNotFoundError**: Package not installed (ask admin)
   - **File not found**: Check dataset paths in `Datasets/`
   - **CUDA out of memory**: Try on different node (shouldn't happen with our settings)
   - **Model not found**: Checkpoint path is wrong (check output folder exists)

4. Resubmit after fixing:
   ```bash
   sbatch jobs/<jobname>.slurm
   ```

## 7. Cancelling jobs

### Cancel one job
```bash
scancel <jobid>
```

### Cancel by name pattern
```bash
scancel -u eray -n "en2da-mbert"  # Cancels all English→Danish mBERT jobs
```

### Cancel all your jobs
```bash
scancel -u eray
```

## 8. When all jobs are done

Check your outputs:
```bash
ls -lh outputs/
ls -lh outputs/*/dev_metrics.json
```

Each job produces:
- `dev_metrics.json` – validation F1, precision, recall
- `dev_predictions.iob2` – predictions on validation set
- `test_predictions.iob2` – predictions on test set

## Useful info

**Your HPC home:** `~/` or `/home/eray`
**Project folder:** `~/NLP-Project`
**Job logs:** `~/NLP-Project/logs/`
**Results:** `~/NLP-Project/outputs/`
**SLURM jobs:** `~/NLP-Project/jobs/`

**Module load (add to ~/.bashrc to auto-load)**:
```bash
module load Anaconda3/2024.06-1
```

**Check HPC status:**
```bash
sinfo
```

**Check your disk usage:**
```bash
du -sh ~
```

---

If you hit issues, check the job's error log first, then ask the team for help with the error message!
