#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <technique-number> [repo_dir]"
  echo "Example: $0 1 /home/you/Active-Querying-on-PEBBLE"
  exit 1
fi

TECHNIQUE="$1"                # e.g., 1 (Disagreement) or 2 (Entropy)
REPO_DIR="${2:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-rime2}"

# EDIT seeds here as needed
SEEDS=(12345 23451 34512 45123 51234)

# Baseline: no sampling -> leave SAMPLING empty
BASELINE_SAMPLING=""

echo "Repo: $REPO_DIR"
echo "Conda env: $CONDA_ENV"
echo "Technique (second batch): $TECHNIQUE"
echo "Seeds: ${SEEDS[*]}"

FIRST_JOBIDS=()
for s in "${SEEDS[@]}"; do
  echo "Submitting baseline job (seed=$s)..."
  jobid=$(sbatch --parsable \
    --export=REPO_DIR="$REPO_DIR",SEED="$s",SAMPLING="$BASELINE_SAMPLING",CONDA_ENV="$CONDA_ENV" \
    job_PEBBLE.sbatch)
  echo " -> baseline jobid: $jobid"
  FIRST_JOBIDS+=("$jobid")
done

# Build dependency string
dependency=$(IFS=:; echo "${FIRST_JOBIDS[*]}")
echo "First batch jobids: ${FIRST_JOBIDS[*]}"
echo "Using dependency afterok:${dependency}"

SECOND_JOBIDS=()
for s in "${SEEDS[@]}"; do
  echo "Submitting dependent job (technique=${TECHNIQUE}, seed=${s})..."
  jobid=$(sbatch --parsable \
    --dependency=afterok:"$dependency" \
    --export=REPO_DIR="$REPO_DIR",SEED="$s",SAMPLING="$TECHNIQUE",CONDA_ENV="$CONDA_ENV" \
    job_PEBBLE.sbatch)
  echo " -> dependent jobid: $jobid"
  SECOND_JOBIDS+=("$jobid")
done

echo "Submitted ${#FIRST_JOBIDS[@]} baseline jobs and ${#SECOND_JOBIDS[@]} dependent jobs."
echo "Baseline jobids: ${FIRST_JOBIDS[*]}"
echo "Dependent jobids: ${SECOND_JOBIDS[*]}"
