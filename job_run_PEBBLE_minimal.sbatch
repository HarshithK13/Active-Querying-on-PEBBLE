#!/bin/bash
export PYTHONUNBUFFERED=1
#SBATCH --job-name=PEBBLE_sampling
## SBATCH --output=logs/pebble_%j.out
## SBATCH --error=logs/pebble_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
# #SBATCH -A sabymath0


ENV_NAME="${ENV_NAME:-walker_walk}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
mkdir -p "$REPO_DIR/logs"
TEACHER_TYPE="${TEACHER_TYPE:-oracle}"
MAX_BUDGET="${MAX_BUDGET:-500}"

POS_ARG="${1:-}"
SAMPLING="${POS_ARG:-${SAMPLING:-}}"
POS_SEED="${2:-}"
if [[ -n "$POS_SEED" ]]; then
  export SEED="$POS_SEED"
fi

RUN_SCRIPT="${REPO_DIR}/scripts/${ENV_NAME}/${MAX_BUDGET}/${TEACHER_TYPE}/run_PEBBLE.sh"

if [[ -n "$SAMPLING" ]]; then
  exec "$RUN_SCRIPT" "$SAMPLING"
else
  exec "$RUN_SCRIPT"
fi


echo "Job: $SLURM_JOB_ID"
echo "Repo: $REPO_DIR"
echo "Target script: $RUN_SCRIPT"
echo "Sampling technique: ${SAMPLING:-<baseline/no sampling>}"
echo "--------------------------------"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "ERROR: target script not found or not executable: $RUN_SCRIPT"
  echo "Make it executable with: chmod +x $RUN_SCRIPT"
  exit 1
fi

cd "$REPO_DIR" || { echo "Cannot cd to $REPO_DIR"; exit 1; }

if [[ -n "$SAMPLING" ]]; then
  echo "Running with sampling technique: $SAMPLING"
  exec "$RUN_SCRIPT" "$SAMPLING"
else
  echo "Running baseline (no sampling technique)"
  exec "$RUN_SCRIPT"
fi

