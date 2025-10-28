#!/bin/bash
#SBATCH --job-name=PEBBLE
#SBATCH --output=logs/pebble_%x_%j.out
#SBATCH --error=logs/pebble_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=9:00:00
##SBATCH --gres=gpu:1
##SBATCH --mem=32G

# --- Exported variables expected:
# REPO_DIR  - path to repo (defaults to cwd)
# SEED      - integer seed for this run
# SAMPLING  - sampling scheme (0=uniform,1=disagreement,2=entropy) OR empty for baseline
# CONDA_ENV - conda env name (default: rime2)

: "${REPO_DIR:=$(pwd)}"
: "${SEED:=12345}"
: "${SAMPLING:=}"
: "${CONDA_ENV:=rime2}"

echo "==== SLURM job starting ===="
echo "Job: $SLURM_JOB_ID"
echo "Repo: $REPO_DIR"
echo "Seed: $SEED"
echo "Sampling: ${SAMPLING:-<none>}"
echo "Conda env: $CONDA_ENV"
echo "============================"

# Activate conda (robust)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "$CONDA_ENV" || {
    echo "Failed to activate conda env: $CONDA_ENV"
    exit 1
  }
else
  echo "conda not found. Make sure you load module or set PATH."
  # exit 1
fi

cd "$REPO_DIR" || { echo "Cannot cd to $REPO_DIR"; exit 1; }

# Export env vars so run_PEBBLE picks them up
export SEED
export SAMPLING

# call the run script (it lives inside repo's scripts/...)
# pass positional args: sampling and seed (both optional)
./scripts/walker_walk/500/oracle/run_PEBBLE.sh "${SAMPLING}" "${SEED}"

exit_code=$?
echo "Job finished with exit code: $exit_code"
exit $exit_code
