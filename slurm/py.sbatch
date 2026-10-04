#!/bin/bash
# Run any python script on one H100. Usage: slurm/submit.sh py <STAGE> bench/<script>.py [args...]
#SBATCH --job-name=agentopt-py
#SBATCH --partition=freecycle-h100
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
: "${STAGE:?set STAGE}"
mkdir -p "results/$STAGE"
python ${ARGS}
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "results/$STAGE/$(basename ${ARGS%% *} .py)_slurm.log"
