#!/bin/bash
# Correctness check of engine presets. Usage: slurm/submit.sh check <STAGE> --ref A --cand B,C
#   CHECK=check_batch slurm/submit.sh check <STAGE> --engine P   (slot-isolation check for batch engines)
#SBATCH --job-name=agentopt-check
#SBATCH --partition=freecycle-h100
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
: "${STAGE:?set STAGE}"
OUT=results/$STAGE
mkdir -p "$OUT"
(hostname; uptime; nvidia-smi) > "$OUT/node_check.txt"
slurm/gpumon.sh "$OUT/gpumon_check.csv" &
GPUMON=$!
trap "kill $GPUMON 2>/dev/null" EXIT
python bench/${CHECK:-check_equiv}.py --stage "$STAGE" ${ARGS:-}
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUT/${CHECK:-check}_slurm.log"
