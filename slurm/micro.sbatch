#!/bin/bash
# Phase-1 microbenchmark. Usage: STAGE=p1_stage00_baseline ARGS="--engine naive" sbatch slurm/micro.sbatch
#SBATCH --job-name=agentopt-micro
#SBATCH --partition=freecycle-h100
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
: "${STAGE:?set STAGE}"
OUT=results/$STAGE
mkdir -p "$OUT"
(hostname; uptime; nvidia-smi) > "$OUT/node_micro.txt"
slurm/gpumon.sh "$OUT/gpumon_micro.csv" &
GPUMON=$!
trap "kill $GPUMON 2>/dev/null" EXIT
python bench/microbench.py --stage "$STAGE" ${ARGS:-}
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUT/micro_slurm.log"
