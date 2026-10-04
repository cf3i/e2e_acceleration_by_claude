#!/bin/bash
# Usage: STAGE=stage00_baseline ARGS="--engine naive --runner serial" sbatch slurm/bench.sbatch
#SBATCH --job-name=agentopt-bench
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
OUT=results/$STAGE
mkdir -p "$OUT"
(hostname; uptime; nvidia-smi) > "$OUT/node.txt"
slurm/gpumon.sh "$OUT/gpumon.csv" &
GPUMON=$!
trap "kill $GPUMON 2>/dev/null" EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt"
python bench/run_bench.py --stage "$STAGE" ${ARGS:-}
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUT/bench_slurm.log"
