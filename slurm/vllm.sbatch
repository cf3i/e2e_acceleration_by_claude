#!/bin/bash
# vLLM reference run (own venv). Usage: slurm/submit.sh vllm <STAGE> [run_vllm.py args...]
#SBATCH --job-name=agentopt-vllm
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
source /mnt/data/u/feic/pjs/venvs/vllm/bin/activate
: "${STAGE:?set STAGE}"
OUT=results/$STAGE
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt"
slurm/gpumon.sh "$OUT/gpumon.csv" &
GPUMON=$!
trap "kill $GPUMON 2>/dev/null" EXIT
python bench/run_vllm.py --stage "$STAGE" ${ARGS:-}
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUT/bench_slurm.log"
