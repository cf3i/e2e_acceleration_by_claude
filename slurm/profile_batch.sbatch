#!/bin/bash
# Steady-state batched decode step: timing + node-level nsys of a few graph replays.
# Usage: slurm/submit.sh profile_batch <STAGE> --engine P [--live N --tag _bN]  (nsys goes to profile<tag>/)
#SBATCH --job-name=agentopt-profile-batch
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
TAG=$(echo "${ARGS:-}" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
P=results/$STAGE/profile$TAG
mkdir -p "$P"
(hostname; uptime; nvidia-smi) > "$P/node.txt"
nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --cpuctxsw=none --force-overwrite=true -o "$P/nsys" \
  python bench/profile_batch.py --stage "$STAGE" ${ARGS:-}
rm -f "$P"/nsys_*.csv
nsys stats --force-export=true --force-overwrite=true --format csv --output "$P/nsys" \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "$P/nsys.nsys-rep" > /dev/null
python bench/analyze_nsys.py "$P" | tee "$P/nsys_summary.txt"
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$P/profile_slurm.log"
