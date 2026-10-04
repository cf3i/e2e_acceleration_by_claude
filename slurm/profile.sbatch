#!/bin/bash
# L2 (torch.profiler) + L3 (nsys) profiles.
# Usage: STAGE=stage00_baseline ARGS="--engine naive --runner serial" sbatch slurm/profile.sbatch
#SBATCH --job-name=agentopt-profile
#SBATCH --partition=freecycle-h100
#SBATCH --qos=freecycle
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
: "${STAGE:?set STAGE}"
P=results/$STAGE/profile
mkdir -p "$P"
(hostname; uptime; nvidia-smi) > "$P/node.txt"
slurm/gpumon.sh "$P/gpumon.csv" &
GPUMON=$!
trap "kill $GPUMON 2>/dev/null" EXIT
TORCH_TASKS=${TORCH_TASKS:-1}
NSYS_TASKS=${NSYS_TASKS:-4}

echo "== torch.profiler"
python bench/profile_torch.py --stage "$STAGE" --tasks "$TORCH_TASKS" ${ARGS:-}

echo "== nsys"
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o "$P/nsys" \
  python bench/run_bench.py --stage "$STAGE" --num-tasks "$NSYS_TASKS" --out "$P/nsys_run" ${ARGS:-}
nsys stats --force-export=true --format csv --output "$P/nsys" \
  --report cuda_gpu_kern_sum,nvtx_sum,cuda_api_sum,cuda_gpu_mem_time_sum "$P/nsys.nsys-rep" > /dev/null
python bench/analyze_nsys.py "$P" | tee "$P/nsys_summary.txt"
cp "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$P/profile_slurm.log"
