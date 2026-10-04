#!/bin/bash
# Submit a job from an immutable snapshot of the committed code, so later edits/installs can't leak into
# queued jobs. Results land in the main results/ via symlink.
#   slurm/submit.sh <bench|profile|micro> <STAGE> [python args...]
set -euo pipefail
kind=$1; stage=$2; shift 2
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain -- agentopt bench slurm data)" ]; then
  echo "error: uncommitted changes in agentopt/bench/slurm/data; commit first" >&2; exit 1
fi
rev=$(git rev-parse HEAD)
snap=runs/${stage}-${kind}-${rev:0:7}-$(date +%m%d-%H%M%S)
mkdir -p "$snap/slurm/logs"
git archive HEAD | tar -x -C "$snap"
echo "$rev" > "$snap/REVISION"
rm -rf "$snap/results"
ln -s "$PWD/results" "$snap/results"
ln -s "$PWD/.venv" "$snap/.venv"
cd "$snap" && STAGE="$stage" ARGS="$*" sbatch --parsable "slurm/$kind.sbatch" | sed "s|$| $snap|"
