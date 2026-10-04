#!/bin/bash
# Hold one H100 for interactive development: run commands inside with
#   srun --jobid=<id> --overlap bash -lc 'cd <repo> && source .venv/bin/activate && python ...'
#SBATCH --job-name=agentopt-dev
#SBATCH --partition=freecycle-h100
#SBATCH --qos=freecycle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
sleep infinity
