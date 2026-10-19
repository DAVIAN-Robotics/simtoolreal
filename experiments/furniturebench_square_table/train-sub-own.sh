#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --account=sub
#SBATCH --qos=sub-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=furniturebench-square_table
#SBATCH --output=slurm-%j.log
set -euo pipefail
# sbatch copies this script; resolve the checkout using the submission directory.
TREE=${SIMTOOLREAL_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
export TMPDIR="/tmp/furniturebench-square-table-${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
# Slurm owns CUDA_VISIBLE_DEVICES. Keep its assigned device unchanged.
exec bash "$TREE/experiments/furniturebench_square_table/train.sh" "$@"
