#!/bin/bash
# One S5 Isaac launch on a kait login GPU, outside Slurm, pinned to its own cores.
# Usage: reset_isaac.sh <gpu> <cores> <start row> <batches> <out dir> [num envs]
set -euo pipefail
ROOT=/home/nas_main/sibeenkim/draftrec
TREE=$ROOT/repos/simtoolreal-050
GPU=$1 CORES=$2 START=$3 BATCHES=$4 OUT=$5 ENVS=${6:-2048}
export PYTHONPATH=$TREE:$ROOT/_curobo2/reset
export CUDA_VISIBLE_DEVICES=$GPU
export LD_LIBRARY_PATH=$ROOT/syslibs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TMPDIR=/tmp/sibeenkim-reset-isaac-$GPU
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
cd "$TREE"
exec taskset -c "$CORES" "$ROOT/venv_isaacsim/bin/python" -u "$ROOT/_curobo2/reset/reset_isaac.py" \
  --start "$START" --batches "$BATCHES" --num-envs "$ENVS" --out "$OUT" ${DIAG:-} ${EXTRA:-}
