#!/bin/bash
# One full-set Isaac worker on one GPU, outside Slurm, pinned to its own cores. It walks the 3,200 cap16 objects in
# list order, dj forward and kait in reverse so the two meet in the middle, and claims each object by mkdir.
# Usage: reset_isaac_all.sh <gpu> <cores> <fwd|rev>
set -euo pipefail
GPU=$1 CORES=$2 ORDER=$3
if [ "$(hostname -s)" = login-0 ]; then  # kait login node, dj also has a /home/nas_main symlink
  R=/home/nas_main/sibeenkim/draftrec
  TREE=$R/repos/simtoolreal-050 SCR=$R/_curobo2/reset PY=$R/venv_isaacsim/bin/python
  POOL=$R/assets_040/objects_cap16 IK=$R/_graspoc/cap16/reset_full/ik GR=$R/_graspoc/cap16/reset_full/grasps
  OUT=$R/_graspoc/cap16/reset_full/isaac
  export LD_LIBRARY_PATH=$R/syslibs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
else  # dj
  W=/home/nas4_user/sibeenkim/work
  TREE=$W/simtoolreal-050 SCR=$W/_graspoc/reset PY=$W/_033_venv_isaacsim/bin/python
  POOL=$W/_graspoc/reset/pool_dj IK=$W/_graspoc/reset/ik_all GR=$W/_graspoc/dexonomy/cap16_deliver/dexonomy
  OUT=$W/_graspoc/reset/isaac_all
fi
export PYTHONPATH=$TREE:$SCR RESET_TREE=$TREE CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/tmp/sibeenkim-reset-isaac-$GPU
mkdir -p "$TMPDIR" "$OUT/claims" "$OUT/final" "$OUT/runs" "$OUT/logs"
H=$(hostname -s)
LOG=$OUT/logs/${H}_g${GPU}_c${CORES}_$$.log
STEMS=$(python3 -c "import json; s = sorted(e['urdf'].split('/')[-1][:-5] for e in json.load(open('$POOL/pool.json'))); print(' '.join(s if '$ORDER' == 'fwd' else s[::-1]))")
cd "$TREE"
for S in $STEMS; do
  mkdir "$OUT/claims/$S" 2>/dev/null || continue  # taken here or synced over from the other cluster
  echo "$(date -u +%FT%TZ) start $S" >> "$LOG"
  taskset -c "$CORES" "$PY" -u "$SCR/reset_isaac.py" --stem "$S" --pool-dir "$POOL" --grasps "$GR/$S.npz" --ik "$IK/$S.npz" \
    --final "$OUT/final/$S.npz" --start 0 --batches 1000 --target 10000 --slim --out "$OUT/runs/$S" >> "$LOG" 2>&1 || true  # the final-file check below decides
  # Isaac Sim can exit 0 without finishing, the file is the proof. A failed object is listed for a rerun on a
  # B200, since two runs on a 24 GB GPU can run out of memory on the heaviest objects.
  [ -e "$OUT/final/$S.npz" ] || echo "$(date -u +%FT%TZ) FAILED $S $H gpu $GPU" | tee -a "$OUT/failed.txt" >> "$LOG"
done
echo "$(date -u +%FT%TZ) worker done" >> "$LOG"
