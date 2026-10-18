#!/usr/bin/env bash
# The EigenDEXplore runs on yj, tactile none, cap16, fr3-xhand-a4hs, in an eigen tree whose own patched rl_games goes
# first on PYTHONPATH. table starts on the table from the unidexgrasp-eigen tree, reset starts from the cap16 grasped
# reset states from the unidexgrasp-reset-eigen tree. The global curriculum runs on yj-3 as minhopark, the per-object
# curriculum (cpo) on yj-4 as sibeenkim. Noise joint is plain IID joint noise, egosuite and dexonomy are EigenDEXplore
# with that basis.
#
#   ssh yj-3-A6000-mhp bash -s -- <gpu> global <joint|egosuite|dexonomy> <table|reset> < launch_eigen_yj.sh
#   ssh yj-4-A6000     bash -s -- <gpu> cpo    <joint|egosuite|dexonomy> <table|reset> < launch_eigen_yj.sh
set -euo pipefail

USAGE="usage: launch_eigen_yj.sh <gpu> <global|cpo> <joint|egosuite|dexonomy> <table|reset>"
GPU="${1:?$USAGE}"
CUR="${2:?$USAGE}"
NOISE="${3:?$USAGE}"
START="${4:?$USAGE}"
ME=$(id -un)

POOL=/home/nas5/minhopark/assets_039/objects_cap16
BASES=/home/nas5/minhopark/_053/eigen
STATES=/home/nas5/minhopark/_053/reset_full_final
if [ "$CUR" = global ]; then
  test "$ME" = minhopark || { echo "global runs as minhopark, this is $ME"; exit 1; }
  REPOS=/home/nas5/minhopark/repos
  PY=/home/nas5/minhopark/repos/simtoolreal/.venv_isaacsim/bin/python
  CACHE=/home/nas5/minhopark/assets_039/usd_cache_objects_cap16
elif [ "$CUR" = cpo ]; then
  test "$ME" = sibeenkim || { echo "cpo runs as sibeenkim, this is $ME"; exit 1; }
  REPOS=/home/nas5/sibeenkim/work
  PY=/home/nas5/sibeenkim/work/simtoolreal/.venv_isaacsim/bin/python
  CACHE=/home/nas5/sibeenkim/work/_040-pool/usd_cache_objects_cap16
else
  echo "$USAGE"; exit 1
fi
NAME=unidexgrasp
TASK=""
if [ "$START" = table ]; then
  TREE=$REPOS/simtoolreal-unidexgrasp-eigen
  WANT_TREE=dbdef5f
elif [ "$START" = reset ]; then
  TREE=$REPOS/simtoolreal-unidexgrasp-reset-eigen
  WANT_TREE=046ea5d
  test "$(ls $STATES | wc -l)" -eq 3200 || { echo "$STATES does not hold 3200 files"; exit 1; }
  NAME=$NAME-reset_dexonomy
  TASK="env.reset.reset_state_dir=$STATES"
else
  echo "$USAGE"; exit 1
fi
if [ "$CUR" = cpo ]; then
  NAME=$NAME-cur_per_obj
  TASK="$TASK env.termination.tolerance_curriculum_per_object=True"
fi
if [ "$NOISE" = joint ]; then
  EIGEN=""
elif [ "$NOISE" = egosuite ] || [ "$NOISE" = dexonomy ]; then
  BASIS=$BASES/eigen_${NOISE}_xhand.npz
  test -f "$BASIS" || { echo "no basis at $BASIS"; exit 1; }
  EIGEN="--eigendexplore_basis $BASIS"
  NAME=$NAME-eigen_$NOISE
else
  echo "$USAGE"; exit 1
fi

test -f "$POOL/pool.json" || { echo "no staged pool at $POOL"; exit 1; }
test -x "$PY" || { echo "no interpreter at $PY"; exit 1; }
test "$(git -C $TREE rev-parse --short HEAD^{tree})" = "$WANT_TREE" || { echo "tree is not $WANT_TREE"; exit 1; }
test ! -e "$TREE/runs/$NAME" || { echo "run dir $TREE/runs/$NAME exists"; exit 1; }
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
test "$busy" -eq 0 || { echo "gpu $GPU has $busy processes on it, refusing"; exit 2; }

TMP=/tmp/$ME-055-$START-$CUR-$NOISE
echo "tree   $TREE at $(git -C $TREE log --oneline -1)   tree hash $(git -C $TREE rev-parse HEAD^{tree})"
echo "run    $NAME   gpu $GPU   ${EIGEN:-joint noise}   $TASK"
cd "$TREE"
mkdir -p runs "$TMP"

tmux new-session -d -s "vram_$NAME" bash -c "
while true; do
  echo \"\$(date +%H:%M:%S) \$(nvidia-smi -i $GPU --query-gpu=memory.used,memory.free --format=csv,noheader)\"
  sleep 20
done > $TREE/runs/$NAME.vram.log 2>&1
"

tmux new-session -d -s "simtoolreal_$NAME" bash -c "
export PYTHONPATH=$TREE:$TREE/rl_games OMNI_KIT_ACCEPT_EULA=YES TMPDIR=$TMP
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $TREE
echo \"rl_games \$($PY -c 'import rl_games; print(rl_games.__file__)')\" > runs/$NAME.launch.log
CUDA_VISIBLE_DEVICES=$GPU $PY -u isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --wandb_activate --wandb_project simtoolreal --wandb_entity draftrec \
  --wandb_name $NAME $EIGEN \
  hydra.run.dir=runs/$NAME \
  agent.params.config.save_frequency=1000 \
  env.robot=fr3-xhand-a4hs \
  env.assets.object_pool_dir=$POOL \
  env.assets.asset_cache_dir=$CACHE \
  $TASK >> runs/$NAME.launch.log 2>&1
"
echo "launched in tmux simtoolreal_$NAME, log $TREE/runs/$NAME.launch.log"
