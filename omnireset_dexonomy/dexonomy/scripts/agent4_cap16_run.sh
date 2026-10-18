#!/bin/bash
# One host's share of the cap16 Dexonomy run: the op sequence of dexonomy/script.py with upstream defaults
# (n_final 10, 1024 points x 8 spins, filters), all objects of this host's chunk (op.object.n_cfg=null),
# its own exp dir, and one new_tmpl_dir shared by every chunk so template evolution crosses objects.
#   init   one process per template, KPAR at once, spread over this host's free GPUs, retried until
#          'Finish initialization' (up to 30 tries, then the template is reported as unfinished)
#   grasp  op=grasp until 10 empty passes after all init, then one more pass
#   eval   op=eval until 10 empty passes after grasp is done, then one more pass
# Usage: agent4_cap16_run.sh <exp> <epochs> <gpus csv> <kpar> <grasp workers> <eval n_worker> <chunk scene_cfg glob> <new_tmpl_dir> <tmpl ...>
set -u
EXP=$1; EPOCH=$2; GPUS=$3; KPAR=$4; NWG=$5; NWE=$6; GLOB=$7; POOL=$8; shift 8
TMPLS=("$@")
D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy
PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
cd $D/Dexonomy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
COMMON="hand=xhand exp_name=$EXP new_tmpl_dir=$POOL"
OBJ1="op.object.cfg_path=\"$GLOB\""
OBJ2="op.object.n_cfg=null"
LOGROOT=output/${EXP}_xhand/log
MARK=output/${EXP}_xhand/driver
mkdir -p $MARK $POOL
rm -f $MARK/INIT_DONE $MARK/GRASP_DONE
IFS=, read -ra GPU <<< "$GPUS"
echo "$(date -u +%FT%TZ) start $EXP host=$(hostname) gpus=$GPUS kpar=$KPAR nwg=$NWG nwe=$NWE templates=${#TMPLS[@]}"

finished_init() { grep -q "Finish initialization" "$1" 2>/dev/null; }
finished_loop() { [ -f "$1" ] || return 1; [ "$(tail -n 10 "$1" | grep -c "$2")" -eq 10 ]; }

run_init() {  # $1 slot index (picks the GPU), $2 template; log_id is the taxonomy number
  local t=$2 g=${GPU[$(( $1 % ${#GPU[@]} ))]}
  local i=${t%%_*}
  local log=$LOGROOT/init_$i/main.log n=0
  while ! finished_init $log && [ $n -lt 30 ]; do
    $PY -m dexonomy.main op=init $COMMON tmpl_name=$t log_id=$i "init_gpu=[$g]" n_worker=4 op.epoch=$EPOCH "$OBJ1" "$OBJ2" > /dev/null 2>&1
    n=$((n+1))
  done
  if finished_init $log; then echo "$(date -u +%FT%TZ) init done $t gpu=$g tries=$n"; else echo "$(date -u +%FT%TZ) INIT_UNFINISHED $t gpu=$g tries=$n"; fi
}

(
  for idx in "${!TMPLS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$KPAR" ]; do sleep 5; done
    run_init $idx ${TMPLS[$idx]} &
  done
  wait
  touch $MARK/INIT_DONE
  echo "$(date -u +%FT%TZ) all init done"
) &
INIT_PID=$!

(
  sleep 30
  until [ -e $MARK/INIT_DONE ] && finished_loop $LOGROOT/grasp_0/main.log "Find 0 initialization"; do
    $PY -m dexonomy.main op=grasp $COMMON n_worker=$NWG > /dev/null 2>&1
    sleep 10
  done
  $PY -m dexonomy.main op=grasp $COMMON n_worker=$NWG > /dev/null 2>&1
  touch $MARK/GRASP_DONE
  echo "$(date -u +%FT%TZ) grasp loop done"
) &
GRASP_PID=$!

(
  sleep 60
  until [ -e $MARK/GRASP_DONE ] && finished_loop $LOGROOT/eval_0/main.log "Find 0 grasp data"; do
    $PY -m dexonomy.main op=eval $COMMON log_id=0 skip_traj=True n_worker=$NWE > /dev/null 2>&1
    sleep 10
  done
  $PY -m dexonomy.main op=eval $COMMON log_id=0 skip_traj=True n_worker=$NWE > /dev/null 2>&1
  touch $MARK/EVAL_DONE
  echo "$(date -u +%FT%TZ) eval loop done"
) &
EVAL_PID=$!

wait $INIT_PID $GRASP_PID $EVAL_PID
echo "$(date -u +%FT%TZ) PIPELINE_DONE $EXP"
