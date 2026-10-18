#!/bin/bash
# Rerun one template's init on one chunk after the driver gave up on it (INIT_UNFINISHED): the driver's exact
# command, on this host's GPU with the most free memory and no other user's process, retried until
# 'Finish initialization' with 30 s between tries (the driver's loop had no pause, so a full GPU burned all 30
# tries in seconds). skip_done resumes. Logs to cap16/run_<host>.log. Usage: agent4_cap16_rerun_init.sh <host> <tmpl>
set -u
H=$1; T=$2
D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy
PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
cd $D/Dexonomy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
GLOB=$(python3 -c "import json; print([x for x in json.load(open('$D/cap16/plan.json'))['hosts'] if x['host']=='$H'][0]['glob'])")
GPUS=$(python3 -c "import json; print(' '.join(map(str, [x for x in json.load(open('$D/cap16/plan.json'))['hosts'] if x['host']=='$H'][0]['gpus'])))")
i=${T%%_*}
log=output/cap16_${H}_xhand/log/init_$i/main.log
n=0
until grep -q "Finish initialization" $log 2>/dev/null || [ $n -ge 30 ]; do
  others=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read p u; do [ "$(ps -o user= -p $p)" != sibeenkim ] && echo $u; done | sort -u)
  g=$(for x in $GPUS; do u=$(nvidia-smi -i $x --query-gpu=uuid --format=csv,noheader); echo "$others" | grep -q "$u" || echo "$(nvidia-smi -i $x --query-gpu=memory.free --format=csv,noheader,nounits) $x"; done | sort -rn | head -1 | cut -d' ' -f2)
  echo "$(date -u +%FT%TZ) rerun init $T try $((n+1)) gpu=$g"
  $PY -m dexonomy.main op=init hand=xhand exp_name=cap16_$H new_tmpl_dir=$D/cap16/new_tmpl_pool tmpl_name=$T log_id=$i "init_gpu=[$g]" n_worker=4 op.epoch=5 "op.object.cfg_path=\"$GLOB\"" op.object.n_cfg=null > /dev/null 2>&1
  n=$((n+1))
  grep -q "Finish initialization" $log || sleep 30
done
if grep -q "Finish initialization" $log; then echo "$(date -u +%FT%TZ) rerun init done $T tries=$n"; else echo "$(date -u +%FT%TZ) RERUN_INIT_UNFINISHED $T tries=$n"; fi
