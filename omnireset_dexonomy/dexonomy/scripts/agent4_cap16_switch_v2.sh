#!/bin/bash
# Switch the shared workers from v1 (listing scans) to v2 (queues): wait until every v1 worker has exited after
# its stop file, bootstrap each chunk's own-test queue, pass list and counters on that chunk's host in parallel,
# remove the stop files, then start one v2 worker per host with refinement model reuse.
set -u
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16; PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
declare -A W=([kaist67]=1238117 [kaist121]=1007117 [kaist140]=3969582 [kaist143]=4071386 [kaist144]=2493069 [kaist145]=1476113 [kaist148]=2917894 [kaist149]=2388232)
declare -A N=([kaist67]=60 [kaist121]=108 [kaist140]=220 [kaist143]=28 [kaist144]=29 [kaist145]=67 [kaist148]=110 [kaist149]=108)
for h in "${!W[@]}"; do
  while ssh -o ControlPath=none $h "kill -0 ${W[$h]} 2>/dev/null"; do sleep 5; done
  echo "$(date -u +%TZ) $h v1 worker ${W[$h]} exited"
done
for h in "${!W[@]}"; do
  ssh -o ControlPath=none $h "cd /tmp && $PY $S/agent4_cap16_bootstrap.py $h 64 2>&1 | grep -v 'Warp CUDA'" &
done
wait
ssh -o ControlPath=none kaist140 "rm -f $S/gsched/stop/*"
for h in "${!N[@]}"; do
  n=${N[$h]}
  ssh -o ControlPath=none $h "bash -c 'cd /tmp; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= nohup $PY $S/agent4_cap16_gworker.py $h $n $((40*n)) reuse > $S/gsched/log/$h.v2.out 2>&1 < /dev/null & echo $h v2 worker pid \$! n=$n at \$(date -u +%TZ)'"
done
