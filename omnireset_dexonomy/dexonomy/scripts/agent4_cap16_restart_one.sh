#!/bin/bash
# Runs on the host itself, detached (A4-D103): drain this host's shared worker through its stop file, wait until it
# has closed its in-flight leases and exited, then start a new worker with <n> processes, and log <token>. Nothing
# here depends on the workstation's ssh session, so a VPN drop cannot leave the host without a worker or with two.
# A mkdir lock keeps a second launch for the same host from running at once.
# Usage: nohup bash agent4_cap16_restart_one.sh <host> <n proc> <token> &
h=$1; n=$2; token=$3
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16; PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
L=$S/gsched/log/restart_$h.log
mkdir $S/gsched/restart_lock_$h 2>/dev/null || { echo "$(date -u +%TZ) $token $h: another restart holds the lock" >> $L; exit 1; }
w=$(pgrep -f "^$PY $S/agent4_cap16_gworker.py $h " | while read p; do [ "$(ps -o ppid= -p $p | tr -d ' ')" = 1 ] && echo $p; done; true)
if [ $(echo $w | wc -w) -gt 1 ]; then
  echo "$(date -u +%TZ) $token $h: expected at most one worker, found '$w'" >> $L; rmdir $S/gsched/restart_lock_$h; exit 1
fi
if [ -n "$w" ]; then
  touch $S/gsched/stop/$h
  while kill -0 $w 2>/dev/null; do sleep 3; done
else
  w=none-dead
fi
rm -f $S/gsched/stop/$h
cd /tmp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= nohup nice -n 10 $PY $S/agent4_cap16_gworker.py $h $n $((40*n)) reuse >> $S/gsched/log/$h.v2.out 2>&1 < /dev/null &
echo "$(date -u +%TZ) $token $h: worker $w exited, new worker $! n=$n" >> $L
rmdir $S/gsched/restart_lock_$h
