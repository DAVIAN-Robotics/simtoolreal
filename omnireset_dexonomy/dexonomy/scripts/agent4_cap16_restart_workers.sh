#!/bin/bash
# Restart hosts' shared workers on the current agent4_cap16_gworker.py, hosts in parallel. Each restart runs detached
# on its host (agent4_cap16_restart_one.sh: stop file, wait for the worker to close its leases and exit, start the
# new one), so a VPN drop mid-restart cannot leave a host without a worker or with two (A4-D103). This script only
# launches it, retrying the launch until ssh succeeds, and waits for the host's log to show the launch's token.
# Usage: agent4_cap16_restart_workers.sh "<host>=<n proc> ..."
set -u
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16
for hn in $1; do
  h=${hn%%=*}; n=${hn##*=}
  (
    token="r$(date -u +%s%N)"
    until ssh -o ControlPath=none -o ConnectTimeout=10 $h "cd /tmp; nohup bash $S/agent4_cap16_restart_one.sh $h $n $token > /dev/null 2>&1 < /dev/null &"; do sleep 20; done
    until line=$(ssh -o ControlPath=none -o ConnectTimeout=10 $h "grep ' $token ' $S/gsched/log/restart_$h.log" 2>/dev/null) && [ -n "$line" ]; do sleep 5; done
    echo "$line"
  ) &
done
wait
