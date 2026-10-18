#!/bin/bash
# Per host: once at most 10 init jobs are left unstarted anywhere (done + running >= 4454) and this host runs no init
# process, raise its worker pool to 90% of its threads minus 1 for the worker's parent, by a graceful restart of that
# host alone (A4-D100). The last unstarted jobs are the slowest pair's chained epochs; one may still start on a raised
# host, where it runs at nice 0 over the nice-10 workers. With init nearly done, worker capacity sets the run's end.
# kaist140 is held at 100 by memory (A4-D101).
set -u
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16
cd /home/sibisibi/Projects/research/dexmanip/.output/051-grasp-poc
for hn in kaist67=56 kaist121=114 kaist140=100 kaist143=35 kaist144=35 kaist145=85 kaist148=114 kaist149=114; do
  h=${hn%%=*}
  (
    until ssh -o ControlPath=none $h "[ \$((\$(ls $S/gsched/ijob_done | wc -l) + \$(ls $S/gsched/ijob_lock | wc -l))) -ge 4454 ] && [ \$(pgrep -fc 'dexonomy[.]main op=init') -eq 0 ]"; do sleep 60; done
    echo "$(date -u +%TZ) $h: no init left for it; raising its pool"
    bash scripts/agent4_cap16_restart_workers.sh "$hn"
  ) &
done
wait
echo "$(date -u +%TZ) all pools raised"
