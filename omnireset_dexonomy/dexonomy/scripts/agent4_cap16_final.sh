#!/bin/bash
# End of the cap16 run: wait until all 496 (stage, chunk, template) pairs are final and every chunk has EVAL_DONE,
# run the pre-delivery gate (agent4_cap16_predeliver.py), then the one delivery as 8 shards, one per host on its own
# NFS client, and print DONE's line. The gate and the shards run detached on dj (nohup), each writing its exit
# status to a file this script polls, so a VPN drop on this side cannot kill them halfway (A4-D103).
set -u
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16; PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
O=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/Dexonomy/output
HOSTS=(kaist67 kaist121 kaist140 kaist143 kaist144 kaist145 kaist148 kaist149)
until ssh -o ControlPath=none kaist140 "[ \$(ls $S/gsched/final | wc -l) -eq 496 ] && for h in ${HOSTS[*]}; do [ -e $O/cap16_\${h}_xhand/driver/EVAL_DONE ] || exit 1; done"; do sleep 60; done
echo "$(date -u +%TZ) all 496 pairs final, EVAL_DONE on all 8 chunks; pre-delivery gate starts"
ssh -o ControlPath=none kaist143 "rm -f $S/predeliver.rc; cd /tmp && nohup bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= $PY $S/agent4_cap16_predeliver.py > $S/predeliver.out 2>&1; echo \$? > $S/predeliver.rc' > /dev/null 2>&1 < /dev/null &"
until ssh -o ControlPath=none kaist143 "[ -e $S/predeliver.rc ]"; do sleep 30; done
rc=$(ssh -o ControlPath=none kaist143 "cat $S/predeliver.rc")
if [ "$rc" != 0 ]; then
  echo "$(date -u +%TZ) pre-delivery gate failed (exit $rc), no delivery:"; ssh -o ControlPath=none kaist143 "grep -v 'Warp CUDA' $S/predeliver.out | tail -20"; exit 1
fi
echo "$(date -u +%TZ) pre-delivery gate passed: $(ssh -o ControlPath=none kaist143 "grep -v 'Warp CUDA' $S/predeliver.out | tail -1 | cut -c1-300")"
for i in "${!HOSTS[@]}"; do
  ssh -o ControlPath=none ${HOSTS[$i]} "rm -f $S/deliver_final/shard_$i.rc; mkdir -p $S/deliver_final; cd /tmp && nohup bash -c 'OMP_NUM_THREADS=1 $PY $S/agent4_cap16_deliver_final.py $i 8 64 > $S/deliver_final/shard_$i.out 2>&1; echo \$? > $S/deliver_final/shard_$i.rc' > /dev/null 2>&1 < /dev/null &"
done
echo "$(date -u +%TZ) 8 delivery shards started detached"
until ssh -o ControlPath=none kaist140 "[ \$(ls $S/deliver_final/ | grep -c '^shard_[0-7].rc$') -eq 8 ]"; do sleep 60; done
bad=$(ssh -o ControlPath=none kaist140 "for f in $S/deliver_final/shard_*.rc; do [ \$(cat \$f) = 0 ] || echo \$f; done")
if [ -n "$bad" ]; then echo "$(date -u +%TZ) delivery shards failed: $bad"; exit 1; fi
ssh -o ControlPath=none kaist140 "cat /home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16_deliver/dexonomy/DONE"
