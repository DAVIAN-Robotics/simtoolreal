#!/bin/bash
# Relaunch one delivery shard with another pool size, keeping the entries it already wrote (A4-D104). Runs detached on
# the shard's host. Kills the shard's bash wrapper first, so it writes no exit status, then the python parent and its
# pool children, all by PID, waits until they are gone, and starts the shard again under the same wrapper, whose
# shard_<i>.rc agent4_cap16_final.sh polls. A temp file left by a killed child is rewritten under the same name.
# Usage: nohup bash agent4_cap16_relaunch_shard.sh <shard> <n proc> <token> > /dev/null 2>&1 < /dev/null &
i=$1; n=$2; token=$3
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16; PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
F=$S/deliver_final; L=$F/relaunch.log
w=$(pgrep -f "^bash -c OMP_NUM_THREADS=1 $PY $S/agent4_cap16_deliver_final.py $i 8 ")
p=$(pgrep -f "^$PY $S/agent4_cap16_deliver_final.py $i 8 ")
parent=$(for x in $p; do [ "$(ps -o ppid= -p $x | tr -d ' ')" = "$w" ] && echo $x; done)
if [ $(echo $w | wc -w) -ne 1 ] || [ $(echo $parent | wc -w) -ne 1 ]; then
  echo "$(date -u +%TZ) $token shard $i: expected one wrapper and one python under it, found wrapper '$w' parent '$parent'" >> $L; exit 1
fi
kill -9 $w
kill -9 $p
for x in $p; do while kill -0 $x 2>/dev/null; do sleep 2; done; done
rm -f $F/shard_$i.rc
cd /tmp
nohup bash -c "OMP_NUM_THREADS=1 $PY $S/agent4_cap16_deliver_final.py $i 8 $n >> $F/shard_$i.out 2>&1; echo \$? > $F/shard_$i.rc" > /dev/null 2>&1 < /dev/null &
echo "$(date -u +%TZ) $token shard $i: killed wrapper $w and $(echo $p | wc -w) python processes, relaunched as $! with $n processes" >> $L
