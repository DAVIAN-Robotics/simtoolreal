#!/bin/bash
# Run agent4_dj_host_check.py on every alive Daejeon host, each on one GPU that is free right now
# (no compute process, under 500 MiB and 5 percent util per nvidia-smi, shown by gpustat first).
D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy
for h in kaist67 kaist120 kaist121 kaist126 kaist139 kaist140 kaist143 kaist144 kaist145 kaist146 kaist148 kaist149; do
  out=$(timeout 240 ssh -o ControlPath=none -o ConnectTimeout=12 $h "
    gpustat --no-color 2>/dev/null | tail -n +2 | tr -s ' ' | cut -c1-90
    busy=\$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    g=\$(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read i u m t; do
          echo \"\$busy\" | grep -q \"\$u\" && continue; [ \$m -lt 500 ] && [ \$t -lt 5 ] && echo \$i && break; done)
    [ -n \"\$g\" ] || { echo NO_FREE_GPU; exit 0; }
    cd /tmp && CUDA_VISIBLE_DEVICES=\$g OMP_NUM_THREADS=1 /home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python $D/agent4_dj_host_check.py 2>&1 | tail -3
  " 2>&1)
  echo "=== $h"
  echo "$out" | tail -4
done
