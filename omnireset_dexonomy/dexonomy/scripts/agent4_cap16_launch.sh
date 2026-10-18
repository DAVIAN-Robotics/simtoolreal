#!/bin/bash
# Launch one host's cap16 chunk from plan.json, after checking with nvidia-smi and gpustat that every planned GPU
# is still free (no compute process, under 500 MiB). Run on that host. Usage: agent4_cap16_launch.sh <host alias>
set -euo pipefail
H=$1
D=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy
cd $D/cap16
read -r GPUS KPAR NWG NWE GLOB <<< "$(python3 -c "
import json; h=[x for x in json.load(open('plan.json'))['hosts'] if x['host']=='$H'][0]
print(','.join(map(str,h['gpus'])), h['kpar'], h['nwg'], h['nwe'], h['glob'])")"
gpustat --no-color | tail -n +2 | cut -c1-80
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
for g in ${GPUS//,/ }; do
  read -r u m <<< "$(nvidia-smi -i $g --query-gpu=uuid,memory.used --format=csv,noheader,nounits | tr -d ',')"
  if echo "$busy" | grep -q "$u" || [ "$m" -ge 500 ]; then echo "GPU $g on $H is taken, not launching"; exit 1; fi
done
TMPLS=$(ls $D/Dexonomy/assets/hand/xhand/init_tmpl | sed 's/\.npy$//' | sort -t_ -k1,1n | tr '\n' ' ')
[ "$(echo $TMPLS | wc -w)" -eq 31 ]
nohup bash $D/cap16/agent4_cap16_run.sh cap16_$H 5 $GPUS $KPAR $NWG $NWE "$GLOB" $D/cap16/new_tmpl_pool $TMPLS > $D/cap16/run_$H.log 2>&1 < /dev/null &
echo "$H launched driver pid $! gpus=$GPUS kpar=$KPAR nwg=$NWG nwe=$NWE at $(date -u +%FT%TZ)"
