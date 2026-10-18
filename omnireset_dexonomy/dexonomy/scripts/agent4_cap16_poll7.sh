#!/bin/bash
# 40-epoch continuation, every 4 minutes (runners exit once all 4464 init jobs are done, 14:26Z, so runner=0 no longer alerts): per host the init runner and shared worker alive, init processes running,
# init retries and failed jobs in the runner log, WARNING lines in the worker log (upstream's safe_wrapper logs a
# swallowed per-item exception, a failed save included, as a WARNING), worker tracebacks, load; then the indexer's
# last cycle and the last progress line. Rewrites status/agent4.json, prints one line per poll and ALERT lines.
ST=/home/sibisibi/Projects/research/dexmanip/.output/051-grasp-poc/src/agent4_status.py
S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16
while true; do
  R=$(for h in kaist67 kaist121 kaist140 kaist143 kaist144 kaist145 kaist148 kaist149; do
    timeout 60 ssh -o ControlPath=none -o ConnectTimeout=10 $h "bash -s $h" <<'EOS' &
h=$1; S=/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16; PY=/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python
orphan() { for p in $(pgrep -f "^$PY $S/$1 $h "); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = 1 ] && echo $p; done | wc -l; }
w=$(orphan agent4_cap16_gworker.py); r=$(orphan agent4_cap16_init_runner.py)
ninit=$(pgrep -f "dexonomy.main op=init hand=xhand" | wc -l)
retry=$(grep -c "failed after" $S/gsched/log/init_$h.log); fail=$(grep -c "FAILED 30 tries" $S/gsched/log/init_$h.log)
warn=$(grep -c "\]\[WARNING\]" $S/gsched/log/$h.log); tb=$(grep -c Traceback $S/gsched/log/$h.v2.out)
wd=$( [ -e $S/gsched/watchdog_$h.pid ] && kill -0 $(cat $S/gsched/watchdog_$h.pid) 2>/dev/null && echo 1 || echo 0)
kills=$(grep -c KILLED $S/gsched/log/watchdog_$h.log 2>/dev/null)
base=$( [ $h = kaist143 ] && echo 1 || echo 0 )  # kaist143: the 02:33 FileNotFoundError, re-tested before delivery (A4-D80)
mem=$(free -g | awk '/Mem/{print $7}')  # available GB; kaist140 ran out at 184 workers (A4-D101)
echo "$h worker=$w runner=$r watchdog=$wd init_failed=$fail new_item_warnings=$((warn - base)) init_procs=$ninit init_retries=$retry watchdog_kills=$kills worker_tracebacks=$tb load=$(cut -d' ' -f1 /proc/loadavg) mem_avail_gb=$mem"
EOS
  done; wait)
  X=$(timeout 60 ssh -o ControlPath=none kaist140 "tail -n 1 $S/gsched/log/indexer.log | cut -c2-160; grep -c Traceback $S/gsched/log/indexer.out; tail -n 1 $S/progress.txt")
  python3 $ST notes="cap16 40-epoch poll $(date -u +%H:%M:%SZ): $(echo "$R" | sort | tr '\n' '|') indexer/progress: $(echo "$X" | tr '\n' '|')" > /dev/null
  echo "$(date -u +%H:%M:%SZ) $(echo "$X" | sed -n 3p | cut -c1-600)"
  echo "$R" | awk '{split($12, m, "="); low = (m[2] + 0 < 20)} $2!="worker=1" || $4!="watchdog=1" || $5!="init_failed=0" || $6!="new_item_warnings=0" || low {print "ALERT " $0}'
  # baseline 2: the indexer crashes of 06:17 (A4-D93) and 11:13 (A4-D102); a new one makes it 3
  echo "$X" | sed -n 2p | awk '$1!="2" {print "ALERT indexer tracebacks " $1}'
  timeout 60 ssh -o ControlPath=none kaist140 "cat $S/gsched/log/watchdog_*.log | grep KILLED | awk '\$1 > \"'$(date -u -d '5 min ago' +%FT%TZ)'\"'" 2>/dev/null | sed 's/^/WATCHDOG /'
  sleep 240
done
