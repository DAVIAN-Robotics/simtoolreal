#!/bin/bash
# nas4 client metrics for this host over <sec> seconds, from the per-op counters in /proc/self/mountstats (awk $2 ops,
# $7 queue ms, $8 RTT ms): files opened per second (OPEN plus OPEN_NOATTR), RPCs per second and per file, the mean
# number of RPCs on the wire and waiting in the client queue (Little's law on the RTT and queue sums), and the mean
# queue and RTT per op for OPEN, READ and DELEGRETURN.
# Usage: ssh <host> bash -s -- <sec> <host> < agent4_cap16_nfsrates.sh
t=$1; h=$2
snap(){ awk '/device <nas4 host>:\/volume2\/user /{f=1; next} /^device/{f=0} f && /^[[:space:]]+[A-Z_]+:/ && NF >= 9 {print $1, $2, $7, $8}' /proc/self/mountstats | sort; }
a=$(mktemp); b=$(mktemp); snap > $a; sleep $t; snap > $b
join $a $b | awk -v h=$h -v t=$t -v at="$(date -u +%TZ)" '
  {d = $5 - $2; q = $6 - $3; r = $7 - $4; n += d; Q += q; R += r
   if ($1 == "OPEN:" || $1 == "OPEN_NOATTR:") f += d
   if ($1 == "OPEN:" || $1 == "READ:" || $1 == "DELEGRETURN:") s = s sprintf(" %s%.0f/%.0f", $1, d ? q / d : 0, d ? r / d : 0)}
  END {printf "%s %s files/s=%.0f rpc/s=%.0f rpc_per_file=%.2f on_wire=%.1f in_client_queue=%.1f queue/rtt_ms%s\n",
       at, h, f / t, n / t, f ? n / f : 0, R / (t * 1000), Q / (t * 1000), s}'
rm -f $a $b
