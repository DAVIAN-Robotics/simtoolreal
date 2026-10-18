"""Per-host control of the cap16 final delivery on all 8 dj hosts, on Sibeen's order of 2026-09-29 (A4-D112). There is
never a global stop, and only DONE ends the run. Runs detached on kaist143, beside the fresh-name probe and one of the 8
reader hosts, and drives agent4_cap16_deliver_final.py shards over ssh by IP.

Each host runs one of 16 shards at a time and takes the next pending shard when its shard exits 0. Every 15 s each
reading host is sampled. When a host completes under 5% of its previous NFS calls per busy process for 30 s, or cannot
be sampled for 60 s, only that host's readers are killed by PID. The shard stays with the host and is relaunched at 12
processes as soon as a fresh-name lookup from that host answers under 50 ms. After 5 healthy minutes at 12 it is
relaunched at 24, unless the host has frozen 3 times, which keeps it at 12. A shard that exits nonzero is relaunched on
its host up to 3 times. Every kill and relaunch goes to RAMP_STATUS.txt with host, files/s and time. After DONE the
controller runs the verification on kaist121.
Usage: python3 agent4_cap16_hostctl.py
"""
import collections
import datetime
import os
import queue
import subprocess
import threading
import time

S = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16"
F = f"{S}/deliver_final"
DEL = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16_deliver/dexonomy"
PY = "/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python"
HOSTS = ["kaist140", "kaist121", "kaist148", "kaist149", "kaist145", "kaist144", "kaist67", "kaist143"]
IP = {h: "143.248.159." + h[len("kaist"):] for h in HOSTS}
N_SHARD, FULL, REDUCED = 16, 24, 12
PROBE_LOG, LOG, STATUS = "/tmp/a4_nas4probe.log", "/tmp/a4_hostctl.log", f"{F}/RAMP_STATUS.txt"
TICK, WINDOW, LOG_EVERY = 15, 300, 60
FROZEN_FRAC, FROZEN_S, UNREACHABLE_S, LOOKUP_OK_MS, HEALTHY_S, MAX_FREEZES, MAX_CRASHES = 0.05, 30, 60, 50, 300, 3, 3

SAMPLE = r"""cd /tmp
for p in $(pgrep -f '^bash -c OMP_NUM_THREADS=1 PY S/agent4_cap16_deliver_final.py '); do echo W $(tr '\0' ' ' < /proc/$p/cmdline | awk '{print $6, $7, $8}'); done
echo B $(ps -o stat= -p "$(pgrep -d, -f '^PY S/agent4_cap16_deliver_final.py ')" 2>/dev/null | grep -c '^[DR]')
awk '/device <nas4 host>:\/volume2\/user /{f=1; next} /^device/{f=0}
     f && /^[[:space:]]+[A-Z_]+:/ && NF >= 9 {n += $2; q += $7; r += $8
       if ($1 == "OPEN:" || $1 == "OPEN_NOATTR:") o += $2
       if ($1 == "OPEN:") {on = $2; ort = $8}
       if ($1 == "DELEGRETURN:") {dn = $2; dq = $7}}
     END {print "M", n, q, r, o, on, ort, dn, dq}' /proc/self/mountstats
""".replace("PY", PY).replace(" S/", f" {S}/")
KILL = r"""cd /tmp
w=$(pgrep -f '^bash -c OMP_NUM_THREADS=1 PY S/agent4_cap16_deliver_final.py @I@ @N@ ')
p=$(pgrep -f '^PY S/agent4_cap16_deliver_final.py @I@ @N@ ')
[ -n "$w" ] && kill -9 $w
[ -n "$p" ] && kill -9 $p
sleep 3
echo killed wrapper $w python $(echo $p | wc -w) left $(pgrep -f 'agent4_cap16_deliver_final[.]py @I@ @N@ ' | wc -l)
"""
LAUNCH = r"""cd /tmp
rm -f F/shard_@I@.rc
nohup bash -c "OMP_NUM_THREADS=1 PY S/agent4_cap16_deliver_final.py @I@ @N@ @P@ >> F/shard_@I@.out 2>&1; echo \$? > F/shard_@I@.rc" > /dev/null 2>&1 < /dev/null &
echo started $!
"""
LOOKUP = r"""cd /tmp
s=$(date +%s%N)
timeout 20 stat S/.a4probe_@H@_$s > /dev/null 2>&1
rc=$?
echo LOOKUP $rc $(( ($(date +%s%N) - s) / 1000000 ))
"""


def fill(t, **kw):
    t = t.replace("PY", PY).replace(" S/", f" {S}/").replace("F/", f"{F}/")
    for k, v in kw.items():
        t = t.replace(f"@{k}@", str(v))
    return t


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{utc()} {msg}\n")


events = []


def event(msg):
    events.append(f"{utc()} {msg}")
    log(f"EVENT {msg}")


def ssh(h, script, timeout=45):
    # A frozen client can hang an ssh session, and control of the other hosts must go on, so a timeout returns None.
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                            IP[h], "bash -s"], input=script, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"ssh {h} timed out")
        return None
    if r.returncode != 0:
        log(f"ssh {h} exit {r.returncode}: {r.stderr.strip()[-300:]}")
        return None
    return r.stdout


probes = collections.deque(maxlen=200)
probe_off = [os.path.getsize(PROBE_LOG)]


def read_probes():
    with open(PROBE_LOG) as f:
        f.seek(probe_off[0])
        buf = f.read()
    end = buf.rfind("\n") + 1
    probe_off[0] += end
    for line in buf[:end].split("\n")[:-1]:
        x = line.split()
        if len(x) < 3 or not x[1].startswith("lookup="):
            continue
        probes.append((time.time(), x[1] == "lookup=OK", int(x[2][:-2]), x[0]))


def sample(h):
    out = ssh(h, SAMPLE, timeout=20)
    if out is None:
        return None
    running, m, busy = {}, None, None
    for line in out.split("\n"):
        x = line.split()
        if x and x[0] == "W" and len(x) == 4 and int(x[2]) == N_SHARD:
            running[int(x[1])] = int(x[3])
        if x and x[0] == "B":
            busy = int(x[1])
        if x and x[0] == "M":
            m = dict(zip(["n", "q", "r", "o", "on", "ort", "dn", "dq"], map(float, x[1:])))
    assert m is not None and busy is not None, out
    m["busy"] = busy
    return running, m


hosts = {h: {"state": "idle", "shard": None, "nproc": None, "since": None, "seen": None, "killed_at": None,
             "freezes": 0, "samples": collections.deque(maxlen=120)} for h in HOSTS}


def window(h):
    """Rates over the last 5 min of this host's samples on its current launch."""
    ss = list(hosts[h]["samples"])
    if len(ss) < 2:
        return None
    t1, m1 = ss[-1]
    old = [s for s in ss if s[0] <= t1 - WINDOW + 5] or ss[:1]
    t0, m0 = old[-1]
    dt = t1 - t0
    if dt < 60:
        return None
    d = {k: m1[k] - m0[k] for k in m1}
    busy = [s[1]["busy"] for s in ss if t0 < s[0] <= t1]
    return {"fps": d["o"] / dt, "busy": sum(busy) / len(busy), "backlog": d["dq"] / dt / 1000,
            "dr_wait_ms": d["dq"] / d["dn"] if d["dn"] else 0.0, "wire": d["r"] / dt / 1000,
            "open_rtt_ms": d["ort"] / d["on"] if d["on"] else 0.0}


def client_frozen(h):
    """The host's NFS calls per busy process over the last 30 s, against the up to 5 min before, or None."""
    ss = list(hosts[h]["samples"])
    if len(ss) < 3:
        return None
    t1 = ss[-1][0]
    a = [s for s in ss if s[0] <= t1 - FROZEN_S + 2]
    if not a:
        return None
    t0 = a[-1][0]
    base = [s for s in ss if t0 - WINDOW - 30 <= s[0] <= t0 - 60]
    if not base:
        return None
    tb = base[0][0]

    def rate(ta, tz):
        ma, mz = [s[1] for s in ss if s[0] == ta][0], [s[1] for s in ss if s[0] == tz][0]
        busy = [s[1]["busy"] for s in ss if ta < s[0] <= tz]
        b = sum(busy) / len(busy)
        return (mz["n"] - ma["n"]) / (tz - ta) / b if b >= 4 else None
    now, before = rate(t0, t1), rate(tb, t0)
    if now is None or before is None or before <= 0:
        return None
    return (now, before) if now < FROZEN_FRAC * before else None


def fps_text(h):
    w = window(h)
    return f"{w['fps']:.0f} files/s over the last {WINDOW // 60} min" if w else "files/s not yet measured"


def launch(h, i, n):
    out = ssh(h, fill(LAUNCH, I=i, N=N_SHARD, P=n))
    if out is None or not out.startswith("started"):
        log(f"launch of shard {i} on {h} failed: {out}")
        return False
    hosts[h].update(state="reading", shard=i, nproc=n, since=time.time(), seen=None)
    hosts[h]["samples"].clear()
    return True


def kill(h):
    i = hosts[h]["shard"]
    out = ssh(h, fill(KILL, I=i, N=N_SHARD))
    if out is None or not out.strip().endswith("left 0"):
        log(f"kill of shard {i} on {h} not confirmed: {out}")
        return False
    return True


def lookup_ms(h):
    out = ssh(h, fill(LOOKUP, H=h), timeout=40)
    if out is None or not out.startswith("LOOKUP"):
        return None
    rc, ms = map(int, out.split()[1:3])
    return ms if rc == 1 else None


def describe(h):
    x = hosts[h]
    head = f"{h} {x['state']}" + (f" shard {x['shard']} procs {x['nproc']}" if x["shard"] is not None else "") \
        + f" freezes {x['freezes']}"
    w = window(h) if x["state"] == "reading" else None
    if w is None:
        return head
    return (head + f", {w['fps']:.0f} files/s, busy {w['busy']:.0f}, DELEGRETURN backlog {w['backlog']:.0f} "
            f"wait {w['dr_wait_ms']:.0f} ms, on wire {w['wire']:.1f}, OPEN RTT {w['open_rtt_ms']:.0f} ms")


status_q = queue.Queue(maxsize=1)


def writer():
    while True:
        text = status_q.get()
        n = len([f for f in os.listdir(DEL) if f.endswith(".npz")])
        text = text.replace("ENTRIES", f"{n} of 3200")
        tmp = f"{F}/.RAMP_STATUS.txt.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, STATUS)


def main():
    threading.Thread(target=writer, daemon=True).start()
    done, failed, crashes = set(), set(), collections.Counter()
    for i in range(N_SHARD):
        rc = f"{F}/shard_{i}.rc"
        if os.path.exists(rc) and open(rc).read().strip() == "0":
            done.add(i)
    for h in HOSTS:
        r = sample(h)
        assert r is not None, h
        assert len(r[0]) <= 1, (h, r[0])
        for i, n in r[0].items():
            hosts[h].update(state="reading", shard=i, nproc=n, since=time.time())
    running = {hosts[h]["shard"] for h in HOSTS if hosts[h]["shard"] is not None}
    pending = collections.deque(i for i in range(N_SHARD) if i not in done and i not in running)
    event(f"CONTROLLER start, done shards {sorted(done)}, adopted "
          f"{ {h: (hosts[h]['shard'], hosts[h]['nproc']) for h in HOSTS if hosts[h]['shard'] is not None} }, "
          f"pending {list(pending)}")
    next_status = next_log = 0.0
    while True:
        read_probes()
        now = time.time()
        for h in HOSTS:
            x = hosts[h]
            if x["state"] == "reading":
                r = sample(h)
                if r is not None:
                    x["samples"].append((now, r[1]))
                    x["seen"] = now
                    i = x["shard"]
                    if i not in r[0]:
                        out = ssh(h, f"cat {F}/shard_{i}.rc 2>/dev/null || echo MISSING")
                        if out is None:
                            continue
                        if out.strip() == "0":
                            done.add(i)
                            event(f"DONE shard {i} on {h}, {fps_text(h)}")
                            x.update(state="idle", shard=None)
                        else:
                            crashes[i] += 1
                            if crashes[i] <= MAX_CRASHES:
                                event(f"CRASH shard {i} on {h}, exit {out.strip()!r}, relaunching, crash {crashes[i]}")
                                launch(h, i, x["nproc"])
                            else:
                                failed.add(i)
                                event(f"FAILED shard {i} on {h} after {crashes[i]} crashes, exit {out.strip()!r}, "
                                      f"not relaunched")
                                x.update(state="idle", shard=None)
                        continue
                fz = client_frozen(h)
                gone = now - (x["seen"] or x["since"])
                if fz or gone >= UNREACHABLE_S:
                    why = (f"{fz[0]:.2f} NFS calls/s per busy process over {FROZEN_S} s against {fz[1]:.2f} before"
                           if fz else f"unreachable for {gone:.0f} s")
                    text = fps_text(h)
                    if kill(h):
                        x["freezes"] += 1
                        x.update(state="killed", killed_at=now)
                        event(f"KILL {h} shard {x['shard']} at {x['nproc']} processes, {why}, {text}, "
                              f"freeze {x['freezes']}")
                    continue
                if x["nproc"] == REDUCED and x["freezes"] < MAX_FREEZES and now - x["since"] >= HEALTHY_S:
                    text = fps_text(h)
                    if kill(h) and launch(h, x["shard"], FULL):
                        event(f"RELAUNCH {h} shard {x['shard']} at {FULL} processes after {HEALTHY_S // 60} healthy "
                              f"min at {REDUCED}, {text}")
            elif x["state"] == "killed":
                ms = lookup_ms(h)
                if ms is not None and ms < LOOKUP_OK_MS:
                    if launch(h, x["shard"], REDUCED):
                        event(f"RELAUNCH {h} shard {x['shard']} at {REDUCED} processes, fresh lookup {ms} ms, "
                              f"{now - x['killed_at']:.0f} s after the kill")
                else:
                    log(f"{h} waiting, fresh lookup {ms}")
            elif x["state"] == "idle" and pending:
                i = pending.popleft()
                n = REDUCED if x["freezes"] >= MAX_FREEZES else FULL
                if launch(h, i, n):
                    event(f"START shard {i} on {h} at {n} processes")
                else:
                    pending.appendleft(i)

        if now >= next_log:
            next_log = now + LOG_EVERY
            lk = [p[2] for p in probes if p[0] >= now - WINDOW]
            log(f"TICK probe max5 {max(lk) if lk else None} | " + " | ".join(describe(h) for h in HOSTS))
        if now >= next_status:
            next_status = now + WINDOW
            lk = [p[2] for p in probes if p[0] >= now - WINDOW]
            fps = sum(window(h)["fps"] for h in HOSTS if hosts[h]["state"] == "reading" and window(h))
            reading = [h for h in HOSTS if hosts[h]["state"] == "reading"]
            text = (f"utc {utc()}\nper-host control, no global stop, only DONE ends the run (A4-D112)\n"
                    f"hosts reading {len(reading)}: " + ", ".join(f"{h} (shard {hosts[h]['shard']}, "
                    f"{hosts[h]['nproc']} processes)" for h in reading) + "\n"
                    f"hosts killed and waiting for a fresh lookup under {LOOKUP_OK_MS} ms: "
                    f"{[h for h in HOSTS if hosts[h]['state'] == 'killed']}\n"
                    f"entries delivered ENTRIES\nfiles/s last 5 min {fps:.0f}\n"
                    f"probe lookup ms on kaist143 last {probes[-1][2] if probes else None}, max over 5 min "
                    f"{max(lk) if lk else None}\n"
                    f"shards done {len(done)} of {N_SHARD}, pending {list(pending)}, failed {sorted(failed)}\n"
                    f"per host:\n" + "\n".join(describe(h) for h in HOSTS) + "\n"
                    f"events, every kill and relaunch with host, files/s and time:\n" + "\n".join(events[-60:]) + "\n")
            try:
                status_q.put_nowait(text)
            except queue.Full:
                log("status writer still busy, update skipped")
        if len(done) + len(failed) == N_SHARD and all(hosts[h]["state"] == "idle" for h in HOSTS):
            break
        time.sleep(TICK)

    if failed:
        event(f"STOPPED with failed shards {sorted(failed)}, no verification")
    else:
        event("ALL 16 shards done, running verification on kaist121")
        out = ssh("kaist121", f"cd /tmp; rm -f {F}/verify.rc; nohup bash -c 'OMP_NUM_THREADS=1 {PY} {S}/agent4_cap16_verify_delivery.py 32 > {F}/verify.out 2>&1; echo $? > {F}/verify.rc' > /dev/null 2>&1 < /dev/null & echo started")
        assert out is not None and "started" in out, out
        while True:
            time.sleep(TICK)
            out = ssh("kaist121", f"if [ -e {F}/verify.rc ]; then echo RC $(cat {F}/verify.rc); tail -2 {F}/verify.out; else echo RUNNING; fi")
            if out and out.startswith("RC "):
                event(f"VERIFY {out.strip()!r}")
                break
    status_q.put(f"utc {utc()}\nfinished\nentries delivered ENTRIES\nevents:\n" + "\n".join(events[-80:]) + "\n")
    time.sleep(30)


if __name__ == "__main__":
    main()
