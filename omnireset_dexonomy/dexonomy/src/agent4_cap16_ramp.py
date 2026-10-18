"""Ramped resume of the cap16 final delivery on Sibeen's ramp rule (A4-D106). Runs detached on kaist143 next to the
fresh-name nas4 probe, and drives agent4_cap16_deliver_final.py shards on the other dj hosts over ssh by IP.

Rule. Start on 1 host (A4-D109, after the 09-29 07:00 freeze of the loaded clients) with 24 processes per host, add one
about every 10 min only when the probe and every active host's DELEGRETURN backlog and wait are healthy over a full
window after warm-up, stop adding at the first rise, drop the newest host at a stall, and stop every shard by PID at a
freeze. A freeze is a probe HANG, or any active host completing under 5% of its previous NFS calls per busy process for
30 s, or an active host unreachable for 60 s. After a freeze the controller resumes one host lower once the probe is
healthy, and at level 1 it stays stopped. kaist143 carries the probe and runs no shard, so at most 7 delivery hosts
run. Each host runs one of 16 shards at a time and takes the next pending shard when its shard exits 0. After DONE the
controller runs the verification on kaist121.

The stop path never touches nas4 in this process. ssh calls have timeouts, the probe log and the decision log are on
local disk, and RAMP_STATUS.txt is written by a separate thread, so a frozen NAS cannot block a kill.
Usage: python3 agent4_cap16_ramp.py
"""
import collections
import datetime
import json
import os
import queue
import subprocess
import threading
import time

S = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16"
F = f"{S}/deliver_final"
DEL = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16_deliver/dexonomy"
PY = "/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python"
HOSTS = ["kaist140", "kaist121", "kaist148", "kaist149", "kaist145", "kaist144", "kaist67"]
IP = {h: "143.248.159." + h[len("kaist"):] for h in HOSTS}
N_SHARD = 16
PROBE_LOG, LOG, CTL, STATE = "/tmp/a4_nas4probe.log", "/tmp/a4_ramp.log", "/tmp/a4_ramp_ctl.json", "/tmp/a4_ramp_state.json"
STATUS = f"{F}/RAMP_STATUS.txt"
TICK, WINDOW, ADD_EVERY, LOG_EVERY = 15, 300, 600, 60
RISE_MS, RISE_COUNT, STALL_MS, RECOVER_PROBES = 20, 2, 1000, 10
BACKLOG_GROWTH, BACKLOG_FLOOR, FPS_DROP, WARMUP, DR_WAIT_MS = 1.5, 200, 0.3, 180, 5000
FROZEN_FRAC, FROZEN_S, UNREACHABLE_S, NPROC = 0.05, 30, 60, 24
RULES = (f"rise = {RISE_COUNT}+ probe lookups over {RISE_MS} ms in 5 min, or a host whose mean DELEGRETURN backlog "
         f"grew over {BACKLOG_GROWTH}x its previous 5 min and above {BACKLOG_FLOOR}, the previous window starting "
         f"{WARMUP} s or more after the host's launch, or any host's DELEGRETURN wait over {DR_WAIT_MS} ms in a window "
         f"after warm-up. An add waits until every active host has that comparison. "
         f"stall = a probe lookup over {STALL_MS} ms, or a host's 5-min files/s per busy process under {FPS_DROP} of its "
         f"previous 5 min. "
         f"freeze = a probe HANG (no answer in 20 s), or an active host completing under {FROZEN_FRAC} of its previous "
         f"NFS calls per busy process for {FROZEN_S} s, or an active host unreachable for {UNREACHABLE_S} s. "
         f"recovery = {RECOVER_PROBES} straight probes at or under {RISE_MS} ms, and at level 1 no resume.")

SAMPLE = r"""cd /tmp
for p in $(pgrep -f '^bash -c OMP_NUM_THREADS=1 PY S/agent4_cap16_deliver_final.py '); do echo W $(tr '\0' ' ' < /proc/$p/cmdline | awk '{print $6, $7}'); done
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
    # A frozen nas4 can hang an ssh login, and the controller must keep going to stop shards, so a timeout returns None.
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
    running, m, busy = [], None, None
    for line in out.split("\n"):
        x = line.split()
        if x and x[0] == "W" and len(x) == 3 and int(x[2]) == N_SHARD:
            running.append(int(x[1]))
        if x and x[0] == "B":
            busy = int(x[1])
        if x and x[0] == "M":
            m = dict(zip(["n", "q", "r", "o", "on", "ort", "dn", "dq"], map(float, x[1:])))
    assert m is not None and busy is not None, out
    m["busy"] = busy
    return running, m


hosts = {h: {"shard": None, "since": None, "seen": None, "samples": collections.deque(maxlen=120)} for h in HOSTS}


def window(h, back):
    """Rates over the 5 min ending `back` s ago, from this host's samples."""
    ss = [s for s in hosts[h]["samples"] if s[0] <= time.time() - back + 5]
    if not ss:
        return None
    t1, m1 = ss[-1]
    old = [s for s in ss if s[0] <= t1 - WINDOW + 5]
    if not old or hosts[h]["since"] is None or old[-1][0] < hosts[h]["since"]:
        return None
    t0, m0 = old[-1]
    dt = t1 - t0
    d = {k: m1[k] - m0[k] for k in m1}
    busy = sum(s[1]["busy"] for s in ss if t0 < s[0] <= t1) / len([s for s in ss if t0 < s[0] <= t1])
    return {"t0": t0, "fps": d["o"] / dt, "busy": busy, "fps_per_busy": d["o"] / dt / busy if busy >= 1 else None, "rpcs": d["n"] / dt, "wire": d["r"] / dt / 1000,
            "backlog": d["dq"] / dt / 1000, "dr_wait_ms": d["dq"] / d["dn"] if d["dn"] else 0.0,
            "open_rtt_ms": d["ort"] / d["on"] if d["on"] else 0.0}


def describe(h, w):
    x = f"{h} shard {hosts[h]['shard']} procs {hosts[h].get('nproc', '?')}"
    if w is None:
        return x + " warming"
    return (x + f" busy {w['busy']:.0f} {w['fps']:.0f} files/s DELEGRETURN backlog {w['backlog']:.0f} "
            f"wait {w['dr_wait_ms']:.0f} ms, on wire {w['wire']:.1f}, OPEN RTT {w['open_rtt_ms']:.0f} ms")


def ctl():
    return json.load(open(CTL)) if os.path.exists(CTL) else {}


def nproc(h):
    c = ctl()
    return c.get("nproc_host", {}).get(h, c.get("nproc_default", NPROC))


def launch(h, i):
    n = nproc(h)
    out = ssh(h, fill(LAUNCH, I=i, N=N_SHARD, P=n))
    assert out is not None and out.startswith("started"), (h, i, out)
    hosts[h].update(shard=i, since=time.time(), seen=None, nproc=n)
    event(f"START shard {i} on {h} with {n} processes ({out.strip()})")


def kill(h):
    i = hosts[h]["shard"]
    out = ssh(h, fill(KILL, I=i, N=N_SHARD))
    if out is None or not out.strip().endswith("left 0"):
        log(f"kill of shard {i} on {h} not confirmed: {out}")
        return False
    event(f"KILL shard {i} on {h}: {out.strip()}")
    hosts[h].update(shard=None, since=None)
    hosts[h]["samples"].clear()
    return True


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


def baseline_ok(h, prev):
    return prev is not None and prev["t0"] >= hosts[h]["since"] + WARMUP


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


def main():
    threading.Thread(target=writer, daemon=True).start()
    done, failed = set(), set()
    for i in range(N_SHARD):
        rc = f"{F}/shard_{i}.rc"
        if os.path.exists(rc) and open(rc).read().strip() == "0":
            done.add(i)
    running = {}
    for h in HOSTS:
        r = sample(h)
        assert r is not None, h
        for i in r[0]:
            running[i] = h
            hosts[h].update(shard=i, since=time.time())
    pending = collections.deque(i for i in range(N_SHARD) if i not in done and i not in running)
    level, adding, frozen, stall_cool = max(1, len(running)), True, False, 0.0
    last_add = time.time()
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        level, adding, last_add, failed = st["level"], st["adding"], st["last_add"], set(st["failed"])
        events.extend(st["events"])
        pending = collections.deque(i for i in pending if i not in failed)
    next_status = next_log = 0.0
    event(f"CONTROLLER start, done shards {sorted(done)}, adopted {running}, pending {list(pending)}, level {level}")
    log(f"RULES {RULES}")
    while True:
        read_probes()
        now = time.time()
        recent = [p for p in probes if p[0] >= now - WINDOW]
        active = [h for h in HOSTS if hosts[h]["shard"] is not None]

        # Samples, every tick, for the per-client freeze trigger and the windows.
        sampled = {}
        for h in active:
            r = sample(h)
            if r is None:
                continue
            hosts[h]["samples"].append((now, r[1]))
            hosts[h]["seen"] = now
            sampled[h] = r[0]

        # Freeze: stop every shard at once, one host lower after recovery, adding stays stopped.
        why = []
        if probes and not probes[-1][1]:
            why.append(f"probe HANG at {probes[-1][3]}")
        for h in active:
            fz = client_frozen(h)
            if fz:
                why.append(f"{h} {fz[0]:.2f} NFS calls/s per busy process over {FROZEN_S} s against {fz[1]:.2f} before")
            if now - (hosts[h]["seen"] or hosts[h]["since"]) >= UNREACHABLE_S:
                why.append(f"{h} unreachable for {now - (hosts[h]['seen'] or hosts[h]['since']):.0f} s")
        if why and not frozen:
            frozen, adding = True, False
            resume = len(active) > 1
            level = max(1, len(active) - 1)
            event(f"FREEZE {'; '.join(why)}. Stopping all {len(active)} shards, "
                  + (f"resume level {level} after recovery" if resume else "no resume at level 1"))
            if not resume:
                level = 0
            next_status = 0.0
        if frozen:
            for h in active:
                i = hosts[h]["shard"]
                if kill(h):
                    pending.appendleft(i)
            tail = list(probes)[-RECOVER_PROBES:]
            if level >= 1 and not [h for h in HOSTS if hosts[h]["shard"] is not None] and len(tail) == RECOVER_PROBES \
                    and all(p[1] and p[2] <= RISE_MS for p in tail):
                frozen = False
                last_add = now
                event(f"RECOVERED, {RECOVER_PROBES} probes at or under {RISE_MS} ms, resuming at level {level}")
            if now >= next_status:
                next_status = now + WINDOW
                try:
                    status_q.put_nowait(f"utc {utc()}\nFROZEN, shards stopped by PID, " + (
                        "no resume at level 1, waiting for the orchestrator" if level == 0 else f"resume level {level}")
                        + f"\nhosts still running a shard: {[h for h in HOSTS if hosts[h]['shard'] is not None]}"
                        f"\nentries delivered ENTRIES\nevents:\n" + "\n".join(events[-30:]) + "\n")
                except queue.Full:
                    log("status writer still busy, update skipped")
            time.sleep(TICK)
            continue

        # Shard exits.
        for h in active:
            if h not in sampled:
                continue
            i = hosts[h]["shard"]
            if i not in sampled[h]:
                out = ssh(h, f"cat {F}/shard_{i}.rc 2>/dev/null || echo MISSING")
                if out is None:
                    continue
                if out.strip() == "0":
                    done.add(i)
                    event(f"DONE shard {i} on {h}")
                else:
                    failed.add(i)
                    adding = False
                    event(f"CRASH shard {i} on {h}, exit {out.strip()!r}, not relaunched, adding stopped")
                hosts[h].update(shard=None)
        active = [h for h in HOSTS if hosts[h]["shard"] is not None]
        cur = {h: window(h, 0) for h in active}
        prev = {h: window(h, WINDOW) for h in active}

        # Stall: drop the newest host.
        stall = [f"probe {p[2]} ms at {p[3]}" for p in recent if p[1] and p[2] > STALL_MS and p[0] >= now - 60]
        stall += [f"{h} files/s per busy process {cur[h]['fps_per_busy']:.2f} against {prev[h]['fps_per_busy']:.2f}"
              for h in active if cur[h] and prev[h] and cur[h]["fps_per_busy"] and prev[h]["fps_per_busy"]
              and cur[h]["fps_per_busy"] < FPS_DROP * prev[h]["fps_per_busy"]]
        if stall and now >= stall_cool and active:
            newest = max(active, key=lambda h: hosts[h]["since"])
            i = hosts[newest]["shard"]
            if kill(newest):
                pending.appendleft(i)
                level, adding, stall_cool = len(active) - 1, False, now + WINDOW
                event(f"STALL {'; '.join(stall)}. Dropped newest host {newest}, level {level}, adding stopped")

        # Rise: stop adding.
        if adding:
            rise = [f"{sum(p[1] and p[2] > RISE_MS for p in recent)} probe lookups over {RISE_MS} ms in 5 min"] \
                if sum(p[1] and p[2] > RISE_MS for p in recent) >= RISE_COUNT else []
            rise += [f"{h} DELEGRETURN backlog {cur[h]['backlog']:.0f} against {prev[h]['backlog']:.0f}" for h in active
                     if cur[h] and baseline_ok(h, prev[h])
                     and cur[h]["backlog"] > BACKLOG_GROWTH * prev[h]["backlog"] and cur[h]["backlog"] > BACKLOG_FLOOR]
            rise += [f"{h} DELEGRETURN wait {cur[h]['dr_wait_ms']:.0f} ms" for h in active
                     if cur[h] and cur[h]["t0"] >= hosts[h]["since"] + WARMUP and cur[h]["dr_wait_ms"] > DR_WAIT_MS]
            if rise:
                adding = False
                event(f"RISE {'; '.join(rise)}. Adding stopped at level {level}")

        # Add one host about every 10 min while healthy.
        if adding and pending and now - last_add >= ADD_EVERY and level < len(HOSTS) \
                and all(cur[h] and baseline_ok(h, prev[h]) for h in active):
            level += 1
            last_add = now
            event(f"ADD level {level}")

        # One-shot relaunch of listed hosts' shards at their current process count, for the fewer-processes test.
        c = ctl()
        if c.get("relaunch"):
            for h in c["relaunch"]:
                if hosts[h]["shard"] is not None:
                    i = hosts[h]["shard"]
                    if kill(h):
                        launch(h, i)
                        event(f"RELAUNCH shard {i} on {h} at {hosts[h]['nproc']} processes")
            c.pop("relaunch")
            with open(CTL, "w") as f:
                json.dump(c, f)

        # Fill free hosts up to the level with pending shards.
        for h in HOSTS:
            if len([x for x in HOSTS if hosts[x]["shard"] is not None]) >= level or not pending:
                break
            if hosts[h]["shard"] is None:
                hosts[h]["since"] = None
                hosts[h]["samples"].clear()
                launch(h, pending.popleft())

        # Log and status.
        active = [h for h in HOSTS if hosts[h]["shard"] is not None]
        per = " | ".join(describe(h, cur.get(h)) for h in active)
        lk = [p[2] for p in recent]
        if now >= next_log:
            next_log = now + LOG_EVERY
            log(f"TICK level {level} adding {adding} probe last {probes[-1][2] if probes else None} max5 "
                f"{max(lk) if lk else None} | {per}")
        if now >= next_status:
            next_status = now + WINDOW
            fps = sum(cur[h]["fps"] for h in active if cur.get(h))
            text = (f"utc {utc()}\nhosts active {len(active)}: " + ", ".join(f"{h} (shard {hosts[h]['shard']})"
                    for h in active) + "\nprocesses per host " + ", ".join(f"{h} {hosts[h].get('nproc', '?')}" for h in active) + "\nentries delivered ENTRIES\n"
                    f"files/s last 5 min {fps:.0f}\nprobe lookup ms last {probes[-1][2] if probes else None}, "
                    f"max over 5 min {max(lk) if lk else None}\nlevel {level}, adding {'on' if adding else 'stopped'}"
                    f"\nshards done {len(done)} of {N_SHARD}, pending {list(pending)}, failed {sorted(failed)}\n"
                    f"per host: {per}\nrules: {RULES}\nevents:\n" + "\n".join(events[-30:]) + "\n")
            try:
                status_q.put_nowait(text)
            except queue.Full:
                log("status writer still busy, update skipped")

        with open(STATE + ".tmp", "w") as f:
            json.dump({"level": level, "adding": adding, "last_add": last_add, "failed": sorted(failed),
                       "events": events[-60:]}, f)
        os.replace(STATE + ".tmp", STATE)
        if len(done) == N_SHARD:
            break
        time.sleep(TICK)

    event("ALL 16 shards done, running verification on kaist121")
    out = ssh("kaist121", f"cd /tmp; rm -f {F}/verify.rc; nohup bash -c 'OMP_NUM_THREADS=1 {PY} {S}/agent4_cap16_verify_delivery.py 32 > {F}/verify.out 2>&1; echo $? > {F}/verify.rc' > /dev/null 2>&1 < /dev/null & echo started")
    assert out is not None and "started" in out, out
    while True:
        time.sleep(TICK)
        out = ssh("kaist121", f"if [ -e {F}/verify.rc ]; then echo RC $(cat {F}/verify.rc); tail -2 {F}/verify.out; else echo RUNNING; fi")
        if out and out.startswith("RC "):
            event(f"VERIFY {out.strip()!r}")
            break
    status_q.put(f"utc {utc()}\nfinished\nentries delivered ENTRIES\nevents:\n" + "\n".join(events[-40:]) + "\n")
    time.sleep(30)


if __name__ == "__main__":
    main()
