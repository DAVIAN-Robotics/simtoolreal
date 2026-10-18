"""AIMD control of the total reader load on nas4 for the cap16 final delivery, on Sibeen's order of 2026-09-29 (A4-D113).
nas4 is one shared server, so the knob is the total number of reader processes over the 8 dj hosts, from 1 host at 8
to 8 hosts at 24, and the objective is total files/s. There is no stop. The floor is 8 readers and only DONE ends it.

Every 60 s the controller samples each reading host's NFS counters and runs one fresh-name lookup from every host, each
over ssh in its own thread with a timeout, so a hung NAS cannot block the loop. A window is bad when total files/s fell
over 10% against the last window, or a host froze, or the median lookup is over 2 s. A freeze cuts total readers by
50% and a slowdown without a freeze by 30%, shedding frozen hosts first and then the slowest per process. After 2 good windows it adds about 10% to one
host, raising the smallest host below 24 or starting a new host with the fastest lookup. A host changes its process
count by a relaunch of its shard, and entries already on disk are skipped. Each host runs one of 16 shards and takes
the next pending shard when its shard exits 0. A shard is never reassigned before its old readers are confirmed dead.
Nothing in this process touches nas4 directly. Its state and log are on local disk, and RAMP_STATUS.txt is written
through ssh to the host with the fastest lookup. After DONE it runs the verification on kaist121.
Usage: python3 agent4_cap16_aimd.py
"""
import collections
import datetime
import json
import os
import queue
import statistics
import subprocess
import threading
import time

S = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16"
F = f"{S}/deliver_final"
DEL = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16_deliver/dexonomy"
PY = "/home/nas4_user/sibeenkim/anaconda3/envs/dexonomy/bin/python"
HOSTS = ["kaist140", "kaist121", "kaist148", "kaist149", "kaist145", "kaist144", "kaist67", "kaist143"]
IP = {h: "143.248.159." + h[len("kaist"):] for h in HOSTS}
N_SHARD, MIN_P, MAX_P, FLOOR = 16, 8, 24, 8
LOG, STATE = "/tmp/a4_aimd.log", "/tmp/a4_aimd_state.json"
WINDOW, CUT_FREEZE, CUT_SLOW, ADD, FELL, LOOKUP_BAD_MS, LOOKUP_TIMEOUT_S, FROZEN_FRAC, GOOD_TO_ADD, MAX_CRASHES = \
    60, 0.5, 0.7, 0.1, 0.9, 2000, 12, 0.05, 2, 3

SAMPLE = r"""cd /tmp
for p in $(pgrep -f '^bash -c OMP_NUM_THREADS=1 PY S/agent4_cap16_deliver_final.py '); do echo W $(tr '\0' ' ' < /proc/$p/cmdline | awk '{print $6, $7, $8, $NF}'); done
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
# The whole launch runs in a background bash, so the ssh returns at once even while nas4 hangs. Each launch writes its
# exit status to its own file, so no stale status file needs removing first.
LAUNCH = r"""cd /tmp
nohup bash -c "OMP_NUM_THREADS=1 PY S/agent4_cap16_deliver_final.py @I@ @N@ @P@ >> F/shard_@I@.out 2>&1; echo \$? > @RC@" > /dev/null 2>&1 < /dev/null &
echo started $!
"""
LOOKUP = r"""cd /tmp
s=$(date +%s%N)
timeout -s KILL @T@ stat S/.a4probe_@H@_$s > /dev/null 2>&1
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


def ssh(h, script, timeout=45):
    # A hung NAS can hang a remote command, and control of the whole run must go on, so a timeout returns None.
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


def parallel(fn, hs):
    out = {}

    def run(h):
        out[h] = fn(h)
    ts = [threading.Thread(target=run, args=(h,)) for h in hs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def sample(h):
    out = ssh(h, SAMPLE, timeout=25)
    if out is None:
        return None
    running, m, busy = {}, None, None
    for line in out.split("\n"):
        x = line.split()
        if x and x[0] == "W" and len(x) == 5 and int(x[2]) == N_SHARD:
            running[int(x[1])] = (int(x[3]), x[4])
        if x and x[0] == "B":
            busy = int(x[1])
        if x and x[0] == "M":
            m = dict(zip(["n", "q", "r", "o", "on", "ort", "dn", "dq"], map(float, x[1:])))
    if m is None or busy is None:
        log(f"sample {h} unparsed: {out[-300:]}")
        return None
    m["busy"] = busy
    return running, m


def lookup(h):
    out = ssh(h, fill(LOOKUP, H=h, T=LOOKUP_TIMEOUT_S), timeout=LOOKUP_TIMEOUT_S + 15)
    if out is None or not out.startswith("LOOKUP"):
        return LOOKUP_TIMEOUT_S * 1000
    rc, ms = map(int, out.split()[1:3])
    return ms if rc == 1 else LOOKUP_TIMEOUT_S * 1000


# p is the process count while reading, 0 when off. A host whose kill is unconfirmed keeps its shard in "stopping".
hosts = {h: {"p": 0, "shard": None, "rc": None, "stopping": False, "prev": None, "rate": None, "launched": 0.0}
         for h in HOSTS}
done, failed, crashes = set(), set(), collections.Counter()
pending = collections.deque()
events, windows = [], []


def event(msg):
    events.append(f"{utc()} {msg}")
    log(f"EVENT {msg}")


def launch(h, i, p):
    rc = f"{F}/shard_{i}.{int(time.time())}.rc"
    out = ssh(h, fill(LAUNCH, I=i, N=N_SHARD, P=p, RC=rc))
    if out is None or not out.startswith("started"):
        log(f"launch of shard {i} on {h} failed: {out}")
        return False
    hosts[h].update(p=p, shard=i, rc=rc, stopping=False, prev=None, rate=None, launched=time.time())
    return True


def kill(h):
    i = hosts[h]["shard"]
    out = ssh(h, fill(KILL, I=i, N=N_SHARD))
    if out is None or not out.strip().endswith("left 0"):
        log(f"kill of shard {i} on {h} not confirmed: {out}")
        hosts[h].update(p=0, stopping=True)
        return False
    hosts[h].update(p=0, stopping=False, prev=None, rate=None)
    return True


def set_procs(h, p, why):
    """Relaunch h's shard at p processes, or stop h at p = 0, which returns its shard to pending."""
    x = hosts[h]
    old = x["p"]
    i = x["shard"]
    if not kill(h):
        event(f"{h} kill for {why} not confirmed, shard {i} held on {h} until its readers are gone")
        return
    if p == 0:
        x["shard"] = None
        pending.appendleft(i)
        event(f"{h} {old} -> 0 readers ({why}), shard {i} back to pending")
    elif launch(h, i, p):
        event(f"{h} {old} -> {p} readers ({why}), shard {i}")
    else:
        x["shard"] = None
        pending.appendleft(i)
        event(f"{h} relaunch at {p} failed ({why}), shard {i} back to pending")


def start(h, p, why):
    i = pending.popleft()
    if launch(h, i, p):
        event(f"{h} 0 -> {p} readers ({why}), shard {i}")
        return True
    pending.appendleft(i)
    return False


def save(extra):
    st = {"hosts": {h: {k: hosts[h][k] for k in ("p", "shard", "rc", "stopping")} for h in HOSTS},
          "done": sorted(done), "failed": sorted(failed), "crashes": dict(crashes), "pending": list(pending),
          "events": events[-80:], "windows": windows[-60:], **extra}
    with open(STATE + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(STATE + ".tmp", STATE)


status_q = queue.Queue(maxsize=1)


def writer(best_host):
    while True:
        text = status_q.get()
        h = best_host[0]
        script = (f"cd /tmp; n=$(ls {DEL} | grep -c 'npz$'); cat > {F}/.RAMP_STATUS.txt.tmp <<'A4EOF'\n{text}A4EOF\n"
                  f"sed -i \"s/ENTRIES/$n of 3200/\" {F}/.RAMP_STATUS.txt.tmp && mv {F}/.RAMP_STATUS.txt.tmp {F}/RAMP_STATUS.txt")
        if ssh(h, script, timeout=60) is None:
            log(f"status write through {h} failed")


def main():
    st = json.load(open(STATE))
    done.update(st["done"])
    failed.update(st["failed"])
    crashes.update({int(k): v for k, v in st.get("crashes", {}).items()})
    events.extend(st.get("events", []))
    windows.extend(st.get("windows", []))
    for h in HOSTS:
        hosts[h].update({k: st["hosts"][h][k] for k in ("p", "shard", "rc", "stopping")} if h in st["hosts"] else {})
    best = st.get("best", [0.0, 0])
    restart = [st.get("restart_from", 0)]
    samp = parallel(sample, HOSTS)
    for h in HOSTS:
        if samp[h] is None:
            continue
        for i, (p, rc) in samp[h][0].items():
            hosts[h].update(p=p, shard=i, rc=rc, stopping=False)
    owned = {hosts[h]["shard"] for h in HOSTS if hosts[h]["shard"] is not None}
    pending.extend(i for i in range(N_SHARD) if i not in done and i not in failed and i not in owned)
    event(f"CONTROLLER start, done {sorted(done)}, hosts "
          f"{ {h: (hosts[h]['shard'], hosts[h]['p']) for h in HOSTS if hosts[h]['shard'] is not None} }, "
          f"pending {list(pending)}")
    best_host = [HOSTS[0]]
    threading.Thread(target=writer, args=(best_host,), daemon=True).start()
    good, fps_prev = 0, {}
    while True:
        t0 = time.time()
        reading = [h for h in HOSTS if hosts[h]["p"] > 0]
        samp = parallel(sample, reading + [h for h in HOSTS if hosts[h]["stopping"]])
        lk = parallel(lookup, HOSTS)
        best_host[0] = min(HOSTS, key=lambda h: lk[h])

        # Hosts whose earlier kill was not confirmed: retry, and release the shard once the readers are gone.
        for h in HOSTS:
            x = hosts[h]
            if x["stopping"] and samp.get(h) is not None and x["shard"] not in samp[h][0]:
                event(f"{h} readers of shard {x['shard']} confirmed gone, shard back to pending")
                pending.appendleft(x["shard"])
                x.update(shard=None, stopping=False)
            elif x["stopping"]:
                kill(h)

        # Shard exits.
        for h in reading:
            x = hosts[h]
            if samp[h] is None or x["shard"] in samp[h][0]:
                continue
            out = ssh(h, f"cat {x['rc']} 2>/dev/null || echo MISSING", timeout=30)
            if out is None:
                continue
            i = x["shard"]
            if out.strip() == "0":
                done.add(i)
                x.update(shard=None, prev=None, rate=None)
                event(f"DONE shard {i} on {h}")
                if pending:
                    start(h, x["p"], "next shard")
                else:
                    x["p"] = 0
            else:
                crashes[i] += 1
                if crashes[i] <= MAX_CRASHES:
                    event(f"CRASH shard {i} on {h}, exit {out.strip()!r}, relaunching, crash {crashes[i]}")
                    launch(h, i, x["p"])
                else:
                    failed.add(i)
                    x.update(p=0, shard=None)
                    event(f"FAILED shard {i} on {h} after {crashes[i]} crashes, exit {out.strip()!r}")

        # Per-host rates over this window.
        reading = [h for h in HOSTS if hosts[h]["p"] > 0]
        frozen, fps = [], {}
        for h in reading:
            x = hosts[h]
            if samp.get(h) is None:
                frozen.append(h)
                continue
            now, m = t0, samp[h][1]
            if x["prev"] is not None:
                tp, mp = x["prev"]
                dt = now - tp
                fps[h] = (m["o"] - mp["o"]) / dt
                calls = (m["n"] - mp["n"]) / dt / max(m["busy"], 1)
                if x["rate"] and m["busy"] >= 4 and calls < FROZEN_FRAC * x["rate"]:
                    frozen.append(h)
                elif m["busy"] >= 4:
                    x["rate"] = calls
            x["prev"] = (now, m)
            if lk[h] >= LOOKUP_TIMEOUT_S * 1000 and h not in frozen:
                frozen.append(h)
        total = sum(fps.values())
        med = statistics.median(lk.values())
        readers = sum(hosts[h]["p"] for h in HOSTS)
        measured = len(fps) == len([h for h in reading if h not in frozen]) and len(fps) > 0

        # AIMD.
        action = "hold"
        bad = []
        if frozen:
            bad.append(f"frozen {frozen}")
        if med > LOOKUP_BAD_MS:
            bad.append(f"median lookup {med:.0f} ms")
        comp = [h for h in fps if h in fps_prev and hosts[h]["launched"] <= t0 - 2 * WINDOW]
        if comp and measured and sum(fps[h] for h in comp) < FELL * sum(fps_prev[h] for h in comp):
            bad.append(f"files/s fell {sum(fps[h] for h in comp):.0f} against {sum(fps_prev[h] for h in comp):.0f} "
                       f"on {len(comp)} steady hosts")
        if measured and total > best[0]:
            best = [total, readers]
        if bad and readers > FLOOR:
            good = 0
            target = max(FLOOR, int(readers * (CUT_FREEZE if frozen else CUT_SLOW)))
            shed = readers - target
            order = [h for h in frozen if hosts[h]["p"] > 0] + sorted(
                [h for h in reading if h not in frozen], key=lambda h: fps.get(h, 0) / hosts[h]["p"])
            for h in order:
                if shed <= 0:
                    break
                p = hosts[h]["p"]
                if h not in frozen and p - shed >= MIN_P:
                    set_procs(h, p - shed, "cut")
                    shed = 0
                else:
                    set_procs(h, 0, "cut, frozen" if h in frozen else "cut")
                    shed -= p
            action = f"cut {readers} -> {sum(hosts[h]['p'] for h in HOSTS)} readers ({'; '.join(bad)})"
        elif bad:
            good = 0
            action = f"hold at floor ({'; '.join(bad)})"
        elif measured:
            good += 1
            if good >= GOOD_TO_ADD and readers < MAX_P * len(HOSTS):
                step = max(MIN_P, round(readers * ADD))
                grow = [h for h in reading if hosts[h]["p"] < MAX_P]
                if grow:
                    h = min(grow, key=lambda h: hosts[h]["p"])
                    set_procs(h, min(MAX_P, hosts[h]["p"] + step), "add")
                    action = f"add {step} on {h}"
                elif pending:
                    idle = [h for h in HOSTS if hosts[h]["p"] == 0 and not hosts[h]["stopping"]]
                    if idle:
                        h = min(idle, key=lambda h: lk[h])
                        start(h, min(MAX_P, step), "add")
                        action = f"add host {h} at {min(MAX_P, step)}"
                good = 0

        # Restart after zero readers, at the freeze cut, 50% below the last total, spread over the hosts whose fresh
        # lookup answers. Floor: never zero readers while work is left and any host answers.
        if sum(hosts[h]["p"] for h in HOSTS) == 0 and pending:
            idle = sorted([h for h in HOSTS if hosts[h]["p"] == 0 and not hosts[h]["stopping"]
                           and lk[h] < LOOKUP_BAD_MS], key=lambda h: lk[h])
            if idle:
                target = max(FLOOR, int(restart[0] * CUT_FREEZE)) if restart[0] else FLOOR
                k = max(1, min(len(idle), len(pending), target // MIN_P))
                p = max(MIN_P, min(MAX_P, target // k))
                for h in idle[:k]:
                    if pending:
                        start(h, p, f"restart at {target} readers")
                restart[0] = 0
                good = 0
                action += f", restart {k} hosts at {p} readers each"
            else:
                action += ", zero readers, no host answers a fresh lookup yet"

        fps_prev = fps
        line = (f"{utc()} readers {sum(hosts[h]['p'] for h in HOSTS)} "
                f"[{', '.join(f'{h}:{hosts[h]['p']}' for h in HOSTS if hosts[h]['p'])}] files/s {total:.0f} "
                f"(" + ", ".join(f"{h} {fps[h]:.0f}" for h in fps) + f") lookup median {med:.0f} ms "
                f"[{', '.join(f'{h}:{lk[h]}' for h in HOSTS)}] action {action}")
        windows.append(line)
        log(f"WINDOW {line}")
        save({"best": best, "restart_from": restart[0]})
        text = (f"utc {utc()}\nAIMD control of total readers on nas4 (A4-D113), no stop, floor {FLOOR} readers, "
                f"only DONE ends the run\nlaw: any host frozen (calls per busy process under {FROZEN_FRAC} of before, "
                f"or its fresh lookup timing out at {LOOKUP_TIMEOUT_S} s) cuts total readers {int(100 * (1 - CUT_FREEZE))}%. "
                f"Median lookup over {LOOKUP_BAD_MS} ms or files/s down over {int(100 * (1 - FELL))}% on steady hosts cuts "
                f"{int(100 * (1 - CUT_SLOW))}%. {GOOD_TO_ADD} good windows add {int(100 * ADD)}% on one host\n"
                f"entries delivered ENTRIES\n"
                f"shards done {len(done)} of {N_SHARD}, pending {list(pending)}, failed {sorted(failed)}\n"
                f"best total files/s {best[0]:.0f} at {best[1]} readers\n"
                f"windows (readers per host, files/s, lookup ms per host, action):\n" + "\n".join(windows[-30:]) +
                "\nevents:\n" + "\n".join(events[-40:]) + "\n")
        try:
            status_q.put_nowait(text)
        except queue.Full:
            log("status writer still busy, update skipped")
        if len(done) + len(failed) == N_SHARD and all(hosts[h]["p"] == 0 and not hosts[h]["stopping"] for h in HOSTS):
            break
        time.sleep(max(0.0, WINDOW - (time.time() - t0)))

    if failed:
        event(f"FINISHED with failed shards {sorted(failed)}, no verification")
    else:
        event("ALL 16 shards done, running verification on kaist121")
        out = ssh("kaist121", f"cd /tmp; nohup bash -c 'OMP_NUM_THREADS=1 {PY} {S}/agent4_cap16_verify_delivery.py 32 > {F}/verify.out 2>&1; echo $? > {F}/verify.rc' > /dev/null 2>&1 < /dev/null & echo started")
        assert out is not None and "started" in out, out
        while True:
            time.sleep(WINDOW)
            out = ssh("kaist121", f"if [ -e {F}/verify.rc ]; then echo RC $(cat {F}/verify.rc); tail -2 {F}/verify.out; else echo RUNNING; fi")
            if out and out.startswith("RC "):
                event(f"VERIFY {out.strip()!r}")
                break
    save({"best": best})
    status_q.put(f"utc {utc()}\nfinished\nentries delivered ENTRIES\nevents:\n" + "\n".join(events[-80:]) + "\n")
    time.sleep(60)


if __name__ == "__main__":
    main()
