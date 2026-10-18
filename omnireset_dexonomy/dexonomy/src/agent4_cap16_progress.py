"""Append one line of cap16 progress (40-epoch continuation) to cap16/progress.txt, from counters the run keeps.

Per-pair init batches come from the indexer's state (gsched/index_state.json, batches it has read). A pair's epochs
done is its batches over its chunk's batches per epoch. The phase is 1 until every pair has 20 epochs, then 2; the
line names the slowest pair. Init jobs done, running and failed come from gsched/ijob_done, ijob_lock and
ijob_failed. Refined starts, kept grasps, own-tested grasps and passes are the workers' per-lease counters plus the
bootstrap counters (gsched/count). ETAs use the rates since the previous line (progress_state.json): phase 1 and
init from the batch rate, refinement plus own test from a cost-weighted work model (0.09 s per refinement start,
1.25 s per own test, every kept grasp own-tested once), and the run's ETA is the later of init and that. The indexer
can lag init (A4-D85), so the line also gives init's rate from the batches of the jobs finished in the last 30 min,
and an init ETA from the batches left in unfinished jobs.
"""
import datetime
import glob
import json
import math
import os
import time

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy"
S = f"{D}/cap16/gsched"
TARGET, PHASE1 = 40, 20
now = time.time()
plan = json.load(open(f"{D}/cap16/plan.json"))
bpe = {h["host"]: math.ceil(h["n_objects"] / 10) for h in plan["hosts"]}
st = json.load(open(f"{S}/index_state.json"))
pair_b = {k: v["batches"] for k, v in st.items()}
starts = sum(v["starts"] for v in st.values())
batches = sum(pair_b.values())
total = sum(TARGET * bpe[k.split("__")[0]] for k in pair_b)
left_p1 = sum(max(PHASE1 * bpe[k.split("__")[0]] - b, 0) for k, b in pair_b.items())
ep = {k: b / bpe[k.split("__")[0]] for k, b in pair_b.items()}
slow = min(ep, key=ep.get)
phase = 1 if min(ep.values()) < PHASE1 else 2
jobs = {j["id"]: j for j in json.load(open(f"{S}/init_jobs.json"))}
jobs_total = len(jobs)
n_done, n_run, n_fail = (len(os.listdir(f"{S}/{d}")) for d in ["ijob_done", "ijob_lock", "ijob_failed"])
# init's own rate, independent of how far the indexer has read: batches of the jobs finished in the last 30 min
done_t = {k: os.stat(f"{S}/ijob_done/{k}").st_mtime for k in os.listdir(f"{S}/ijob_done")}
r_jobs = sum(jobs[k]["batches"] for k, t in done_t.items() if t >= now - 1800) / 1800
left_jobs = sum(j["batches"] for k, j in jobs.items() if k not in done_t)

items = {"grasp": 0, "eval": 0}
outs = {"grasp": 0, "eval": 0}
for f in glob.glob(f"{S}/count/*.txt"):
    for line in open(f).read().split("\n")[:-1]:  # whole lines only, a worker may be appending
        _, kind, _, _, n, n_out = line.split()
        items[kind] += int(n)
        outs[kind] += int(n_out)
proj = starts * total / batches
done_f = f"{D}/cap16_deliver/dexonomy/DONE"
delivery = open(done_f).read().strip() if os.path.exists(done_f) else "once after epoch 40"


def hm(t):
    return "stalled" if t == float("inf") else datetime.datetime.utcfromtimestamp(t).strftime("%m-%d %H:%MZ")


cur = {"t": now, "batches": batches, "grasp": items["grasp"], "eval": items["eval"]}
state_f = f"{D}/cap16/progress_state.json"
eta = "rates from the next line"
if os.path.exists(state_f):
    prev = json.load(open(state_f))
    dt = now - prev["t"]
    r_b = (batches - prev["batches"]) / dt
    r_ref, r_own = (items["grasp"] - prev["grasp"]) / dt, (items["eval"] - prev["eval"]) / dt
    keep = outs["grasp"] / items["grasp"]
    work_left = 0.09 * max(proj - items["grasp"], 0) + 1.25 * max(proj * keep - items["eval"], 0)
    work_rate = 0.09 * r_ref + 1.25 * r_own
    t_p1 = now + left_p1 / r_b if r_b > 0 else float("inf")
    t_init = now + (total - batches) / r_b if r_b > 0 else float("inf")
    t_work = now + work_left / work_rate if work_rate > 0 else float("inf")
    eta = (f"phase 1 {hm(t_p1) if left_p1 else 'done'}, init {hm(t_init)}, run {hm(max(t_init, t_work))} "
           f"at {r_b:.2f} batches/s, {r_ref:,.0f} starts/s refined, {r_own:,.0f}/s own-tested, {work_rate:,.0f} busy CPU-s/s")
json.dump(cur, open(state_f, "w"))

line = (f"{datetime.datetime.utcfromtimestamp(now).strftime('%Y-%m-%dT%H:%MZ')} | phase {phase} | pairs at epochs "
        f"{min(ep.values()):.1f} to {max(ep.values()):.1f} of {TARGET}, slowest {slow} | init batches {batches:,} of {total:,} "
        f"({batches / total:.1%}), jobs {n_done}/{jobs_total} done, {n_run} running, {n_fail} failed | starts {starts:,} of "
        f"~{proj:,.0f} | refined {items['grasp']:,}, {outs['grasp']:,} kept | own-tested {items['eval']:,}, "
        f"{outs['eval']:,} passed | delivery: {delivery} | ETA {eta} | init by finished jobs {r_jobs:.2f} batches/s "
        f"over 30 min, {left_jobs:,} batches in unfinished jobs, init ETA {hm(now + left_jobs / r_jobs if r_jobs > 0 else float('inf'))}")
with open(f"{D}/cap16/progress.txt", "a") as f:
    f.write(line + "\n")
print(line)
