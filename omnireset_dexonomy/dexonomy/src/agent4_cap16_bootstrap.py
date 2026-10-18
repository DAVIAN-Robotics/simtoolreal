"""Build the v2 workers' own-test queues, pass lists and counters for one chunk from what its own loops and the v1
workers did. Run after every v1 worker has stopped, one process per chunk on that chunk's host, so each uses its own
NFS client.

For each of the chunk's pairs, the refined grasps whose init is in the grasp done list and whose grasp_data file
exists go to gsched/todo/eval__<pair>.txt (complete files, their refinement returned). Own-tested grasps whose
succ_grasp link exists go to gsched/pass/<chunk>__<tmpl>.txt. One counter line per pair and kind goes to
gsched/count/backfill_<chunk>.txt (unique items, outputs), in the workers' counter format.
Usage: agent4_cap16_bootstrap.py <chunk> <n threads>
"""
import glob
import os
import sys
import time
from multiprocessing.pool import ThreadPool

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy"
S = f"{D}/cap16/gsched"
chunk, nth = sys.argv[1], int(sys.argv[2])
os.chdir(f"{D}/Dexonomy")
out_f = f"{S}/count/backfill_{chunk}.txt"
assert not os.path.exists(out_f), out_f


def kept(p):
    q = p.replace("/init_data/", "/grasp_data/", 1)
    return q if os.path.exists(q) else None


def passed(p):
    return p if os.path.lexists(p.replace("/grasp_data/", "/succ_grasp/", 1)) else None


counters = []
t0 = time.time()
with ThreadPool(nth) as pool:
    for f in sorted(glob.glob(f"{S}/done/*__{chunk}__*.txt")):
        kind, c, t = os.path.basename(f)[:-4].split("__")
        assert c == chunk
        ps = sorted(set(open(f).read().split("\n")) - {""})
        out = [x for x in pool.map(kept if kind == "grasp" else passed, ps, chunksize=64) if x]
        target = f"{S}/todo/eval__{c}__{t}.txt" if kind == "grasp" else f"{S}/pass/{c}__{t}.txt"
        assert not os.path.exists(target), target
        if out:
            open(target, "w").write("\n".join(out) + "\n")
        counters.append(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {kind} {c} {t} {len(ps)} {len(out)}")
open(out_f, "w").write("\n".join(counters) + "\n")
tot = {}
for line in counters:
    _, kind, _, _, n, o = line.split()
    tot.setdefault(kind, [0, 0])
    tot[kind][0] += int(n)
    tot[kind][1] += int(o)
print(f"bootstrap {chunk}: {len(counters)} done lists in {time.time() - t0:.0f} s; "
      + ", ".join(f"{k} {v[0]} items {v[1]} outputs" for k, v in tot.items()))
