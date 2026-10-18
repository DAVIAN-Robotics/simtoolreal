"""Build the template-pool index for one chunk from the pool as it is now, then check it equals the glob.

For each of the 31 templates, lists new_tmpl_pool/<tmpl>/<stem>/floating/real for this chunk's stems (threads,
one directory per object, never a recursive glob) and writes gsched/pool/<chunk>__<tmpl>.txt with the paths as
os.path.join(pool, tmpl, stem, "floating", "real", file), the strings upstream's recursive glob returns. With
--check <tmpl>, also runs upstream's glob(os.path.join(pool, tmpl, "**/*.npy"), recursive=True) and asserts the
union of all chunks' index files for that template equals it (run once every chunk is indexed).
Usage: agent4_cap16_pool_index.py <chunk> <n threads> | --check <tmpl>
"""
import glob
import json
import os
import sys
from multiprocessing.pool import ThreadPool

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy"
POOL = f"{D}/cap16/new_tmpl_pool"
IDX = f"{D}/cap16/gsched/pool"
TMPLS = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{D}/Dexonomy/assets/hand/xhand/init_tmpl/*.npy"))
assert len(TMPLS) == 31

if sys.argv[1] == "--check":
    t = sys.argv[2]
    idx = set()
    for f in glob.glob(f"{IDX}/*__{t}.txt"):
        idx.update(open(f).read().split("\n")[:-1])
    g = set(glob.glob(os.path.join(POOL, t, "**/*.npy"), recursive=True))
    print(f"{t}: index {len(idx)}, glob {len(g)}, equal {idx == g}")
    assert idx == g
    sys.exit(0)

chunk, nth = sys.argv[1], int(sys.argv[2])
plan = json.load(open(f"{D}/cap16/plan.json"))
stems = [x for x in plan["hosts"] if x["host"] == chunk][0]["stems"]


def files(job):
    t, stem = job
    d = os.path.join(POOL, t, stem, "floating", "real")
    return t, [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy")] if os.path.isdir(d) else []


os.makedirs(IDX, exist_ok=True)
by = {t: [] for t in TMPLS}
with ThreadPool(nth) as pool:
    for t, fs in pool.imap_unordered(files, [(t, s) for t in TMPLS for s in stems], chunksize=16):
        by[t] += fs
n = 0
for t, fs in by.items():
    out = f"{IDX}/{chunk}__{t}.txt"
    assert not os.path.exists(out), out
    tmp = f"{IDX}/.{chunk}__{t}.txt.tmp"
    open(tmp, "w").write("".join(p + "\n" for p in sorted(fs)))
    os.rename(tmp, out)
    n += len(fs)
print(f"{chunk}: {n} pool paths indexed over {len(TMPLS)} templates, {len(stems)} objects")
