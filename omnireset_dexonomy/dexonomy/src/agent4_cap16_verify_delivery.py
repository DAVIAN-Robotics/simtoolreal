"""Independent check of the cap16 final delivery, after DONE.

Reads every <stem>.npz and extra/<stem>.npz in the relay dir and checks: the five float32 fields with 3/4/12/12/12
columns and equal row counts per entry, finite values, unit quaternions, extra rows equal to main rows, every
extra src a unique pass-list path of that stem, the delivered sources equal to the set of all pass-list lines
(nothing missing, nothing extra), no dot-temp files left, and the DONE totals equal to what is on disk.
Usage: agent4_cap16_verify_delivery.py <n proc>
"""
import collections
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy"
CLONE_REAL = os.path.realpath(f"{D}/Dexonomy")
DEL = f"{D}/cap16_deliver/dexonomy"
S = f"{D}/cap16/gsched"
COLS = {"hand_pos": 3, "hand_quat": 4, "q_grasp": 12, "q_squeeze": 12, "q_pregrasp": 12}


def check(path):
    stem = os.path.basename(path)[:-4]
    m, e = np.load(path), np.load(f"{DEL}/extra/{stem}.npz")
    assert sorted(m.files) == sorted(COLS), (stem, m.files)
    n = m["hand_pos"].shape[0]
    for k, c in COLS.items():
        assert m[k].dtype == np.float32 and m[k].shape == (n, c), (stem, k, m[k].dtype, m[k].shape)
        assert np.isfinite(m[k]).all(), (stem, k)
    assert np.allclose(np.linalg.norm(m["hand_quat"], axis=1), 1, atol=1e-5), stem
    assert e["tmpl"].shape == (n,) and e["src"].shape == (n,), stem
    src = [s[len(CLONE_REAL) + 1:] for s in e["src"].tolist()]
    assert len(set(src)) == n and all(s.split("/")[4] == stem for s in src), stem
    return stem, n, src


if __name__ == "__main__":
    main = sorted(glob.glob(f"{DEL}/*.npz"))
    extra = sorted(glob.glob(f"{DEL}/extra/*.npz"))
    assert [os.path.basename(p) for p in main] == [os.path.basename(p) for p in extra], "main and extra differ"
    dots = [f for f in os.listdir(DEL) + os.listdir(f"{DEL}/extra") if f.startswith(".")]
    assert not dots, dots
    with Pool(int(sys.argv[1])) as pool:
        res = pool.map(check, main, chunksize=8)
    delivered = collections.Counter(s for _, _, src in res for s in src)
    passes = set()
    n_lines = 0
    for pf in glob.glob(f"{S}/pass/*.txt"):
        ls = open(pf).read().split("\n")[:-1]
        n_lines += len(ls)
        passes.update(ls)
    assert max(delivered.values()) == 1, "a source delivered twice"
    missing, extra_src = passes - set(delivered), set(delivered) - passes
    rows = sum(n for _, n, _ in res)
    done = open(f"{DEL}/DONE").read()
    m = re.search(r"(\d+) of 3200 entries with own-test passes, (\d+) grasps", done)
    print(f"{len(main)} entries, {rows} grasps on disk; pass lists {n_lines} lines, {len(passes)} unique; "
          f"missing {len(missing)}, not in pass lists {len(extra_src)}; DONE says {m.group(1)} entries, {m.group(2)} grasps")
    assert not missing and not extra_src
    assert int(m.group(1)) == len(main) and int(m.group(2)) == rows
    print("delivery verified")
