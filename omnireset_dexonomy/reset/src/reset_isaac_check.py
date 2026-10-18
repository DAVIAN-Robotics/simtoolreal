"""Check the full-set Isaac finals in one dir: shapes, finiteness, unit quaternions, rows kept, tested and passed.
Prints one line per object as TSV to --out, and a summary. Usage: reset_isaac_check.py <final dir> --out <tsv>
"""
import argparse
import glob
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("final_dir")
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(f"{a.final_dir}/*.npz")):
    if os.path.basename(f).startswith("."):
        continue
    d = np.load(f)
    n = d["obj_pos"].shape[0]
    assert d["obj_quat_wxyz"].shape == (n, 4) and d["arm_q"].shape == (n, 7) and d["hand_q"].shape == (n, 12), f
    assert d["hand_target"].shape == (n, 12) and d["grasp"].shape == (n,) and d["candidate_row"].shape == (n,), f
    for k in ("obj_pos", "obj_quat_wxyz", "arm_q", "hand_q", "hand_target"):
        assert np.isfinite(d[k]).all(), (f, k)
    assert np.abs(np.linalg.norm(d["obj_quat_wxyz"], axis=1) - 1).max() < 1e-4, f
    assert n == min(10000, int(d["passed"])), (f, n, int(d["passed"]))
    rows.append((os.path.basename(f)[:-4], n, int(d["tested"]), int(d["passed"])))
with open(a.out, "w") as o:
    o.write("stem\tkept\ttested\tpassed\n")
    for r in rows:
        o.write("\t".join(map(str, r)) + "\n")
kept = np.array([r[1] for r in rows]); rate = np.array([r[3] / r[2] for r in rows])
print(f"{len(rows)} objects, kept 10,000 on {int((kept == 10000).sum())}, fewer on {int((kept < 10000).sum())}, "
      f"pass rate min {rate.min():.3f} median {np.median(rate):.3f} max {rate.max():.3f}")
