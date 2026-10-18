"""The grasp-type file type_adaptive_sampling reads, from the Dexonomy delivery's per-grasp template names.

types is the 31 template names in numeric order. gtype__<stem> is the int8 index into types of every delivered grasp of
that object, in delivery row order, which is the order the reset files' grasp column indexes.
Usage: make_grasp_types.py <delivery extra dir> <out npz>
"""
import sys
from pathlib import Path

import numpy as np

extra, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(extra.glob("*.npz"))
assert len(files) == 3200, len(files)
tmpl = {}
for f in files:
    with np.load(f) as z:
        tmpl[f.stem] = z["tmpl"]
types = sorted({str(t) for names in tmpl.values() for t in names}, key=lambda t: int(t.split("_")[0]))
assert len(types) == 31, types
index = {t: i for i, t in enumerate(types)}
arrays = {"types": np.array(types)}
for stem, names in tmpl.items():
    arrays[f"gtype__{stem}"] = np.array([index[str(t)] for t in names], dtype=np.int8)
np.savez(out, **arrays)
print("saved", out, len(tmpl), "objects", sum(len(v) for v in tmpl.values()), "grasps")
