"""Own-test pass counts of the 40-epoch cap16 Dexonomy run, per object and per grasp type, from the pass lists.

Writes counts.csv (3,200 objects x 31 types plus total) and pass_counts.png to cap16/pass_counts on nas4.
"""
import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16"
OUT = f"{D}/pass_counts"

stems = sorted(x["stem"] for x in json.load(open(f"{D}/manifest.json")))
assert len(stems) == 3200
files = sorted(glob.glob(f"{D}/gsched/pass/*.txt"))
assert len(files) == 248, len(files)
types = sorted({os.path.basename(f)[:-4].split("__")[1] for f in files}, key=lambda t: int(t.split("_")[0]))
assert len(types) == 31, types

si = {s: i for i, s in enumerate(stems)}
ti = {t: j for j, t in enumerate(types)}
C = np.zeros((3200, 31), dtype=np.int64)
for f in files:
    tmpl = os.path.basename(f)[:-4].split("__")[1]
    raw = open(f).read()
    assert raw.endswith("\n"), f
    per = collections.Counter()
    for rel in set(raw.split("\n")[:-1]):
        p = rel.split("/")
        assert p[3] == tmpl, (f, rel)
        per[p[4]] += 1
    for stem, n in per.items():
        C[si[stem], ti[tmpl]] += n

os.makedirs(OUT, exist_ok=True)
tot = C.sum(1)
with open(f"{OUT}/counts.csv", "w") as fh:
    fh.write("object," + ",".join(types) + ",total\n")
    for s, row, t in zip(stems, C, tot):
        fh.write(s + "," + ",".join(map(str, row)) + f",{t}\n")

print("total passes", C.sum())
print("objects with 0 passes", (tot == 0).sum(), "| per object min / median / max", tot.min(), np.median(tot), tot.max())
cov = (C > 0).sum(1)
print("types with >=1 pass per object, min / median / max", cov.min(), np.median(cov), cov.max())
for j, t in enumerate(types):
    c = C[:, j]
    print(f"{t:32s} total {c.sum():9d}  objects with 0: {(c == 0).sum():5d}  median {np.median(c):6.0f}  max {c.max():6d}")

order = np.argsort(-tot, kind="stable")
fig = plt.figure(figsize=(20, 22))
gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.1, 0.7], hspace=0.35)

ax = fig.add_subplot(gs[0])
M = np.ma.masked_equal(C[order].T, 0)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("white")
im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=1, vmax=C.max()))
ax.set_yticks(range(31), types, fontsize=9)
ax.set_xlabel("3,200 objects, sorted by total passes (white = 0 passes)")
ax.set_title("Own-test passes per object and grasp type")
fig.colorbar(im, ax=ax, pad=0.01, label="passes")

ax = fig.add_subplot(gs[1])
ax.boxplot([C[:, j] + 1 for j in range(31)], whis=(0, 100), showfliers=False)
ax.set_yscale("log")
ax.set_xticks(range(1, 32), types, rotation=60, ha="right", fontsize=9)
ax.set_ylabel("passes per object + 1 (box = quartiles, whisker = min to max)")
ax.set_title("Per grasp type, spread across the 3,200 objects. Top label = total passes, bottom label = objects with 0")
top = C.max() * 3
for j in range(31):
    ax.text(j + 1, top, f"{C[:, j].sum() / 1e3:.0f}k", ha="center", va="bottom", fontsize=8)
    ax.text(j + 1, 0.55, f"{(C[:, j] == 0).sum()}", ha="center", va="top", fontsize=8, color="crimson")
ax.set_ylim(0.5, top * 3)

gsb = gs[2].subgridspec(1, 2, wspace=0.2)
ax = fig.add_subplot(gsb[0])
bins = np.logspace(0, np.log10(tot.max() + 1), 50)
ax.hist(tot, bins=bins)
ax.set_xscale("log")
ax.set_xlabel("total passes per object, all 31 types")
ax.set_ylabel("objects")
ax.set_title(f"Per object total. median {np.median(tot):.0f}, min {tot.min()}, max {tot.max()}")
ax = fig.add_subplot(gsb[1])
ax.hist(cov, bins=np.arange(-0.5, 32.5, 1))
ax.set_xlabel("grasp types with at least 1 pass")
ax.set_ylabel("objects")
ax.set_title(f"Per object, types covered. median {np.median(cov):.0f} of 31")

fig.suptitle(f"cap16 Dexonomy, 40 epochs, {C.sum():,} own-test passes on {(tot > 0).sum()} of 3,200 objects", fontsize=15, y=0.995)
fig.savefig(f"{OUT}/pass_counts.png", dpi=110, bbox_inches="tight")
print("wrote", f"{OUT}/pass_counts.png", f"{OUT}/counts.csv")
