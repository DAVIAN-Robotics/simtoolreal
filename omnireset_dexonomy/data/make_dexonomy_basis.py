"""Dexonomy own-test passes as our hand basis npz, a PCA over hand postures pooled per grasp.

Each pass gives three postures, as DexLearn stacks them. They are the pregrasp (row 0 of Dexonomy's pregrasp_qpos, the
open start of the approach), the grasp and the squeeze. Every posture weighs the same, so every grasp weighs the same.
The delivery's columns are qpos[7:], the 12 hinges of the XHand MJCF, which are already in our canonical hand order.
Float64 sums stream over the 3,200 entries, then the covariance about the mean is diagonalized. k is the smallest count
reaching 90% of the variance. The npz has the same keys as the EgoSuite one plus mean and n_poses.
Usage: make_dexonomy_basis.py <delivery dir> <out npz>
"""
import sys
from pathlib import Path

import numpy as np

HAND = ["thumb_joint0", "thumb_joint1", "thumb_joint2", "index_joint0", "index_joint1", "index_joint2",
        "middle_joint0", "middle_joint1", "ring_joint0", "ring_joint1", "pinky_joint0", "pinky_joint1"]

src, out = Path(sys.argv[1]), sys.argv[2]
files = sorted(src.glob("*.npz"))
assert len(files) == 3200, len(files)
n, s, ss = 0, np.zeros(12), np.zeros((12, 12))
for i, f in enumerate(files):
    z = np.load(f)
    q = np.concatenate([z["q_pregrasp"], z["q_grasp"], z["q_squeeze"]]).astype(np.float64)
    assert q.shape[1] == 12 and len(q) == 3 * len(z["q_grasp"]), (f, q.shape)
    n += len(q)
    s += q.sum(0)
    ss += q.T @ q
    if i % 400 == 0:
        print(i, f.name, n, flush=True)
mean = s / n
cov = ss / n - np.outer(mean, mean)
w, v = np.linalg.eigh(cov)
w, v = w[::-1], v[:, ::-1]
assert w[-1] > 0, w
ratio = w / w.sum()
k = int(np.searchsorted(np.cumsum(ratio), 0.9) + 1)
print("poses", n, "grasps", n // 3)
print("mean", np.round(mean, 4).tolist())
print("k", k, "retained", round(float(np.cumsum(ratio)[k - 1]), 4), "ratio", np.round(ratio, 4).tolist())
print("component_std", np.round(np.sqrt(w), 4).tolist())
for r in range(k):
    print("pc", r, dict(zip(HAND, np.round(v[:, r], 3).tolist())))
np.savez(out, joint_names=np.array(HAND), components=v.T, component_std=np.sqrt(w), explained_ratio=ratio, k=k,
         mean=mean, n_poses=n, source=f"dexonomy cap16 own-test passes {n // 3} grasps, pregrasp row 0 grasp squeeze")
print("saved", out)
