"""Each reset state's palm pose in the env frame, one <stem>.npz per object (palm_pos, palm_quat_wxyz).

FK of the state's arm and hand joints (hand_q, the contact pose) on the training URDF, the palm link placed by the
robot base pose, the same frame as the reset states' object poses.
Usage: RESET_TREE=<tree> goal_palm.py <reset final dir> <out dir> <shard i> <n shards>
"""
import sys
from pathlib import Path

import numpy as np
import torch

import reset_common as rc

src, out = Path(sys.argv[1]), Path(sys.argv[2])
shard, n_shard = int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
u = rc.Urdf()
T_wb = rc.homog(rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)), torch.tensor(rc.BASE_POS, dtype=torch.float64))
files = sorted(src.glob("*.npz"))[shard::n_shard]
for i, f in enumerate(files):
    with np.load(f) as z:
        q19 = np.concatenate([z["arm_q"], z["hand_q"]], axis=1)
    q = {j: torch.as_tensor(q19[:, k], dtype=torch.float64) for k, j in enumerate(rc.ARM_JOINTS + rc.HAND_JOINTS)}
    T = T_wb @ u.fk_from("fr3_link0", q)["palm"]
    np.savez(out / f.name, palm_pos=T[:, :3, 3].numpy().astype(np.float32),
             palm_quat_wxyz=rc.mat_to_quat(T[:, :3, :3]).numpy().astype(np.float32))
    if i % 100 == 0:
        print(f"shard {shard} {i}/{len(files)} {f.stem}", flush=True)
print(f"shard {shard} done {len(files)}", flush=True)
