"""Each reset state's 5 fingertip pad points in its object's frame, one <stem>.npz per object (ft_obj, bodies).

FK of the state's arm and hand joints (hand_q, the contact pose) on the training URDF, the pad points at the
fingertip_offset_by_body of the fr3-xhand profile, then into the object frame of the state's object pose.
Usage: RESET_TREE=<tree> goal_fingertips.py <reset final dir> <out dir> <shard i> <n shards>
"""
import sys
from pathlib import Path

import numpy as np
import torch

import reset_common as rc

# robots.py fr3-xhand fingertip_offset_by_body, the vendor taxel centroids in each link2 frame.
PADS = {
    "thumb_rota_link2": (0.0, 0.035083167, 0.0034695),
    "index_rota_link2": (0.004874, 0.0, 0.0307745),
    "mid_link2": (0.004874, 0.0, 0.0307745),
    "ring_link2": (0.004874, 0.0, 0.0307745),
    "pinky_link2": (0.004874, 0.0, 0.0307745),
}
robots = (rc.TREE / "isaacsimenvs/tasks/simtoolreal/robots.py").read_text()
for body, off in PADS.items():
    assert f'"{body}": {off}' in robots, (body, off)

src, out = Path(sys.argv[1]), Path(sys.argv[2])
shard, n_shard = int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
u = rc.Urdf()
T_wb = rc.homog(rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)), torch.tensor(rc.BASE_POS, dtype=torch.float64))
names = list(PADS)
pad = torch.tensor([PADS[b] + (1.0,) for b in names], dtype=torch.float64)  # (5, 4)
files = sorted(src.glob("*.npz"))[shard::n_shard]
for i, f in enumerate(files):
    with np.load(f) as z:
        q19 = np.concatenate([z["arm_q"], z["hand_q"]], axis=1)
        obj_pos, obj_quat = z["obj_pos"], z["obj_quat_wxyz"]
    q = {j: torch.as_tensor(q19[:, k], dtype=torch.float64) for k, j in enumerate(rc.ARM_JOINTS + rc.HAND_JOINTS)}
    T = u.fk_from("fr3_link0", q)
    world = torch.stack([(T_wb @ T[b] @ pad[k])[:, :3] for k, b in enumerate(names)], dim=1)  # (S, 5, 3)
    R = rc.quat_to_mat(torch.as_tensor(obj_quat, dtype=torch.float64))
    ft_obj = torch.einsum("sji,skj->ski", R, world - torch.as_tensor(obj_pos, dtype=torch.float64)[:, None])
    np.savez(out / f.name, ft_obj=ft_obj.numpy().astype(np.float32), bodies=np.array(names))
    if i % 100 == 0:
        print(f"shard {shard} {i}/{len(files)} {f.stem}", flush=True)
print(f"shard {shard} done {len(files)}", flush=True)
