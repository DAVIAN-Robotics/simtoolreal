"""OmniReset-style candidate generation, one random grasp per object pose, then IK top-1.

Draws object poses as reset_sample.py does, redraws any below the table, pairs each with one random
grasp, drops pairs whose hand dips below the table, and solves IK for the rest in fixed calls of
1,992 wrists, the last call padded by repetition. Times each part and logs GPU memory.
Usage: reset_ik_omni.py --n 12000 --out <npz>
"""
import argparse
import time

import numpy as np
import torch
import trimesh
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

import reset_common as rc

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--robot", default="/home/nas_main/sibeenkim/draftrec/_curobo2/reset/ik/shard0.robot.yml")
ap.add_argument("--table", default="/home/nas_main/sibeenkim/draftrec/_curobo2/reset/ik/shard0.table.yml")
ap.add_argument("--out", required=True)
ap.add_argument("--franka-limits", action="store_true", help="IK joint limits from Franka's suggested rectangular limits")
a = ap.parse_args()
import xml.etree.ElementTree as ET  # noqa: E402

import yaml  # noqa: E402

# Franka's suggested rectangular position limits for FR3, https://frankarobotics.github.io/docs/robot_specifications.html
FRANKA_BOX = [(-2.3476, 2.3476), (-1.5454, 1.5454), (-2.4937, 2.4937), (-2.7714, -0.4226), (-2.5100, 2.5100), (0.7773, 4.2841), (-2.7045, 2.7045)]

dev = torch.device("cuda")
mem = lambda: f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB"
t0 = time.time()
G = rc.load_grasps(dev)
n_g = len(G["tmpl"])
obj_v = torch.as_tensor(trimesh.util.concatenate(rc.object_pieces()).convex_hull.vertices, device=dev)
gen = torch.Generator(device=dev).manual_seed(a.seed)
lo, hi = torch.tensor(rc.BOX_MIN, device=dev, dtype=torch.float64), torch.tensor(rc.BOX_MAX, device=dev, dtype=torch.float64)
pos = lo + (hi - lo) * torch.rand(a.n, 3, device=dev, generator=gen, dtype=torch.float64)
quat = torch.nn.functional.normalize(torch.randn(a.n, 4, device=dev, generator=gen, dtype=torch.float64), dim=-1)
for _ in range(100):
    bad = pos[:, 2] + (rc.quat_to_mat(quat)[:, 2, :] @ obj_v.T).min(-1).values < rc.TABLE_TOP
    if not bad.any():
        break
    k = int(bad.sum())
    pos[bad] = lo + (hi - lo) * torch.rand(k, 3, device=dev, generator=gen, dtype=torch.float64)
    quat[bad] = torch.nn.functional.normalize(torch.randn(k, 4, device=dev, generator=gen, dtype=torch.float64), dim=-1)
assert not bad.any()
grasp = torch.randint(0, n_g, (a.n,), device=dev, generator=gen)
t_sample = time.time() - t0

u = rc.Urdf()
hand_links = [l for l in u.subtree("palm") if u.links[l].find("collision") is not None]
q = {j: G["q_grasp"][grasp, i].double() for i, j in enumerate(rc.HAND_JOINTS)}
T = u.fk_from("palm", q)
T_obj_palm = rc.homog(rc.quat_to_mat(G["hand_quat"][grasp].double()), G["hand_pos"][grasp].double())
T_wo = rc.homog(rc.quat_to_mat(quat), pos)
low = torch.full((a.n,), float("inf"), dtype=torch.float64, device=dev)
for l in hand_links:
    for v in u.collision_hulls(l):
        v = torch.as_tensor(v, device=dev)
        Tw = T_wo @ T_obj_palm @ T[l]
        low = torch.minimum(low, (Tw[:, 2, :3] @ v.T).min(-1).values + Tw[:, 2, 3])
hand_ok = low >= rc.TABLE_TOP
t_hand = time.time() - t0 - t_sample
print(f"sampled {a.n} poses in {t_sample:.1f} s, hand check {t_hand:.1f} s, pairs kept {int(hand_ok.sum())}, {mem()}", flush=True)

robot = yaml.safe_load(open(a.robot))
if a.franka_limits:
    tree = ET.parse(robot["robot_cfg"]["kinematics"]["urdf_path"])
    for j in tree.getroot().findall("joint"):
        if j.get("name") in rc.ARM_JOINTS:
            lo_, hi_ = FRANKA_BOX[rc.ARM_JOINTS.index(j.get("name"))]
            j.find("limit").set("lower", str(lo_)), j.find("limit").set("upper", str(hi_))
    robot["robot_cfg"]["kinematics"]["urdf_path"] = f"{a.out}.fr3_arm_franka_box.urdf"
    tree.write(robot["robot_cfg"]["kinematics"]["urdf_path"])
ik = InverseKinematics(InverseKinematicsCfg.create(robot=robot, scene_model=yaml.safe_load(open(a.table)),
                                                   num_seeds=32, self_collision_check=True, position_tolerance=0.001,
                                                   orientation_tolerance=0.01, max_batch_size=1992))
T_wb = rc.homog(rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)), torch.tensor(rc.BASE_POS, dtype=torch.float64)).to(dev)
T_7p = torch.as_tensor(rc.t_7_palm(u), device=dev)
idx = torch.nonzero(hand_ok).squeeze(-1)
T_b7 = torch.linalg.inv(T_wb) @ T_wo[idx] @ T_obj_palm[idx] @ torch.linalg.inv(T_7p)
t1 = time.time()
succ, sol = [], []
for s in range(0, len(idx), 1992):
    e = min(s + 1992, len(idx))
    pad = torch.cat([torch.arange(s, e, device=dev), torch.full((1992 - (e - s),), s, device=dev)])
    ik.reset_seed()
    r = ik.solve_pose(GoalToolPose.from_poses({"fr3_link7": Pose(position=T_b7[pad, :3, 3].float().contiguous(),
                                                                   quaternion=rc.mat_to_quat(T_b7[pad, :3, :3]).float().contiguous())},
                                              num_goalset=1), return_seeds=1)
    succ.append(r.success.view(1992)[: e - s].cpu()), sol.append(r.js_solution.position.view(1992, -1)[: e - s].cpu())
    print(f"call {s // 1992} wrists {e - s} {mem()}", flush=True)
t_ik = time.time() - t1
succ, sol = torch.cat(succ), torch.cat(sol)
np.savez(a.out, pos=pos[idx].cpu().numpy(), quat=quat[idx].cpu().numpy(), grasp=grasp[idx].cpu().numpy(), success=succ.numpy(), q=sol.numpy())
# Isaac inputs, the reached wrists in one random order.
ok = torch.nonzero(succ).squeeze(-1).numpy()
perm = np.random.default_rng(a.seed).permutation(len(ok))
sel = ok[perm]
np.savez(a.out.replace(".npz", "_poses.npz"), pos=pos[idx].cpu().numpy()[sel], quat=quat[idx].cpu().numpy()[sel])
np.savez(a.out.replace(".npz", "_cands.npz"), pose=np.arange(len(sel)), grasp=grasp[idx].cpu().numpy()[sel], q=sol.numpy()[sel])
lim = np.array(FRANKA_BOX if a.franka_limits else [[-2.9007, 2.9007], [-1.8361, 1.8361], [-2.9007, 2.9007], [-3.0770, -0.1169], [-2.8763, 2.8763], [0.4398, 4.6216], [-3.0508, 3.0508]])
q_ok = sol.numpy()[ok]
print("min margin to the limits in use, rad, per joint", np.minimum(q_ok - lim[:, 0], lim[:, 1] - q_ok).min(0).round(4))
print(f"IK {len(idx)} wrists in {t_ik:.1f} s including setup, reached {int(succ.sum())}, total {time.time() - t0:.1f} s", flush=True)
