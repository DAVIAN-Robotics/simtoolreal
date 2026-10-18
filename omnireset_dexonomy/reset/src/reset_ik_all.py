"""IK for every cap16 object on dj, the full-set scheme, one worker per GPU.

Per object. Draw 100,000 object poses in the 050 goal box with uniform orientation, redrawing any
whose collision hull dips below the table top. Pair each with one random grasp of the object's
delivered passes, as OmniReset draws one grasp per attempt. Drop pairs whose hand hull dips below the
table top. Solve FR3 IK for the rest, top-1 of 32 seeds, inside Franka's suggested rectangular limits,
in fixed calls of 1,992 wrists. Save the reached wrists in one random order, ready for the Isaac test.
Workers claim objects through mkdir locks and take an object once its delivered file exists.
Usage: reset_ik_all.py --worker <i>
"""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import torch
import trimesh
import yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

import reset_common as rc

D = "/home/nas4_user/sibeenkim/work/_graspoc"
DEL = f"{D}/dexonomy/cap16_deliver/dexonomy"
OUT = f"{D}/reset/ik_all"
FRANKA_BOX = [(-2.3476, 2.3476), (-1.5454, 1.5454), (-2.4937, 2.4937), (-2.7714, -0.4226), (-2.5100, 2.5100), (0.7773, 4.2841), (-2.7045, 2.7045)]
ap = argparse.ArgumentParser()
ap.add_argument("--worker", required=True)
ap.add_argument("--n", type=int, default=100000)
a = ap.parse_args()
rc.check_tree()
dev = torch.device("cuda")
os.makedirs(f"{OUT}/claims", exist_ok=True)
manifest = {m["stem"]: m for m in json.load(open(f"{D}/dexonomy/cap16/manifest.json"))}
stems = sorted(manifest)
assert len(stems) == 3200

robot = yaml.safe_load(open(f"{D}/reset/shard0.robot.yml"))
tree = ET.parse(f"{D}/reset/shard0.fr3_arm.urdf")
for j in tree.getroot().findall("joint"):
    if j.get("name") in rc.ARM_JOINTS:
        lo_, hi_ = FRANKA_BOX[rc.ARM_JOINTS.index(j.get("name"))]
        j.find("limit").set("lower", str(lo_)), j.find("limit").set("upper", str(hi_))
arm_urdf = f"{OUT}/fr3_arm_franka_box.worker{a.worker}.urdf"
tree.write(arm_urdf)
robot["robot_cfg"]["kinematics"]["urdf_path"] = arm_urdf
ik = InverseKinematics(InverseKinematicsCfg.create(robot=robot, scene_model=yaml.safe_load(open(f"{D}/reset/shard0.table.yml")),
                                                   num_seeds=32, self_collision_check=True, position_tolerance=0.001,
                                                   orientation_tolerance=0.01, max_batch_size=1992))
assert list(ik.tool_frames) == ["fr3_link7"]
u = rc.Urdf()
hand_links = [l for l in u.subtree("palm") if u.links[l].find("collision") is not None]
hand_hulls = {l: [torch.as_tensor(v, device=dev) for v in u.collision_hulls(l)] for l in hand_links}
T_wb = rc.homog(rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)), torch.tensor(rc.BASE_POS, dtype=torch.float64)).to(dev)
T_bw, T_7p_inv = torch.linalg.inv(T_wb), torch.linalg.inv(torch.as_tensor(rc.t_7_palm(u), device=dev))
lo, hi = torch.tensor(rc.BOX_MIN, device=dev, dtype=torch.float64), torch.tensor(rc.BOX_MAX, device=dev, dtype=torch.float64)
print(f"[w{a.worker}] ready", flush=True)


def lowest(T_z_rows, T_z_off, v, chunk=8192):
    """Lowest world z of vertices v under each transform, in chunks of poses so one large hull fits in memory."""
    out = torch.empty(len(T_z_rows), dtype=torch.float64, device=dev)
    for c in range(0, len(T_z_rows), chunk):
        out[c:c + chunk] = (T_z_rows[c:c + chunk] @ v.T).min(-1).values + T_z_off[c:c + chunk]
    return out


def object_hull(stem):
    pts = []
    for c in manifest[stem]["collision"]:
        m = trimesh.load(c["mesh"], force="mesh")
        m.apply_scale(c["scale"])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rc.rpy_mat(*c["rpy"]), c["xyz"]
        m.apply_transform(T)
        pts.append(m.vertices)
    return torch.as_tensor(trimesh.convex.convex_hull(np.concatenate(pts)).vertices, device=dev)


def run(stem):
    t0 = time.time()
    g = np.load(f"{DEL}/{stem}.npz")
    n_g = g["hand_pos"].shape[0]
    hp, hq, qg = (torch.as_tensor(g[k], device=dev).double() for k in ("hand_pos", "hand_quat", "q_grasp"))
    gen = torch.Generator(device=dev).manual_seed(zlib.crc32(stem.encode()))
    ov = object_hull(stem)
    pos = lo + (hi - lo) * torch.rand(a.n, 3, device=dev, generator=gen, dtype=torch.float64)
    quat = torch.nn.functional.normalize(torch.randn(a.n, 4, device=dev, generator=gen, dtype=torch.float64), dim=-1)
    redrawn = 0
    for _ in range(1000):
        bad = lowest(rc.quat_to_mat(quat)[:, 2, :], pos[:, 2], ov) < rc.TABLE_TOP
        if not bad.any():
            break
        k = int(bad.sum())
        redrawn += k
        pos[bad] = lo + (hi - lo) * torch.rand(k, 3, device=dev, generator=gen, dtype=torch.float64)
        quat[bad] = torch.nn.functional.normalize(torch.randn(k, 4, device=dev, generator=gen, dtype=torch.float64), dim=-1)
    assert not bad.any(), f"{stem} could not place {int(bad.sum())} poses above the table"
    grasp = torch.randint(0, n_g, (a.n,), device=dev, generator=gen)
    T_wo = rc.homog(rc.quat_to_mat(quat), pos) @ rc.homog(rc.quat_to_mat(hq[grasp]), hp[grasp])
    T = u.fk_from("palm", {j: qg[grasp, i] for i, j in enumerate(rc.HAND_JOINTS)})
    low = torch.full((a.n,), float("inf"), dtype=torch.float64, device=dev)
    for l, vs in hand_hulls.items():
        Tw = T_wo @ T[l]
        for v in vs:
            low = torch.minimum(low, lowest(Tw[:, 2, :3], Tw[:, 2, 3], v))
    idx = torch.nonzero(low >= rc.TABLE_TOP).squeeze(-1)
    T_b7 = T_bw @ T_wo[idx] @ T_7p_inv
    pq = rc.mat_to_quat(T_b7[:, :3, :3]).float()
    pp = T_b7[:, :3, 3].float()
    succ, sol = [], []
    for s in range(0, len(idx), 1992):
        e = min(s + 1992, len(idx))
        pad = torch.cat([torch.arange(s, e, device=dev), torch.full((1992 - (e - s),), s, device=dev)])
        ik.reset_seed()
        r = ik.solve_pose(GoalToolPose.from_poses({"fr3_link7": Pose(position=pp[pad].contiguous(), quaternion=pq[pad].contiguous())},
                                                  num_goalset=1), return_seeds=1)
        succ.append(r.success.view(1992)[: e - s]), sol.append(r.js_solution.position.view(1992, -1)[: e - s])
    succ, sol = torch.cat(succ), torch.cat(sol)
    ok = torch.nonzero(succ).squeeze(-1)
    ok = ok[torch.randperm(len(ok), device=dev, generator=gen)]
    sel = idx[ok]
    tmp = f"{OUT}/.{stem}.npz"
    np.savez(tmp, pos=pos[sel].float().cpu().numpy(), quat=quat[sel].float().cpu().numpy(), grasp=grasp[sel].int().cpu().numpy(),
             q=sol[ok].float().cpu().numpy())
    os.rename(tmp, f"{OUT}/{stem}.npz")
    st = dict(stem=stem, grasps=n_g, poses=a.n, redrawn=redrawn, hand_ok=len(idx), reached=len(ok), seconds=round(time.time() - t0, 1),
              worker=a.worker, host=os.uname().nodename)
    json.dump(st, open(f"{OUT}/claims/{stem}/stats.json", "w"))
    print(f"[w{a.worker}] {stem} grasps {n_g} hand_ok {len(idx)} reached {len(ok)} in {st['seconds']} s", flush=True)


def listed(d, suffix):
    return {f[: -len(suffix)] if suffix else f for f in os.listdir(d) if f.endswith(suffix) and not f.startswith(".")}


# One listing per directory per pass, never a stat per object, to keep lookups off the dj NAS.
while True:
    did = 0
    finished, delivered, claimed = listed(OUT, ".npz"), listed(DEL, ".npz"), listed(f"{OUT}/claims", "")
    remaining = [s for s in stems if s not in finished]
    if not remaining:
        break
    for s in remaining:
        if s in claimed or s not in delivered:
            continue
        try:
            os.mkdir(f"{OUT}/claims/{s}")  # atomic claim, another worker got it first when this raises
        except FileExistsError:
            continue
        run(s)
        did += 1
    if did == 0:
        time.sleep(120)
print(f"[w{a.worker}] all objects done", flush=True)
