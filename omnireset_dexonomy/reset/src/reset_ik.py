"""S3. Arm IK with cuRobo v2 for every kept pose and grasp pair.

S3. One object pose fixes each grasp's palm pose, so the IK target is FR3 link 7 at palm pose times
link7-to-palm inverse, in the robot base frame. cuRobo solves the batch of wrists with the table box
as obstacle, self-collision on, 32 seeds, and returns every seed. Arm spheres are cricket's foam fit.
The object filter, S4, runs in reset_arm_object.py, since cuRobo's collision checker in the same process
corrupted the IK solver's CUDA graph replay on the second pose.
Writes one npz per pose range with every successful seed.
"""
import argparse
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

import reset_common as rc

ap = argparse.ArgumentParser()
ap.add_argument("--poses", required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=32)
ap.add_argument("--pos-tol", type=float, default=0.001)
ap.add_argument("--rot-tol", type=float, default=0.01)
a = ap.parse_args()
rc.check_tree()
dev = torch.device("cuda")
t0 = time.time()

# Arm-only URDF cut from the training URDF, links 0 to 8, no geometry.
u = rc.Urdf()
arm = ET.Element("robot", name="fr3_arm")
keep = [f"fr3_link{i}" for i in range(9)]
for l in keep:
    ET.SubElement(arm, "link", name=l)
for j in u.root.findall("joint"):
    if j.find("parent").get("link") in keep and j.find("child").get("link") in keep:
        arm.append(j)
arm_urdf = f"{a.out}.fr3_arm.urdf"
ET.ElementTree(arm).write(arm_urdf)

links = [f"fr3_link{i}" for i in range(8)]
sph = yaml.safe_load(open(rc.SPHERES))["collision_spheres"]
ignore = {"fr3_link0": ["fr3_link1", "fr3_link2"], "fr3_link1": ["fr3_link2", "fr3_link3", "fr3_link4"],
          "fr3_link2": ["fr3_link3", "fr3_link4"], "fr3_link3": ["fr3_link4", "fr3_link6"],
          "fr3_link4": ["fr3_link5", "fr3_link6", "fr3_link7"], "fr3_link5": ["fr3_link6", "fr3_link7"],
          "fr3_link6": ["fr3_link7"]}  # cuRobo franka.yml pairs, renamed
robot = {"robot_cfg": {"kinematics": {
    "format_version": 2.0, "urdf_path": arm_urdf, "asset_root_path": "", "base_link": "fr3_link0",
    "tool_frames": ["fr3_link7"], "collision_link_names": links,
    "collision_spheres": {l: [{"center": s["center"], "radius": s["radius"]} for s in sph[l]] for l in links},
    "collision_sphere_buffer": 0.0, "self_collision_ignore": ignore,
    "self_collision_buffer": {l: 0.0 for l in links}, "lock_joints": None,
    "cspace": {"joint_names": rc.ARM_JOINTS, "default_joint_position": rc.ARM_DEFAULT,
               "null_space_weight": [1.0] * 7, "cspace_distance_weight": [1.0] * 7,
               "max_acceleration": 10.0, "max_jerk": 500.0}}}}

# Table box in the base frame.
T_wb = rc.homog(rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)), torch.tensor(rc.BASE_POS, dtype=torch.float64))
T_bw = torch.linalg.inv(T_wb)
tc = T_bw @ torch.tensor([rc.TABLE_OFF[0], rc.TABLE_OFF[1], rc.TABLE_Z, 1.0], dtype=torch.float64)
tq = rc.mat_to_quat(T_bw[:3, :3])
table = {"cuboid": {"table": {"dims": list(rc.TABLE_BOX), "pose": tc[:3].tolist() + tq.tolist()}}}
print("table in base frame", table, flush=True)

P = np.load(a.poses)
G = rc.load_grasps(dev)
n_g = G["hand_pos"].shape[0]
ik = InverseKinematics(InverseKinematicsCfg.create(
    robot=robot, scene_model=table, num_seeds=a.seeds, self_collision_check=True,
    position_tolerance=a.pos_tol, orientation_tolerance=a.rot_tol, max_batch_size=n_g // 2))
assert n_g % 2 == 0
assert list(ik.tool_frames) == ["fr3_link7"], ik.tool_frames
yaml.safe_dump(robot, open(f"{a.out}.robot.yml", "w"))
yaml.safe_dump(table, open(f"{a.out}.table.yml", "w"))
print(f"setup {time.time() - t0:.1f} s", flush=True)

T_7p = torch.as_tensor(rc.t_7_palm(u), device=dev)
T_obj_palm = rc.homog(rc.quat_to_mat(G["hand_quat"].double()), G["hand_pos"].double())
T_bw = T_bw.to(dev)
out = {k: [] for k in ("pose", "grasp", "seed", "q", "pos_err", "rot_err")}
stats = []
for p in range(a.start, a.end):
    tp = time.time()
    # Every pose solves all grasps in two fixed halves. cuRobo's CUDA graph replay crashed at one
    # batch of 3,984 by 32 seeds and ran clean at 2,000. Pairs that failed the hand check are
    # solved and then dropped.
    hand_ok = torch.as_tensor(P["hand_clear"][p] >= 0, device=dev)
    gi = torch.arange(n_g, device=dev)
    T_wm = rc.homog(rc.quat_to_mat(torch.as_tensor(P["quat"][p], device=dev)), torch.as_tensor(P["pos"][p], device=dev))
    T_b7 = T_bw @ T_wm @ T_obj_palm[gi] @ torch.linalg.inv(T_7p)
    goal = Pose(position=T_b7[:, :3, 3].float().contiguous(), quaternion=rc.mat_to_quat(T_b7[:, :3, :3]).float().contiguous())
    h = n_g // 2
    parts = []
    for s0 in (0, h):
        # The seeds come from a Halton sequence that advances every call, and four identical solves
        # lost 34 percent of their successful seeds. Resetting gives every half the same seed set.
        ik.reset_seed()
        g = Pose(position=goal.position[s0:s0 + h].contiguous(), quaternion=goal.quaternion[s0:s0 + h].contiguous())
        r = ik.solve_pose(GoalToolPose.from_poses({"fr3_link7": g}, num_goalset=1), return_seeds=a.seeds)
        parts.append((r.success.view(h, a.seeds), r.js_solution.position.view(h, a.seeds, -1),
                      r.position_error.view(h, a.seeds), r.rotation_error.view(h, a.seeds)))
    succ, qs, pe, re_ = (torch.cat(x) for x in zip(*parts))
    bi, si = torch.nonzero(succ & hand_ok[:, None], as_tuple=True)
    qk = qs[bi, si]
    out["pose"].append(np.full(len(bi), p, np.int16)); out["grasp"].append(gi[bi].short().cpu().numpy())
    out["seed"].append(si.byte().cpu().numpy()); out["q"].append(qk.float().cpu().numpy())
    out["pos_err"].append(pe[bi, si].half().cpu().numpy()); out["rot_err"].append(re_[bi, si].half().cpu().numpy())
    stats.append((p, int(hand_ok.sum()), int((succ & hand_ok[:, None]).any(-1).sum()), len(bi), time.time() - tp))
    if (p - a.start) % 50 == 0:
        print(f"mem pose {p} allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
    if p < a.start + 3 or (p - a.start) % 100 == 0:
        print(f"pose {p} pairs {stats[-1][1]} wrists solved {stats[-1][2]} solutions {len(bi)} in {stats[-1][4]:.2f} s", flush=True)
np.savez(a.out, **{k: np.concatenate(v) for k, v in out.items()}, stats=np.array(stats))
print(f"wrote {a.out} poses {a.start}-{a.end} in {time.time() - t0:.1f} s", flush=True)
