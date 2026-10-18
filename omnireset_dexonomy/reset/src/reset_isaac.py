"""S5. Test candidate reset states in the 050 training scene, OmniReset's check for a grasped object.

The env is SimToolReal as train.py builds it, dataclass defaults plus the task yaml, with the object as
its single object. Each env gets one candidate. The arm starts at its IK angles and holds them as
targets, the fingers start at the grasp angles and target the squeeze angles clamped to the joint
limits training clamps to, the object starts at its pose at rest, gravity is on. Physics runs at 1/120 s
with 2 substeps per step, bypassing the policy action pipeline.
A state passes at the first step where, as in OmniReset's check_reset_state_success, it has been
stable for the required run, the object and end effector are within their drift limits, the object centre
is above the table top, and no joint exceeds twice its velocity limit. Failing states time out.
One named deviation from OmniReset. The penetration limit applies to the start state. OmniReset tests
it at the success step, where our squeeze presses fingertips 0.4 to 2.3 mm into the object under
training's 8 solver iterations, against OmniReset's 192. The start states sit 0.17 mm or more outside.
The settled-step depth is still recorded per state, so the OmniReset verdict can be read back.
Robot points are 1,024 per body, farthest-point sampled on the URDF collision meshes as OmniReset
samples them. The object pieces are convex, so a point's depth inside a piece is exact from its planes.
Usage: reset_isaac.py --start <candidate row> --batches <n> --out <dir>
Full set: reset_isaac.py --stem <pool stem> --pool-dir <dir> --grasps <npz> --ik <npz> --final <npz> --start 0 --batches 1000
--target 10000 --slim --out <dir>. The IK file already holds the reached rows in their random test order.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--reset-dir", default="/home/nas_main/sibeenkim/draftrec/_curobo2/reset")
ap.add_argument("--cands", default=None, help="candidates npz, default <reset-dir>/candidates.npz")
ap.add_argument("--poses", default=None, help="poses npz, default <reset-dir>/poses.npz")
ap.add_argument("--stem", default=None, help="full set, the pool stem, with --pool-dir, --grasps and --ik")
ap.add_argument("--pool-dir", default=None, help="full set, dir holding <stem>.urdf and pool.json")
ap.add_argument("--grasps", default=None, help="full set, the object's delivered passes npz")
ap.add_argument("--ik", default=None, help="full set, the object's reset_ik_all.py output, replaces --cands and --poses")
ap.add_argument("--final", default=None, help="full set, write the first --keep passing states here")
ap.add_argument("--keep", type=int, default=10000)
ap.add_argument("--slim", action="store_true", help="leave the per-step recordings out of the batch files")
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--batches", type=int, required=True, help="most batches to run, the pool size bounds it too")
ap.add_argument("--target", type=int, default=0, help="stop after the batch that brings passes to this many, 0 runs every batch")
ap.add_argument("--out", required=True)
ap.add_argument("--num-envs", type=int, default=2048)
ap.add_argument("--steps", type=int, default=120)  # 2 s at 60 Hz
ap.add_argument("--obj-drift", type=float, default=0.05)
ap.add_argument("--ee-drift", type=float, default=0.1)
ap.add_argument("--stable-steps", type=int, default=30)  # 0.5 s, OmniReset's 5 steps of 0.1 s
ap.add_argument("--joint-speed", type=float, default=5.0)
ap.add_argument("--obj-lin-speed", type=float, default=0.1)
ap.add_argument("--obj-ang-speed", type=float, default=1.0)
ap.add_argument("--penetration", type=float, default=0.0005)
ap.add_argument("--points", type=int, default=1024)
ap.add_argument("--diag", action="store_true", help="also record per-body depth at the start state and at the end")
ap.add_argument("--log-vel", action="store_true", help="diagnostic, record the raw speed readings the stability test sums")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.headless = True
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces  # noqa: E402

import isaacsimenvs.tasks  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg  # noqa: E402

import reset_common as rc  # noqa: E402

out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
t0 = time.time()
rc.check_tree()
if a.stem:
    obj_urdf = f"{a.pool_dir}/{a.stem}.urdf"
    obj_scale = tuple(next(e["scale"] for e in json.load(open(f"{a.pool_dir}/pool.json")) if e["urdf"].endswith(f"/{a.stem}.urdf")))
    ik_file = np.load(a.ik)
    C = {"pose": np.arange(len(ik_file["q"])), "grasp": ik_file["grasp"], "q": ik_file["q"]}
    P = {"pos": ik_file["pos"].astype(np.float64), "quat": ik_file["quat"].astype(np.float64)}  # saved as float32, the transforms here are float64
    G = np.load(a.grasps)
else:
    obj_urdf, obj_scale = rc.OBJ_URDF, rc.OBJ_SCALE
    C = np.load(a.cands or f"{a.reset_dir}/candidates.npz")
    P = np.load(a.poses or f"{a.reset_dir}/poses.npz")
    G = np.load(rc.GRASPS)

cfg = replace_env_cfg_spaces_with_strings(SimToolRealEnvCfg())
cfg.from_dict(yaml.safe_load(open(rc.TREE / "isaacsimenvs/cfg/task/SimToolReal.yaml")))
cfg = replace_strings_with_env_cfg_spaces(cfg)
assert cfg.robot == "fr3-xhand" and cfg.assets.table_urdf == "assets/urdf/table_davian.urdf"
assert cfg.sim.dt == 0.00833333333 and cfg.decimation == 2, (cfg.sim.dt, cfg.decimation)
cfg.scene.num_envs = a.num_envs
cfg.assets.object_urdf = str(obj_urdf)
cfg.assets.object_scale = obj_scale
cfg.reset.table_reset_z_range = 0.0
env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg).unwrapped
env.reset()
N, dev = env.num_envs, env.device
origins = env.scene.env_origins
jid = env.robot.find_joints(rc.ARM_JOINTS + rc.HAND_JOINTS, preserve_order=True)[0]
assert len(jid) == 19 and env.robot.num_joints == 19
ee = env.robot.find_bodies("fr3_link7")[0]
assert len(ee) == 1
ee = ee[0]
limits = env.robot.data.joint_pos_limits[:, jid]  # the limits action_utils clamps targets to
vel_lim = env.robot.data.joint_vel_limits
print(f"[isaac] env up in {time.time() - t0:.0f} s, bodies {env.robot.body_names}", flush=True)

# Robot surface points per Isaac body, from the URDF collision meshes merged into it.
u = rc.Urdf()
body_pts, body_ids = [], []
for bi, b in enumerate(env.robot.body_names):
    meshes = []
    for link in u.links:
        anc, T = link, np.eye(4)
        while anc not in env.robot.body_names:
            j = u.child_joint[anc]
            assert j.get("type") == "fixed", (link, anc)
            T = rc.origin_T(j) @ T
            anc = j.find("parent").get("link")
        if anc != b:
            continue
        for c in u.links[link].findall("collision"):
            m = trimesh.load((rc.URDF.parent / c.find("geometry/mesh").get("filename")).resolve(), force="mesh")
            m.apply_transform(T @ rc.origin_T(c))
            meshes.append(m)
    if not meshes:
        continue
    pts = torch.as_tensor(trimesh.sample.sample_surface(trimesh.util.concatenate(meshes), 2 * a.points, seed=42)[0], device=dev)
    sel = [0]
    d = (pts - pts[0]).norm(dim=-1)
    for _ in range(a.points - 1):
        k = int(d.argmax())
        sel.append(k)
        d = torch.minimum(d, (pts - pts[k]).norm(dim=-1))
    body_pts.append(pts[sel].float())
    body_ids.append(bi)
body_pts = torch.stack(body_pts)  # (nb, points, 3)
print(f"[isaac] {len(body_ids)} robot bodies with colliders, {a.points} points each", flush=True)

# Object piece planes in the object frame, n.x - d > 0 outside.
planes = []
pieces = rc.object_pieces(obj_urdf)
for h in pieces:
    n = torch.as_tensor(np.array(h.face_normals), device=dev)
    planes.append(torch.cat([n, (n * torch.as_tensor(h.triangles[:, 0], device=dev)).sum(-1, keepdim=True)], -1))
F = max(len(p) for p in planes)
K = len(planes)
planes = torch.stack([torch.cat([p, p[:1].expand(F - len(p), 4)]) for p in planes]).float()  # (K, F, 4)
plane_n, plane_d = planes[..., :3].reshape(-1, 3).T.contiguous(), planes[..., 3].reshape(-1)
print(f"[isaac] object pieces {K}, planes per piece up to {F}", flush=True)


R_OBJ = float(np.linalg.norm(trimesh.util.concatenate(pieces).vertices, axis=1).max())


def deepest_per_body(env_ids):
    """(len(env_ids), n bodies) signed depth of each body's deepest point, for diagnostics."""
    bp = env.robot.data.body_link_pos_w[env_ids][:, body_ids]
    bq = env.robot.data.body_link_quat_w[env_ids][:, body_ids]
    w = (rc.quat_to_mat(bq)[:, :, None] @ body_pts[None, :, :, :, None]).squeeze(-1) + bp[:, :, None]
    om = rc.quat_to_mat(env.object.data.root_quat_w[env_ids])
    loc = (w - env.object.data.root_pos_w[env_ids][:, None, None]) @ om[:, None]
    out_b = torch.empty(len(env_ids), len(body_ids), device=dev)
    for i in range(len(env_ids)):
        x = loc[i].reshape(-1, 3)
        out_b[i] = (x @ plane_n - plane_d).view(len(x), K, F).max(-1).values.min(-1).values.view(len(body_ids), -1).min(-1).values
    return out_b


def deepest(env_ids):
    """Signed depth of the deepest robot point inside any object piece, per env, negative is inside.

    Points farther from the object origin than its farthest vertex cannot be inside, so only the rest are
    tested against the piece planes. Envs with no point that close read +inf.
    """
    out_d = torch.full((len(env_ids),), float("inf"), device=dev)
    for s in range(0, len(env_ids), 64):
        e = env_ids[s:s + 64]
        bp = env.robot.data.body_link_pos_w[e][:, body_ids]
        bq = env.robot.data.body_link_quat_w[e][:, body_ids]
        w = (rc.quat_to_mat(bq)[:, :, None] @ body_pts[None, :, :, :, None]).squeeze(-1) + bp[:, :, None]
        om = rc.quat_to_mat(env.object.data.root_quat_w[e])
        loc = (w.reshape(len(e), -1, 3) - env.object.data.root_pos_w[e][:, None]) @ om  # into the object frame
        near = loc.norm(dim=-1) <= R_OBJ
        ei, pi = torch.nonzero(near, as_tuple=True)
        for c in range(0, len(ei), 2000):  # 2,000 points against up to 16 x 7,354 planes fits two runs on a 24 GB GPU
            x = loc[ei[c:c + 2000], pi[c:c + 2000]]
            d = (x @ plane_n - plane_d).view(len(x), K, F).max(-1).values.min(-1).values
            out_d[s:s + 64].scatter_reduce_(0, ei[c:c + 2000], d, reduce="amin")
    return out_d


T_7p = torch.as_tensor(rc.t_7_palm(u), device=dev)
T_obj_palm = rc.homog(rc.quat_to_mat(torch.as_tensor(G["hand_quat"], device=dev).double()), torch.as_tensor(G["hand_pos"], device=dev).double())
q_grasp = torch.as_tensor(G["q_grasp"], device=dev)
q_squeeze = torch.as_tensor(G["q_squeeze"], device=dev)
zero7 = torch.zeros(N, 6, device=dev)
passed_total = 0
tested_rows, tested_ok = [], []
for b in range(a.batches):
    tb = time.time()
    rows = np.arange(a.start + b * N, a.start + (b + 1) * N)
    rows = rows[rows < len(C["pose"])]
    if len(rows) == 0:
        print(f"[isaac] pool used up after {b} batches, {passed_total} passed", flush=True)
        break
    n = len(rows)
    idx = torch.as_tensor(np.concatenate([rows, np.full(N - n, rows[0])]), device=dev)
    pose = torch.as_tensor(C["pose"], device=dev)[idx]
    grasp = torch.as_tensor(C["grasp"], device=dev)[idx]
    q_arm = torch.as_tensor(C["q"], device=dev)[idx]
    pos = torch.as_tensor(P["pos"], device=dev)[pose]
    quat = torch.as_tensor(P["quat"], device=dev)[pose]
    jp = env.robot.data.default_joint_pos.clone()
    jp[:, jid] = torch.cat([q_arm, q_grasp[grasp]], -1)
    tgt = jp.clone()
    tgt[:, jid[7:]] = torch.clamp(q_squeeze[grasp], limits[:, 7:, 0], limits[:, 7:, 1])
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    env.robot.set_joint_position_target(tgt)
    env.object.write_root_pose_to_sim(torch.cat([pos.float() + origins, quat.float()], -1))
    env.object.write_root_velocity_to_sim(zero7)
    env.table.write_root_pose_to_sim(torch.cat([origins + torch.tensor([0.0, 0.0, rc.TABLE_Z], device=dev),
                                                torch.tensor([1.0, 0, 0, 0], device=dev).expand(N, 4)], -1))
    env.goal_viz.write_root_pose_to_sim(torch.cat([origins + torch.tensor([0.0, 0.0, -5.0], device=dev),
                                                   torch.tensor([1.0, 0, 0, 0], device=dev).expand(N, 4)], -1))
    T_w7 = rc.homog(rc.quat_to_mat(quat), pos) @ T_obj_palm[grasp] @ torch.linalg.inv(T_7p)
    ee0, obj0 = T_w7[:, :3, 3].float(), pos.float()

    env.sim.forward()
    env.scene.update(dt=0.0)
    start_d = deepest(torch.arange(N, device=dev))
    start_ok = start_d >= -a.penetration
    if a.diag:
        start_depth = deepest_per_body(torch.arange(min(n, 64), device=dev))
    counter = torch.zeros(N, dtype=torch.long, device=dev)
    max_counter = torch.zeros(N, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    success_step = torch.full((N,), -1, dtype=torch.long, device=dev)
    abnormal = torch.zeros(N, dtype=torch.bool, device=dev)
    pen_seen = torch.full((N,), float("inf"), device=dev)
    rec_q = torch.empty(a.steps + 1, N, 19, device=dev)
    rec_o = torch.empty(a.steps + 1, N, 7, device=dev)
    rec_e = torch.empty(a.steps + 1, N, 3, device=dev)
    rec_q[0], rec_o[0], rec_e[0] = jp[:, jid], torch.cat([obj0, quat.float()], -1), ee0
    if a.log_vel:
        rec_jv = torch.zeros(a.steps, N, 19, device=dev)
        rec_ov = torch.zeros(a.steps, N, 6, device=dev)
    for st in range(a.steps):
        for _ in range(cfg.decimation):
            env.robot.set_joint_position_target(tgt)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        rd = env.robot.data
        opos = env.object.data.root_pos_w - origins
        rec_q[st + 1] = rd.joint_pos[:, jid]
        rec_o[st + 1] = torch.cat([opos, env.object.data.root_quat_w], -1)
        rec_e[st + 1] = rd.body_link_pos_w[:, ee] - origins
        abnormal |= (rd.joint_vel.abs() > 2 * vel_lim).any(-1)
        if a.log_vel:
            rec_jv[st] = rd.joint_vel[:, jid]
            rec_ov[st] = torch.cat([env.object.data.root_lin_vel_w, env.object.data.root_ang_vel_w], -1)
        stable = ((rd.joint_vel.abs().sum(-1) < a.joint_speed)
                  & (env.object.data.root_lin_vel_w.abs().sum(-1) < a.obj_lin_speed)
                  & (env.object.data.root_ang_vel_w.abs().sum(-1) < a.obj_ang_speed))
        counter = torch.where(stable, counter + 1, torch.zeros_like(counter))
        max_counter = torch.maximum(max_counter, counter)
        within = (((opos - obj0).norm(dim=-1) <= a.obj_drift) & ((rec_e[st + 1] - ee0).norm(dim=-1) <= a.ee_drift)
                  & (opos[:, 2] >= rc.TABLE_TOP))
        cand = (counter >= a.stable_steps) & within & ~abnormal & ~done & start_ok
        if cand.any():
            e = torch.nonzero(cand).squeeze(-1)
            pen_seen[e] = deepest(e)  # the settled-step depth OmniReset would test
            success_step[e] = st + 1
            done[e] = True
        if done[:n].all():
            rec_q, rec_o, rec_e = rec_q[:st + 2], rec_o[:st + 2], rec_e[:st + 2]
            break
    if a.diag:
        end_depth = deepest_per_body(torch.arange(min(n, 64), device=dev))
        np.savez(out / f"diag{b}.npz", start=start_depth.cpu().numpy(), end=end_depth.cpu().numpy(),
                 bodies=np.array([env.robot.body_names[i] for i in body_ids]))
    if a.log_vel:
        np.savez(out / f"vel{b}.npz", jv=rec_jv[:, :n].cpu().numpy(), ov=rec_ov[:, :n].cpu().numpy(), row=rows)
    obj_d = (rec_o[:, :, :3] - obj0).norm(dim=-1)
    ee_d = (rec_e - ee0).norm(dim=-1)
    np.savez(out / f"batch{b}.npz", row=rows, pose=C["pose"][rows], grasp=C["grasp"][rows],
             success=done[:n].cpu().numpy(), success_step=success_step[:n].cpu().numpy(), abnormal=abnormal[:n].cpu().numpy(),
             pen_start=start_d[:n].cpu().numpy(), pen_at_success=pen_seen[:n].cpu().numpy(), obj_drift_final=obj_d[-1, :n].cpu().numpy(), obj_drift_max=obj_d[:, :n].max(0).values.cpu().numpy(),
             ee_drift_final=ee_d[-1, :n].cpu().numpy(), ee_drift_max=ee_d[:, :n].max(0).values.cpu().numpy(),
             obj_z_final=rec_o[-1, :n, 2].cpu().numpy(), max_stable_run=max_counter[:n].cpu().numpy(),
             grasp_in_limits=((jp[:n, jid[7:]] >= limits[:n, 7:, 0]) & (jp[:n, jid[7:]] <= limits[:n, 7:, 1])).all(-1).cpu().numpy(),
             **({} if a.slim else dict(rec_q=rec_q[:, :n].half().cpu().numpy(), rec_o=rec_o[:, :n].cpu().numpy(),
                                       rec_e=rec_e[:, :n].half().cpu().numpy())))
    tested_rows.append(rows), tested_ok.append(done[:n].cpu().numpy())
    passed_total += int(done[:n].sum())
    print(f"[isaac] batch {b} rows {rows[0]}-{rows[-1]} passed {int(done[:n].sum())}/{n} steps {rec_q.shape[0] - 1} "
          f"in {time.time() - tb:.1f} s, total passed {passed_total}", flush=True)
    if a.target and passed_total >= a.target:
        print(f"[isaac] target {a.target} reached after {b + 1} batches", flush=True)
        break
if a.final:
    rows, ok = np.concatenate(tested_rows), np.concatenate(tested_ok)
    keep = rows[ok][:a.keep]
    gi = C["grasp"][keep].astype(np.int64)
    tmp = str(Path(a.final).with_name("." + Path(a.final).name))
    np.savez(tmp, obj_pos=P["pos"][keep], obj_quat_wxyz=P["quat"][keep], arm_q=C["q"][keep], hand_q=G["q_grasp"][gi],
             hand_target=G["q_squeeze"][gi], grasp=gi, candidate_row=keep, tested=len(rows), passed=int(ok.sum()),
             arm_joints=np.array(rc.ARM_JOINTS), hand_joints=np.array(rc.HAND_JOINTS),
             note=np.array("env-local frame of the simtoolreal-050 scene. hand_target is the squeeze, clamped to the joint limits at use."))
    os.rename(tmp, a.final)
    print(f"[isaac] final {len(keep)} states of {int(ok.sum())} passed in {len(rows)} tested, written", flush=True)
print(f"[isaac] done in {time.time() - t0:.0f} s", flush=True)
os._exit(0)  # app.close() hangs at full CPU in Isaac Sim 5.1
