"""Galleries of what each step dropped, chosen at random with seed 0, never by look.

S1, the poses the table clearance check rejected, replayed from the sampler's seed.
S2, pairs whose hand fell below the table. S3, hand-clear pairs where no IK seed reached the wrist,
with the robot at its default pose for scale.
Usage: reset_figs_fail.py <reset dir> <out dir>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import trimesh

import reset_common as rc
import reset_render as rr

D, OUT = sys.argv[1:3]
S, P, G = np.load(f"{D}/stats.npz"), np.load(f"{D}/poses.npz"), np.load(rc.GRASPS)
pos, quat = P["pos"], P["quat"]
rng = np.random.default_rng(0)
CAM = dict(azimuth=138.8, elevation=-22.0)


def hand_scene(p_pos, p_quat, g, robot_q=None, obj_rgba=rr.OBJ_BLUE):
    s = mujoco.MjSpec()
    s.visual.global_.offwidth = s.visual.global_.offheight = 640
    s.visual.headlight.ambient = [0.45, 0.45, 0.45]
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[b / 2 for b in rc.TABLE_BOX],
                         pos=[rc.TABLE_OFF[0], rc.TABLE_OFF[1], rc.TABLE_Z], rgba=[0.82, 0.62, 0.45, 1], contype=0, conaffinity=0)
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.125, 0.125, 0.15], pos=[0, 0.515, rc.TABLE_Z],
                         rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)
    T = rc.homog(rc.quat_to_mat(torch.as_tensor(p_quat)), torch.as_tensor(p_pos)).numpy()
    if g is not None:
        Tp = T @ rc.homog(rc.quat_to_mat(torch.as_tensor(G["hand_quat"][g]).double()), torch.as_tensor(G["hand_pos"][g]).double()).numpy()
        qd = {j: torch.tensor([float(x)], dtype=torch.float64) for j, x in zip(rc.HAND_JOINTS, G["q_grasp"][g])}
        for link, Tl in rr.U.fk_from("palm", qd).items():
            for i, mm in enumerate(rr.link_meshes(link)):
                rr.add_mesh(s, f"h_{link}_{i}", mm, Tp @ Tl[0].numpy(), rr.HAND_DARK)
    if robot_q is not None:
        for link, Tl in rr.robot_link_poses(robot_q).items():
            for i, mm in enumerate(rr.link_meshes(link)):
                rr.add_mesh(s, f"r_{link}_{i}", mm, Tl, (rr.ARM_GREY if link.startswith("fr3_") else rr.HAND_DARK)[:3] + [0.35])
    rr.add_mesh(s, "obj", rr.object_mesh(), T, obj_rgba)
    m = s.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def gallery(items, name, title):
    """items, list of (image, caption)."""
    n = len(items)
    cols = 5 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.5))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (img, cap) in zip(np.ravel(axs), items):
        ax.imshow(img)
        ax.set_title(cap, fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    fig.savefig(f"{OUT}/{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# S1 rejected poses, replaying reset_sample.py's draws with its seed.
dev = torch.device("cuda")
obj_v = torch.as_tensor(trimesh.util.concatenate(rc.object_pieces()).convex_hull.vertices, device=dev)
gen = torch.Generator(device=dev).manual_seed(0)
lo, hi = torch.tensor(rc.BOX_MIN, device=dev, dtype=torch.float64), torch.tensor(rc.BOX_MAX, device=dev, dtype=torch.float64)
rej, drawn = [], 0
while drawn < int(S["n_drawn"]):
    p = lo + (hi - lo) * torch.rand(4096, 3, device=dev, generator=gen, dtype=torch.float64)
    q = torch.nn.functional.normalize(torch.randn(4096, 4, device=dev, generator=gen, dtype=torch.float64), dim=-1)
    low = p[:, 2] + (rc.quat_to_mat(q)[:, 2, :] @ obj_v.T).min(-1).values
    for i in torch.nonzero(low < rc.TABLE_TOP).squeeze(-1).tolist():
        rej.append((p[i].cpu().numpy(), q[i].cpu().numpy(), float(low[i] - rc.TABLE_TOP)))
    drawn += 4096
assert len(rej) == int(S["n_rejected"]), (len(rej), int(S["n_rejected"]))
items = []
for p_, q_, below in rej:
    m, d = hand_scene(p_, q_, None)
    items.append((rr.render(m, d, [p_[0], p_[1], 0.56], 0.5, 180.0, -5.0), f"centre z {p_[2]:.3f} m, lowest point {100 * below:+.1f} cm"))
gallery(items, "s1_rejected", f"S1. all {len(rej)} poses rejected for table clearance, side view at table height")

# S2 hand below the table.
pp, gg = np.nonzero(~S["hand_ok"])
items = []
for k in rng.choice(len(pp), 6, replace=False):
    p_, g_ = pp[k], gg[k]
    m, d = hand_scene(pos[p_], quat[p_], g_)
    items.append((rr.render(m, d, [pos[p_][0], pos[p_][1], 0.58], 0.55, 150.0, -8.0),
                  f"pose {p_}, {G['tmpl'][g_]}\nhand {100 * P['hand_clear'][p_, g_]:+.1f} cm from the table top"))
gallery(items, "s2_dropped_gallery", "S2. six random pairs with the hand below the table top")

# S3 IK found no arm pose for these hand-clear wrists.
pp, gg = np.nonzero(S["hand_ok"] & ~S["solved"])
items = []
for k in rng.choice(len(pp), 6, replace=False):
    p_, g_ = pp[k], gg[k]
    m, d = hand_scene(pos[p_], quat[p_], g_, robot_q=np.array(rc.ARM_DEFAULT + [0.0] * 12))
    items.append((rr.render(m, d, [0.0, 0.2, 0.75], 1.7, **CAM), f"pose {p_}, {G['tmpl'][g_]}"))
gallery(items, "s3_ik_fail_gallery", "S3. six random hand-clear wrists with no IK solution, robot at its default pose translucent")

print("galleries written")
