"""Figures for S1 to S3 from stats.npz, plus renders of chosen cases. Mechanical plots, no judgments.
Usage: reset_figs_ik.py <reset dir> <out dir>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

import reset_common as rc
import reset_render as rr

D, OUT = sys.argv[1:3]
S = np.load(f"{D}/stats.npz")
P = np.load(f"{D}/poses.npz")
G = np.load(rc.GRASPS)
pos, quat = S["pose_pos"], S["pose_quat"]
types, tid = list(S["types"]), S["tid"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def table_outline(ax, plane):
    x0, x1 = -0.45, 0.45
    y0, y1 = -0.09 - 0.45, -0.09 + 0.45
    if plane == "xy":
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#b07a4f", lw=1.5, label="table top")
        ax.plot([-0.35, 0.35, 0.35, -0.35, -0.35], [-0.2, -0.2, 0.2, 0.2, -0.2], "k--", lw=1, label="goal box")
        ax.plot(0, 0.48, "ks", ms=9, label="robot base")
    else:
        ax.axhline(rc.TABLE_TOP, color="#b07a4f", lw=1.5, label="table top z 0.53")
        ax.plot([-0.35, 0.35, 0.35, -0.35, -0.35], [0.6, 0.6, 0.95, 0.95, 0.6], "k--", lw=1, label="goal box")


# S1 positions
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.35, color="#3060c0")
table_outline(ax, "xy")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"S1. {len(pos):,} object positions, top view")
save(fig, "s1_positions_top")
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(pos[:, 0], pos[:, 2], s=1, alpha=0.35, color="#3060c0")
table_outline(ax, "xz")
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.legend(loc="lower left", fontsize=8)
ax.set_title("S1. object positions, side view")
save(fig, "s1_positions_side")
obj_v = trimesh.util.concatenate(rc.object_pieces()).convex_hull.vertices
low = pos[:, 2] + np.einsum("pk,vk->pv", rc.quat_to_mat(torch.as_tensor(quat)).numpy()[:, 2, :], obj_v).min(-1)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist((low - rc.TABLE_TOP) * 100, bins=80, color="#3060c0")
ax.set_xlabel("object lowest point above the table top (cm), kept poses"); ax.set_ylabel("poses")
ax.set_title(f"S1. {int(S['n_rejected'])} of {int(S['n_drawn']):,} drawn poses fell below the table and were redrawn")
save(fig, "s1_clearance")

# S2
hand_ok = S["hand_ok"]
drop = 1 - hand_ok.mean(1)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.scatter(pos[:, 2], drop * 100, s=2, alpha=0.3, color="#c05030")
bins = np.linspace(0.6, 0.95, 15)
b = np.digitize(pos[:, 2], bins)
ax.plot([(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))], [drop[b == i].mean() * 100 for i in range(1, len(bins))], "k-", lw=2, label="bin mean")
ax.set_xlabel("object centre height z (m)"); ax.set_ylabel("grasps dropped (%)"); ax.legend()
ax.set_title(f"S2. hand below the table top, {100 * (1 - hand_ok.mean()):.1f}% of pose and grasp pairs")
save(fig, "s2_drop_vs_height")

# S3
solved = S["solved"]
rate = solved.sum(1) / np.maximum(hand_ok.sum(1), 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (i, j, lab) in zip(axs, [(0, 1, "y"), (0, 2, "z")]):
    sc = ax.scatter(pos[:, i], pos[:, j], c=rate * 100, s=3, cmap="viridis", vmin=0, vmax=100)
    ax.set_xlabel("x (m)"); ax.set_ylabel(f"{lab} (m)")
    if lab == "y":
        ax.plot(0, 0.48, "ks", ms=8); ax.set_aspect("equal")
fig.colorbar(sc, ax=axs, label="wrists IK reached (%)")
fig.suptitle(f"S3. IK success per object pose, {100 * solved.sum() / hand_ok.sum():.1f}% of hand-clear pairs overall")
save(fig, "s3_success_box")
per_type = np.array([solved[:, tid == k].sum() / max(hand_ok[:, tid == k].sum(), 1) for k in range(len(types))])
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.bar(range(len(types)), per_type * 100, color="#3a7d44")
ax.set_xticks(range(len(types)), types, rotation=60, ha="right", fontsize=8); ax.set_ylabel("IK reached (%)")
ax.set_title("S3. IK success per grasp type, over all object poses")
save(fig, "s3_success_per_type")
h = S["per_wrist_hist"]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(range(len(h)), h, color="#3a7d44")
ax.set_xlabel("successful seeds per wrist target (of 32)"); ax.set_ylabel("pose and grasp pairs")
ax.set_title(f"S3. {int(S['n_solutions']):,} arm solutions in total")
save(fig, "s3_solutions_per_wrist")
lim = [[-2.9007, 2.9007], [-1.8361, 1.8361], [-2.9007, 2.9007], [-3.0770, -0.1169], [-2.8763, 2.8763], [0.4398, 4.6216], [-3.0508, 3.0508]]
qs = S["q_sample"]
fig, axs = plt.subplots(1, 7, figsize=(15, 2.8), sharey=True)
for k, ax in enumerate(axs):
    ax.hist(qs[:, k], bins=60, color="#3a7d44")
    for x in lim[k]:
        ax.axvline(x, color="k", lw=1)
    ax.axvline(rc.ARM_DEFAULT[k], color="#c05030", lw=1, ls="--")
    ax.set_title(f"joint {k + 1}", fontsize=9)
fig.suptitle(f"S3. arm joint spread over a random {len(qs):,} solutions, black the training limits, red dashed the default pose")
save(fig, "s3_joint_spread")
pe, re_ = S["perr_sample"] * 1000, np.degrees(S["rerr_sample"])
print(f"S3 position error mm p50 {np.median(pe):.3f} p99 {np.percentile(pe, 99):.3f} max {pe.max():.3f}; rotation deg p50 {np.median(re_):.3f} p99 {np.percentile(re_, 99):.3f} max {re_.max():.3f}")

# Renders. Chosen by rule, not by look.
cam = dict(azimuth=138.8, elevation=-22.0)
rng = np.random.default_rng(0)
lowest = int(np.argmin(pos[:, 2]))
hc = P["hand_clear"][lowest]
g_drop, g_keep = int(np.argmin(hc)), int(np.argmax(hc))
T = rc.homog(rc.quat_to_mat(torch.as_tensor(quat[lowest])), torch.as_tensor(pos[lowest]))
for g, name in [(g_drop, "s2_dropped_pair"), (g_keep, "s2_kept_pair")]:
    # a floating hand, no arm, at the grasp on the lowest object pose
    Tp = T @ rc.homog(rc.quat_to_mat(torch.as_tensor(G["hand_quat"][g]).double()), torch.as_tensor(G["hand_pos"][g]).double())
    spec_q = np.zeros(19)
    spec_q[7:] = G["q_grasp"][g]
    m, d = rr.build([], (pos[lowest], quat[lowest]))
    # add the hand by hand, posed at Tp
    import mujoco
    s = mujoco.MjSpec()
    s.visual.global_.offwidth = s.visual.global_.offheight = 640
    s.visual.headlight.ambient = [0.45, 0.45, 0.45]
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[b / 2 for b in rc.TABLE_BOX], pos=[rc.TABLE_OFF[0], rc.TABLE_OFF[1], rc.TABLE_Z],
                         rgba=[0.82, 0.62, 0.45, 1], contype=0, conaffinity=0)
    qd = {j: torch.tensor([float(x)], dtype=torch.float64) for j, x in zip(rc.HAND_JOINTS, G["q_grasp"][g])}
    for link, Tl in rr.U.fk_from("palm", qd).items():
        for i, mm in enumerate(rr.link_meshes(link)):
            rr.add_mesh(s, f"h_{link}_{i}", mm, (Tp @ Tl[0]).numpy(), rr.HAND_DARK)
    Tm = T.numpy()
    rr.add_mesh(s, "obj", rr.object_mesh(), Tm, rr.OBJ_BLUE)
    m = s.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    img = rr.render(m, d, pos[lowest], 0.45, 150.0, -8.0)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(img); ax.axis("off")
    ax.set_title(f"{G['tmpl'][g]}, hand lowest point {100 * hc[g]:+.1f} cm from the table top", fontsize=9)
    save(fig, name)

# S3 overlay, the pose and grasp pair with the most successful seeds, every seed drawn translucent.
sh = np.load(f"{D}/ik/shard0.npz")
key = sh["pose"].astype(np.int64) * 3984 + sh["grasp"]
u_, cnt = np.unique(key, return_counts=True)
k0 = u_[np.argmax(cnt)]
sel = np.nonzero(key == k0)[0]
p0, g0 = int(k0 // 3984), int(k0 % 3984)
robots = [(np.concatenate([sh["q"][i], G["q_grasp"][g0]]), 0.25) for i in sel]
m, d = rr.build(robots, (pos[p0], quat[p0]))
img = rr.render(m, d, [0.0, 0.2, 0.75], 1.7, **cam)
fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(img); ax.axis("off")
ax.set_title(f"S3. {len(sel)} arm solutions on one wrist, pose {p0}, {G['tmpl'][g0]}", fontsize=9)
save(fig, "s3_overlay")

print("figures written to", OUT)
