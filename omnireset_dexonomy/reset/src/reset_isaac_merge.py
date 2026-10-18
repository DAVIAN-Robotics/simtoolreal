"""Pool the Isaac batches, write the final reset states, the S5 and S6 figures, the S5 failure gallery,
and the random pick of pages for the pose viewer.

The final set is the first 10,000 passing candidates in the random order the test walked them.
A failing state gets the first reason that applies, in this order. Start penetration, joint speed
past twice its limit, object moved past the drift limit, object below the table top, end effector moved,
and last never stable for the required run.
Usage: reset_isaac_merge.py <reset dir> <run dir> <out figs dir>
"""
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import reset_common as rc
import reset_render as rr

D, RUN, OUT = sys.argv[1:4]
P, G, C = np.load(f"{D}/poses.npz"), np.load(rc.GRASPS), np.load(f"{D}/candidates.npz")
files = sorted(glob.glob(f"{RUN}/l*/batch*.npz"), key=lambda f: int(np.load(f)["row"][0]))
B = [np.load(f) for f in files]
cat = {k: np.concatenate([b[k] for b in B]) for k in ("row", "pose", "grasp", "success", "success_step", "abnormal", "pen_start",
                                                       "pen_at_success", "obj_drift_final", "obj_drift_max", "ee_drift_final",
                                                       "ee_drift_max", "obj_z_final", "max_stable_run", "grasp_in_limits")}
assert (np.diff(cat["row"]) == 1).all() and cat["row"][0] == 0
ok = cat["success"]
reason = np.full(len(ok), "passed", dtype=object)
rules = [("robot inside the object at the start", cat["pen_start"] < -0.0005), ("joint speed past 2x its limit", cat["abnormal"]),
         ("object moved more than 5 cm", cat["obj_drift_final"] > 0.05), ("object below the table top", cat["obj_z_final"] < rc.TABLE_TOP),
         ("end effector moved more than 0.1 m", cat["ee_drift_final"] > 0.1), ("never stable for 0.5 s", cat["max_stable_run"] < 30)]
for name, m in rules[::-1]:
    reason[~ok & m] = name
assert (reason[~ok] != "passed").all(), "a failing state matched no rule"
names = ["passed"] + [r[0] for r in rules]
counts = {n: int((reason == n).sum()) for n in names}
print("tested", len(ok), counts)

keep = np.nonzero(ok)[0][:10000]
assert len(keep) == 10000
tested_until = int(keep[-1]) + 1
pi, gi = cat["pose"][keep].astype(np.int64), cat["grasp"][keep].astype(np.int64)
np.savez(f"{D}/object_reset_states.npz", obj_pos=P["pos"][pi], obj_quat_wxyz=P["quat"][pi], arm_q=C["q"][keep],
         hand_q=G["q_grasp"][gi], hand_target=G["q_squeeze"][gi], grasp=gi, tmpl=G["tmpl"][gi], pose=pi, candidate_row=keep,
         arm_joints=np.array(rc.ARM_JOINTS), hand_joints=np.array(rc.HAND_JOINTS),
         note=np.array("env-local frame of the simtoolreal-050 scene. hand_target is the squeeze, clamped to the joint limits at use."))
print(f"final set 10,000 from the first {tested_until} candidates, pass rate there {100 * 10000 / tested_until:.1f}%")

types = sorted(set(G["tmpl"].tolist()), key=lambda t: int(t.split("_")[0]))
tid = np.array([types.index(t) for t in G["tmpl"]])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 3.4))
ax.barh(names[::-1], [counts[n] for n in names[::-1]], color=["#999"] * (len(names) - 1) + ["#3a7d44"])
for i, n in enumerate(names[::-1]):
    ax.text(counts[n], i, f" {counts[n]:,}", va="center", fontsize=9)
ax.set_xlabel("candidates"); ax.set_title(f"S5. {counts['passed']:,} of {len(ok):,} candidates passed ({100 * ok.mean():.1f}%)")
save(fig, "s5_outcomes")
fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
axs[0].hist(np.clip(cat["obj_drift_final"] * 100, 0, 20), bins=80, color="#3060c0"); axs[0].axvline(5, color="k")
axs[0].set_xlabel("object drift at the end (cm), clipped at 20"); axs[0].set_yscale("log")
axs[1].hist(cat["ee_drift_final"] * 1000, bins=80, color="#3060c0"); axs[1].axvline(100, color="k")
axs[1].set_xlabel("end effector drift at the end (mm)"); axs[1].set_yscale("log")
pa = cat["pen_at_success"][ok] * 1000
axs[2].hist(np.clip(pa, -5, 5), bins=80, color="#7a3fa0"); axs[2].axvline(-0.5, color="k")
axs[2].set_xlabel("deepest robot point in the object at success (mm)")
axs[2].set_title(f"{100 * (pa >= -0.5).mean():.1f}% within OmniReset's 0.5 mm", fontsize=9)
fig.suptitle("S5. drift and settled depth, black lines at the limits")
save(fig, "s5_drift")
per = np.array([ok[tid[cat["grasp"]] == k].mean() if (tid[cat["grasp"]] == k).any() else np.nan for k in range(len(types))])
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.bar(range(len(types)), per * 100, color="#3a7d44")
ax.set_xticks(range(len(types)), types, rotation=60, ha="right", fontsize=8); ax.set_ylabel("passed (%)")
ax.set_title("S5. Isaac pass rate per grasp type")
save(fig, "s5_per_type")
z = P["pos"][cat["pose"].astype(np.int64), 2]
bins = np.linspace(0.6, 0.95, 15)
bi = np.digitize(z, bins)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot([(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))], [ok[bi == i].mean() * 100 for i in range(1, len(bins))], "o-", color="#3a7d44")
ax.set_xlabel("object centre height z (m)"); ax.set_ylabel("passed (%)"); ax.set_title("S5. Isaac pass rate against object height")
save(fig, "s5_vs_height")

# S6
fp = P["pos"][pi]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
axs[0].scatter(fp[:, 0], fp[:, 1], s=1, alpha=0.35, color="#3a7d44"); axs[0].plot(0, 0.48, "ks", ms=8); axs[0].set_aspect("equal")
axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)")
axs[1].scatter(fp[:, 0], fp[:, 2], s=1, alpha=0.35, color="#3a7d44"); axs[1].axhline(rc.TABLE_TOP, color="#b07a4f")
axs[1].set_xlabel("x (m)"); axs[1].set_ylabel("z (m)")
fig.suptitle("S6. object positions of the 10,000 passing states")
save(fig, "s6_positions")
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.bar(range(len(types)), np.bincount(tid[gi], minlength=len(types)), color="#3a7d44")
ax.set_xticks(range(len(types)), types, rotation=60, ha="right", fontsize=8); ax.set_ylabel("states")
ax.set_title("S6. grasp type mix of the 10,000 passing states")
save(fig, "s6_types")
lim = [[-2.9007, 2.9007], [-1.8361, 1.8361], [-2.9007, 2.9007], [-3.0770, -0.1169], [-2.8763, 2.8763], [0.4398, 4.6216], [-3.0508, 3.0508]]
fig, axs = plt.subplots(1, 7, figsize=(15, 2.8), sharey=True)
for k, ax in enumerate(axs):
    ax.hist(C["q"][keep][:, k], bins=50, color="#3a7d44")
    for x in lim[k]:
        ax.axvline(x, color="k", lw=1)
    ax.axvline(rc.ARM_DEFAULT[k], color="#c05030", lw=1, ls="--"); ax.set_title(f"joint {k + 1}", fontsize=9)
fig.suptitle("S6. arm joint spread of the 10,000 passing states, black the limits, red dashed the default pose")
save(fig, "s6_arm_spread")

# Pages for the pose viewer, 12 passing and 12 failing at random, and the S5 failure gallery.
rng = np.random.default_rng(0)
tested = np.arange(len(ok))
pick_pass = rng.choice(tested[ok], 12, replace=False)
pick_fail = rng.choice(tested[~ok], 12, replace=False)
np.savez(f"{D}/html_pick.npz", passing=pick_pass, failing=pick_fail, reason=reason.astype(str), files=np.array(files))
CAM = dict(azimuth=138.8, elevation=-22.0)
rows = {int(b["row"][0]): b for b in B}
starts = np.array(sorted(rows))


def frames(r):
    b = rows[int(starts[starts <= r].max())]
    j = int(r - b["row"][0])
    return b["rec_q"][:, j].astype(np.float64), b["rec_o"][:, j].astype(np.float64)


fails = []
for name in names[1:]:
    idx = tested[reason == name]
    if len(idx):
        fails.append((name, rng.choice(idx, min(3, len(idx)), replace=False)))
fig, axs = plt.subplots(len(fails), 6, figsize=(6 * 2.6, len(fails) * 2.9))
axs = np.atleast_2d(axs)
for ax in axs.ravel():
    ax.axis("off")
for i, (name, idx) in enumerate(fails):
    for k, r in enumerate(idx):
        q, o = frames(r)
        for t, (lab, step) in enumerate((("start", 0), ("end", -1))):
            m, d = rr.build([(q[step], 1.0)], (o[step, :3], o[step, 3:]))
            axs[i, 2 * k + t].imshow(rr.render(m, d, o[0, :3], 1.1, **CAM, size=480) if False else rr.render(m, d, o[0, :3], 1.1, **CAM))
            axs[i, 2 * k + t].set_title(f"{lab}, row {r}" + (f"\n{name}" if k == 0 and t == 0 else ""), fontsize=7.5)
fig.suptitle("S5. failing states by reason, start and end frame, up to three random each", fontsize=11)
fig.savefig(f"{OUT}/s5_fail_gallery.png", dpi=90, bbox_inches="tight")
plt.close(fig)
thumbs = []
for tag, idx in (("pass", pick_pass), ("fail", pick_fail)):
    for k, r in enumerate(idx):
        q, o = frames(r)
        m, d = rr.build([(q[0], 1.0)], (o[0, :3], o[0, 3:]))
        plt.imsave(f"{OUT}/thumb_{tag}_{k:02d}.png", rr.render(m, d, o[0, :3], 1.1, **CAM))
        thumbs.append((tag, k, int(r), reason[r], G["tmpl"][cat["grasp"][r]], P["pos"][cat["pose"][r]].round(3).tolist()))
np.save(f"{OUT}/thumbs.npy", np.array(thumbs, dtype=object), allow_pickle=True)
print("S5 S6 figures, gallery and thumbnails written")
