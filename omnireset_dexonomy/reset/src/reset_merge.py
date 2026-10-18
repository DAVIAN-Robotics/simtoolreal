"""Pool the IK shards and their object verdicts, write the figure statistics and the Isaac candidates.

Candidates are drawn uniformly at random over every IK solution, one row per successful seed as cuRobo
returned it. The arm-object check from reset_arm_object.py is reported but not applied, at Sibeen's call. It
dropped 0.22 percent of solutions, and the Isaac start-state check tests every robot body against the
object anyway. The Isaac test walks the candidates in this order.
Usage: reset_merge.py <reset dir> <n candidates>
"""
import sys

import numpy as np

import reset_common as rc

D, n_cand = sys.argv[1], int(sys.argv[2])
P = np.load(f"{D}/poses.npz")
g = np.load(rc.GRASPS)
types = sorted(set(g["tmpl"].tolist()), key=lambda t: int(t.split("_")[0]))
tid = np.array([types.index(t) for t in g["tmpl"]])
n_pose, n_g = len(P["pos"]), len(g["tmpl"])

pose, grasp, q, perr, rerr, object, stats = [], [], [], [], [], [], []
for i in range(8):
    s = np.load(f"{D}/ik/shard{i}.npz")
    pose.append(s["pose"]), grasp.append(s["grasp"]), q.append(s["q"]), perr.append(s["pos_err"]), rerr.append(s["rot_err"])
    object.append(np.load(f"{D}/ik/shard{i}_obj.npy")), stats.append(s["stats"])
pose, grasp, q = np.concatenate(pose).astype(np.int64), np.concatenate(grasp).astype(np.int64), np.concatenate(q)
perr, rerr, object, stats = np.concatenate(perr), np.concatenate(rerr), np.concatenate(object), np.concatenate(stats)
assert len(stats) == n_pose and (np.sort(stats[:, 0]) == np.arange(n_pose)).all()
stats = stats[np.argsort(stats[:, 0])]

hand_ok = P["hand_clear"] >= 0  # (pose, grasp)
key = pose * n_g + grasp
solved = np.zeros(n_pose * n_g, bool)
solved[key] = True
solved = solved.reshape(n_pose, n_g)
assert not (solved & ~hand_ok).any()
obj_key = key[object]
kept = np.zeros(n_pose * n_g, bool)
kept[obj_key] = True
kept = kept.reshape(n_pose, n_g)
per_wrist = np.bincount(key, minlength=n_pose * n_g).reshape(n_pose, n_g)

rng = np.random.default_rng(0)
elig = np.arange(len(q))
pick = elig[rng.permutation(len(elig))[:n_cand]]
np.savez(f"{D}/candidates.npz", pose=pose[pick], grasp=grasp[pick], q=q[pick], row=pick, obj_ok=object[pick])

sub = rng.choice(len(q), min(len(q), 2_000_000), replace=False)
np.savez(f"{D}/stats.npz", types=np.array(types), tid=tid, pose_pos=P["pos"], pose_quat=P["quat"],
         hand_ok=hand_ok, solved=solved, kept=kept, per_wrist_hist=np.bincount(per_wrist[hand_ok], minlength=33),
         q_sample=q[sub], perr_sample=perr[sub].astype(np.float32), rerr_sample=rerr[sub].astype(np.float32),
         n_drawn=P["n_drawn"], n_rejected=P["n_rejected"], rejected_z=P["rejected_z"], hand_clear_min=P["hand_clear"].min(1),
         stats=stats, n_solutions=len(q), n_obj_ok=int(object.sum()), obj_drop_pose=np.bincount(pose[~object], minlength=n_pose))
print(f"pairs {n_pose * n_g} hand ok {hand_ok.sum()} wrists solved {solved.sum()} solutions {len(q)} "
      f"object ok {object.sum()} wrists kept {kept.sum()} candidates {len(pick)}")
print("IK wall per pose median s", np.median(stats[:, 4]))
