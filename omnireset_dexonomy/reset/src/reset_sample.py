"""S1 and S2. Sample object poses the way 050's absolute goal sampler does, then run the two table checks.

S1. Position uniform in the goal box, orientation uniform over SO(3), as sample_absolute_goal_pose.
The 050 goal sampler has no table clearance, so a pose is rejected and redrawn when the object's lowest
collision point sits below the table top. That rejection is our addition, since the object is a real body here.
S2. A pose and grasp pair drops when the hand's lowest collision point, at the grasp angles, sits below
the table top. Both checks read convex hull vertices, which is exact against a flat table.
Writes poses.npz with the kept poses, the rejection count, and the per pair hand clearance.
"""
import argparse
import time

import numpy as np
import torch
import trimesh

import reset_common as rc

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rc.check_tree()
dev = torch.device("cuda")
t0 = time.time()

obj_v = torch.as_tensor(trimesh.util.concatenate(rc.object_pieces()).convex_hull.vertices, device=dev)
gen = torch.Generator(device=dev).manual_seed(a.seed)
lo, hi = torch.tensor(rc.BOX_MIN, device=dev, dtype=torch.float64), torch.tensor(rc.BOX_MAX, device=dev, dtype=torch.float64)
kept_p, kept_q, n_drawn, rej_z = [], [], 0, []
while sum(len(p) for p in kept_p) < a.n:
    m = 4096
    pos = lo + (hi - lo) * torch.rand(m, 3, device=dev, generator=gen, dtype=torch.float64)
    quat = torch.nn.functional.normalize(torch.randn(m, 4, device=dev, generator=gen, dtype=torch.float64), dim=-1)
    low = pos[:, 2] + (rc.quat_to_mat(quat)[:, 2, :] @ obj_v.T).min(-1).values
    ok = low >= rc.TABLE_TOP
    n_drawn += m
    rej_z.append(pos[~ok, 2].cpu())
    kept_p.append(pos[ok]), kept_q.append(quat[ok])
pos, quat = torch.cat(kept_p)[: a.n], torch.cat(kept_q)[: a.n]
n_used = n_drawn - (sum(len(p) for p in kept_p) - a.n)  # draws up to the n-th kept pose
print(f"S1 kept {a.n} poses, drew about {n_drawn}, rejected {n_drawn - sum(len(p) for p in kept_p)} for table clearance", flush=True)

# S2, hand hull in the object frame per grasp, at q_grasp.
u, g = rc.Urdf(), rc.load_grasps(dev)
hand_links = [l for l in u.subtree("palm") if u.links[l].find("collision") is not None]
q = {j: g["q_grasp"][:, i].double() for i, j in enumerate(rc.HAND_JOINTS)}
T = u.fk_from("palm", q)
T_obj_palm = rc.homog(rc.quat_to_mat(g["hand_quat"].double()), g["hand_pos"].double())
pts = []
for l in hand_links:
    for v in u.collision_hulls(l):
        v = torch.as_tensor(v, device=dev)
        pts.append((T_obj_palm @ T[l])[:, None, :3, :3].matmul(v[None, :, :, None]).squeeze(-1) + (T_obj_palm @ T[l])[:, None, :3, 3])
pts = torch.cat(pts, 1)  # (G, V, 3) hand hull vertices in the object frame
hulls = [trimesh.convex.convex_hull(p) for p in pts.cpu().numpy()]
H = max(len(h.vertices) for h in hulls)
hv = torch.zeros(len(hulls), H, 3, dtype=torch.float64, device=dev)
for i, h in enumerate(hulls):
    hv[i, : len(h.vertices)] = torch.as_tensor(h.vertices, device=dev)
    hv[i, len(h.vertices):] = torch.as_tensor(h.vertices[0], device=dev)
dz = rc.quat_to_mat(quat)[:, 2, :]  # world z in the object frame
low = torch.empty(a.n, len(hulls), dtype=torch.float32)
for s in range(0, a.n, 256):
    low[s:s + 256] = (pos[s:s + 256, 2, None] + torch.einsum("pk,gvk->pgv", dz[s:s + 256], hv).min(-1).values).float().cpu()
clear = low - rc.TABLE_TOP
print(f"S2 hand links {len(hand_links)}, hull vertices per grasp up to {H}; pairs kept {(clear >= 0).sum().item()} of {clear.numel()}", flush=True)
np.savez(a.out, pos=pos.cpu().numpy(), quat=quat.cpu().numpy(), n_drawn=n_drawn, n_rejected=n_drawn - sum(len(p) for p in kept_p),
         rejected_z=torch.cat(rej_z).numpy(), hand_clear=clear.numpy(), table_top=rc.TABLE_TOP)
print(f"wrote {a.out} in {time.time() - t0:.1f} s")
