"""Direct comparison of two per-object shape codes on the cap16 collision pieces, BPS and the hull ray code (OMD).

Both codes are 1024 points x 3 per object in one frame, centred on the bounding box and scaled into the unit ball, so
object_scales keeps carrying size. BPS stores, for 1024 fixed basis points, the vector to the nearest surface point.
OMD merges the pieces into one convex hull and stores where 1024 Fibonacci rays from the hull's centre of mass leave it.
Each code gets its own PCA at k = 8 to 128. The rebuilt points of both are scored against one truth, 100,000 points on
the surface of the union of the pieces, in mm at true size, in both directions. Piece faces hidden inside another piece
are not surface, so their samples are dropped from the truth and from the BPS input alike.
Usage: shape_code_compare.py <object pool dir> <out dir> <workers>
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import reset_common as rc

KS = [8, 16, 32, 64, 128]
N_POINTS, N_BUILD, N_TRUTH = 1024, 20000, 100000
HIDDEN_EPS = 1e-4  # m, cut faces of neighbouring pieces sit up to 0.1 mm apart, the hidden share is flat from 1e-5 on
rng = np.random.default_rng(0)
v = rng.normal(size=(N_POINTS, 3))
BASIS = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(size=(N_POINTS, 1)) ** (1 / 3)
i = np.arange(N_POINTS)
z = 1 - (2 * i + 1) / N_POINTS
RAYS = np.stack([np.sqrt(1 - z ** 2) * np.cos(np.pi * (3 - np.sqrt(5)) * i),
                 np.sqrt(1 - z ** 2) * np.sin(np.pi * (3 - np.sqrt(5)) * i), z], 1)
METHODS = ["BPS", "OMD"]
COLORS = {"BPS": "#2b6cb0", "OMD": "#c05621"}


def planes(mesh):
    n = mesh.face_normals
    return n, (n * mesh.vertices[mesh.faces[:, 0]]).sum(1)


def surface(pieces, n, seed):
    pts, face = trimesh.sample.sample_surface(trimesh.util.concatenate(pieces), n, seed=seed)
    owner = np.searchsorted(np.cumsum([len(p.faces) for p in pieces]), face, side="right")
    hidden = np.zeros(len(pts), bool)
    for j, p in enumerate(pieces):
        nj, dj = planes(p)
        for s in range(0, len(pts), 2000):  # pieces have up to 12,118 faces
            hidden[s:s + 2000] |= ((pts[s:s + 2000] @ nj.T - dj).max(1) < HIDDEN_EPS) & (owner[s:s + 2000] != j)
    return pts[~hidden], hidden.mean()


def ray_hits(hull):
    n, d = planes(hull)
    o = hull.center_mass
    gap = d - n @ o
    assert (gap > 0).all(), gap.min()
    nu = RAYS @ n.T
    with np.errstate(divide="ignore"):
        t = np.where(nu > 0, gap / nu, np.inf).min(1)
    assert np.isfinite(t).all()
    return o + t[:, None] * RAYS


def describe(urdf):
    pieces = rc.object_pieces(urdf)
    pts, hidden = surface(pieces, N_BUILD, seed=0)
    center = (pts.min(0) + pts.max(0)) / 2
    radius = np.linalg.norm(pts - center, axis=1).max()
    pts_n = (pts - center) / radius
    _, idx = cKDTree(pts_n).query(BASIS)
    hull = trimesh.convex.convex_hull(np.vstack([p.vertices for p in pieces]))
    omd = (ray_hits(hull) - center) / radius
    return ((pts_n[idx] - BASIS).ravel().astype(np.float32), omd.ravel().astype(np.float32), center, radius,
            len(pieces), hidden)


def rebuild(m, j, k):
    X, mean, Vt = CODES[m]
    rec = X[j] if k is None else ((X[j] - mean) @ Vt[:k].T) @ Vt[:k] + mean
    return rec.reshape(N_POINTS, 3) + (BASIS if m == "BPS" else 0)


def truth(j):
    pts, _ = surface(rc.object_pieces(urdfs[j]), N_TRUTH, seed=1)
    return (pts - center[j]) / radius[j]


def score(j):
    t = truth(j)
    tree = cKDTree(t)
    out = np.zeros((len(METHODS), len(KS) + 1, 2))
    for a, m in enumerate(METHODS):
        for b, k in enumerate(KS + [None]):
            p = rebuild(m, j, k)
            out[a, b] = tree.query(p)[0].mean(), cKDTree(p).query(t)[0].mean()
    return out * radius[j] * 1000


pool_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
workers = int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
urdfs = sorted(pool_dir.glob("*.urdf"))
names = [u.stem for u in urdfs]
t0 = time.time()
with Pool(workers) as p:
    res = p.map(describe, urdfs, chunksize=8)
t_describe = time.time() - t0
bps = np.stack([r[0] for r in res])
omd = np.stack([r[1] for r in res])
center = np.stack([r[2] for r in res])
radius = np.array([r[3] for r in res])
n_pieces = np.array([r[4] for r in res])
hidden = np.array([r[5] for r in res])
np.savez(out / "shape_codes.npz", bps=bps, omd=omd, center=center, radius=radius, n_pieces=n_pieces,
         names=np.array(names), basis=BASIS, rays=RAYS)

CODES, var = {}, {}
for m, X in (("BPS", bps), ("OMD", omd)):
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    CODES[m] = (X, mean, Vt[:max(KS)])
    var[m] = S ** 2 / (S ** 2).sum()

t0 = time.time()
with Pool(workers) as p:
    err = np.stack(p.map(score, range(len(urdfs)), chunksize=8))  # (objects, method, k and full, [rebuilt->truth, truth->rebuilt])
t_score = time.time() - t0
np.save(out / "shape_code_compare_err_mm.npy", err)

first = {}
for j, nm in enumerate(names):
    first.setdefault(nm.rsplit("_s", 1)[0], j)
u = np.array(sorted(first.values()))


def nearest_other(Z):
    q = cKDTree(Z).query(Z, k=2)[1]
    return np.where(q[:, 0] == np.arange(len(Z)), q[:, 1], q[:, 0])


agree = {}
for m in METHODS:
    X, mean, Vt = CODES[m]
    full = nearest_other(X[u])
    agree[m] = [(nearest_other((X[u] - mean) @ Vt[:k].T) == full).mean() for k in KS]

one, many = n_pieces == 1, n_pieces > 1
groups = (("all", np.ones(len(names), bool)), ("1 piece", one), ("2-16 pieces", many))
lines = [f"{len(names)} objects ({one.sum()} with 1 piece, {many.sum()} with 2-16), {len(u)} unique meshes, "
         f"median radius {np.median(radius) * 1000:.1f} mm",
         f"hidden piece-face samples dropped, median over 2-16 piece objects {np.median(hidden[many]) * 100:.1f}%, "
         f"max {hidden.max() * 100:.1f}%",
         f"time: describe {t_describe:.0f} s, score {t_score:.0f} s, {workers} workers on the kait login node",
         "",
         "Error in mm at true size, median over objects (p95 in brackets).",
         "rebuilt->truth = mean distance from each rebuilt point to the true surface.",
         "truth->rebuilt = mean distance from each true surface point to the nearest rebuilt point.",
         "full = the 3072-number code with no PCA."]
for gname, g in groups:
    lines += ["", f"[{gname}, {g.sum()} objects]",
              "k     " + "  ".join(f"{m} rebuilt->truth   {m} truth->rebuilt" for m in METHODS)]
    for b, k in enumerate(KS + [None]):
        row = f"{'full' if k is None else k:<5} "
        for a, m in enumerate(METHODS):
            for c in range(2):
                e = err[g, a, b, c]
                row += f"  {np.median(e):6.2f} ({np.percentile(e, 95):6.2f})    "
        lines.append(row)
lines += ["", "Within each method, not a head-to-head. Explained variance is a share of that code's own total.",
          "Same nearest other mesh is k-code against that method's full code, over unique meshes.",
          "k     " + "  ".join(f"{m} var   {m} same_nn" for m in METHODS)]
for b, k in enumerate(KS):
    lines.append(f"{k:<5} " + "  ".join(f"{var[m][:k].sum():.4f}   {agree[m][b]:.4f}    " for m in METHODS))
(out / "shape_code_compare.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)

fig, ax = plt.subplots(len(groups), 2, figsize=(12, 4 * len(groups)))
for r, (gname, g) in enumerate(groups):
    for c, what in enumerate(["rebuilt points -> true surface", "true surface -> rebuilt points"]):
        a_ = ax[r, c]
        for a, m in enumerate(METHODS):
            a_.plot(KS, [np.median(err[g, a, b, c]) for b in range(len(KS))], "o-", color=COLORS[m],
                    label=f"{m} median")
            a_.plot(KS, [np.percentile(err[g, a, b, c], 95) for b in range(len(KS))], "o--", color=COLORS[m],
                    alpha=0.6, label=f"{m} 95th percentile")
            a_.axhline(np.median(err[g, a, -1, c]), color=COLORS[m], ls=":", lw=1.5, label=f"{m} full code, median")
        a_.set_xscale("log", base=2)
        a_.set_xticks(KS)
        a_.set_xticklabels(KS)
        a_.set_xlabel("components k")
        a_.set_ylabel("error (mm)")
        a_.set_title(f"{what}, {gname} ({g.sum()} objects)", fontsize=11)
        a_.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "shape_code_compare_curves.png", dpi=120)


def category(name):
    return name.split("-")[0]


rng = np.random.default_rng(3)
by_cat = {}
for j in rng.permutation(len(names)):
    by_cat.setdefault(category(names[j]), j)
pick = [by_cat[c] for c in rng.choice(sorted(by_cat), 12, replace=False)]
cols = ["true surface"] + [f"k = {k}" for k in KS] + ["full code"]
for page in range(4):
    objs = pick[page * 3:(page + 1) * 3]
    fig = plt.figure(figsize=(3 * len(cols), 3 * 2 * len(objs)))
    for r, j in enumerate(objs):
        t = truth(j)[::40]
        for a, m in enumerate(METHODS):
            row = 2 * r + a
            clouds = [t] + [rebuild(m, j, k) for k in KS] + [rebuild(m, j, None)]
            for c, p in enumerate(clouds):
                a_ = fig.add_subplot(2 * len(objs), len(cols), row * len(cols) + c + 1, projection="3d")
                a_.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.5, c=p[:, 2], cmap="viridis")
                a_.set_xlim(-1, 1)
                a_.set_ylim(-1, 1)
                a_.set_zlim(-1, 1)
                a_.set_box_aspect((1, 1, 1))
                a_.view_init(elev=20, azim=35)
                a_.set_axis_off()
                a_.set_title(cols[c] if row == 0 else "", fontsize=11)
            label = category(names[j]).split("__")[-1][:12]
            fig.text(0.002, 1 - (row + 0.5) / (2 * len(objs)), f"{m}  {label}  {n_pieces[j]} pc", rotation=90,
                     va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"shape_code_compare_recon_{page + 1}.png", dpi=100)
    print("saved page", page + 1, [(names[j], int(n_pieces[j])) for j in objs], flush=True)
