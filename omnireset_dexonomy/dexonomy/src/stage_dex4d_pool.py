"""Stage the Dex4D train set as SimToolReal pool assets.

One URDF per object-scale pair, in the object's own oriented-box frame, with the
link named object_root, the collision built from the CoACD parts that already
ship, and an explicit inertial computed from geometry and a chosen density.

The box frame comes from the 036 tree's own ``dextoolbench.oriented_box``, not
from a second implementation, so the pool and the reward agree by construction.

Mesh files are referenced in place. The UniDexGrasp assets are read-only and
nothing here writes into them.
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

from dextoolbench.oriented_box import oriented_box, reexpress_origin

OBJECT_ROOT_LINK = "object_root"
OBJECT_BASE_SIZE = 0.04


def pairs_from_split(path):
    text = Path(path).read_text()
    out = []
    for code, scales in re.findall(r"'([^']+)'\s*:\s*\[([^\]]*)\]", text):
        for s in [x.strip() for x in scales.split(",") if x.strip()]:
            out.append((code, float(s)))
    return out


def inertial(pieces, density):
    """Mass, centre of mass and the full inertia tensor about that centre.

    Each CoACD part is watertight and convex, so each carries its own mass
    properties and the parallel axis theorem assembles them. Summing a union of
    parts is what the geometry is, and it is what DexToolBench's decomposed urdf
    authors.
    """
    m = np.array([abs(p.volume) * density for p in pieces])
    c = np.array([p.center_mass for p in pieces])
    total = m.sum()
    com = (m[:, None] * c).sum(0) / total
    I = np.zeros((3, 3))
    for mi, ci, p in zip(m, c, pieces):
        Ii = p.moment_inertia * density  # trimesh reports for unit density
        d = ci - com
        I += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return float(total), com, I


def cap_parts(pieces, max_parts):
    """Keep the largest parts, collapse the rest into one hull.

    PhysX holds every convex shape for every env, so the tail of a 121-part
    decomposition costs far more than it buys. Keeping the big parts preserves
    the shape features a hand grips and folds the fine detail into a single
    enclosing hull.
    """
    if max_parts <= 0 or len(pieces) <= max_parts:
        return pieces
    order = sorted(range(len(pieces)), key=lambda i: abs(pieces[i].volume), reverse=True)
    keep = [pieces[i] for i in order[:max_parts]]
    rest = [pieces[i] for i in order[max_parts:]]
    return keep + [trimesh.util.concatenate(rest).convex_hull]


def stage_one(code, scale, assets, out_dir, density, max_parts=0, variant=None):
    """variant names a sibling of the object's coacd dir, e.g.
    coacd_max_convex_hull_8, holding a re-decomposition of the same mesh. Its
    pieces carry the upstream file names and the same unscaled frame, so the
    scale factor applies unchanged."""
    src = Path(assets) / code / (variant or "coacd")
    tag = f"{int(round(scale * 100)):03d}"
    piece_paths = sorted(src.glob("coacd_convex_piece_*.obj"),
                         key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not piece_paths:
        raise FileNotFoundError(f"no CoACD parts for {code}")
    visual = Path(assets) / code / "coacd" / f"decomposed_{tag}.obj"
    if not visual.exists():
        raise FileNotFoundError(visual)

    pieces = []
    for p in piece_paths:
        m = trimesh.load(str(p), process=False)
        m.apply_scale(scale)
        pieces.append(m)
    pieces = cap_parts(pieces, max_parts)
    merged = trimesh.util.concatenate(pieces)

    extents, box_in_mesh = oriented_box(merged)
    mesh_in_box = np.linalg.inv(box_in_mesh)
    xyz, rpy = reexpress_origin(mesh_in_box, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    origin = f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>'

    mass, com, I = inertial(pieces, density)
    icom = (mesh_in_box @ np.append(com, 1.0))[:3]
    R = mesh_in_box[:3, :3]
    Ib = R @ I @ R.T

    name = code.replace("/", "__") + f"_s{tag}"
    bodies = [f"    <visual>\n      {origin}\n"
              f'      <geometry><mesh filename="{visual}" scale="1 1 1"/></geometry>\n'
              f'      <material name="grey"><color rgba="0.75 0.75 0.75 1.0"/></material>\n'
              f"    </visual>"]
    if max_parts and len(piece_paths) > max_parts:
        # the parts changed, so they have to be written out rather than cited
        part_dir = Path(out_dir) / "parts" / name
        part_dir.mkdir(parents=True, exist_ok=True)
        refs = []
        for i, m in enumerate(pieces):
            f = part_dir / f"part_{i:03d}.obj"
            m.export(f)
            refs.append((f, 'scale="1 1 1"'))
    else:
        refs = [(p, f'scale="{scale} {scale} {scale}"') for p in piece_paths]
    for f, sc in refs:
        bodies.append(f"    <collision>\n      {origin}\n"
                      f'      <geometry><mesh filename="{f}" {sc}/></geometry>\n'
                      f"    </collision>")
    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{OBJECT_ROOT_LINK}">
{chr(10).join(bodies)}
    <inertial>
      <origin xyz="{icom[0]} {icom[1]} {icom[2]}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{Ib[0,0]}" ixy="{Ib[0,1]}" ixz="{Ib[0,2]}"
               iyy="{Ib[1,1]}" iyz="{Ib[1,2]}" izz="{Ib[2,2]}"/>
    </inertial>
  </link>
</robot>
"""
    path = Path(out_dir) / f"{name}.urdf"
    path.write_text(urdf)
    return dict(urdf=str(path), key=f"{code}@{tag}",
                scale=[float(e / OBJECT_BASE_SIZE) for e in extents],
                extent_m=[float(e) for e in extents],
                mass_kg=mass, parts=len(pieces))


def _stage_star(x):
    return stage_one(*x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--density", type=float, required=True)
    ap.add_argument("--max-parts", type=int, default=0,
                    help="0 keeps every shipped CoACD part")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--variant", default=None,
                    help="sibling of coacd/ to take collision from, e.g. coacd_max_convex_hull_8")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    todo = pairs_from_split(a.split)
    args = [(c, s, a.assets, a.out_dir, a.density, a.max_parts, a.variant) for c, s in todo]
    rows = []
    if a.workers <= 1:
        for i, x in enumerate(args):
            rows.append(stage_one(*x))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(_stage_star, args, chunksize=8)):
                rows.append(r)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)

    names = [Path(r["urdf"]).stem for r in rows]
    assert len(set(names)) == len(names), "staged urdf names are not unique"
    (a.out_dir / "pool.json").write_text(json.dumps(rows, indent=1))

    m = np.array([r["mass_kg"] for r in rows])
    s = np.array([r["scale"] for r in rows])
    print(f"\nstaged {len(rows)} urdfs, {len(set(names))} unique names")
    print(f"mass kg   p05 {np.percentile(m,5):.4f}  median {np.median(m):.4f}  "
          f"p95 {np.percentile(m,95):.4f}  max {m.max():.4f}")
    print(f"normalized scale  min {s.min():.3f}  max {s.max():.3f}  "
          f"(policy obs clamp is 10)")
    pc = np.array([r["parts"] for r in rows])
    print(f"convex parts  total {pc.sum()}  mean {pc.mean():.1f}  max {pc.max()}")


if __name__ == "__main__":
    main()
