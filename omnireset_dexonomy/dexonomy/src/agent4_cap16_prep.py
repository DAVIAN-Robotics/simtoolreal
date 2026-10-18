"""Dexonomy object inputs for the cap16 pool, one per pool entry, from its URDF collision pieces.

Follows the m05 exact-hull path (agent3_exact_hulls.py, agent3_exact_meta.py) with no MeshProcess C++ step:
each URDF collision element becomes its own convex hull piece in the object root frame (origin and scale
applied, as PhysX hulls it), the MuJoCo model and URDF are MeshProcess's export_mjcf and export_urdf of those
pieces, the world mesh is the manifold union of the pieces (one outer surface), the info file is
MeshProcess's get_basic_info of that union, and the floating scene cfg holds the real size (scale 1) at the
identity pose. Scene cfg paths are absolute, so chunk dirs can link to the cfg file. The scene cfg also
carries mass_urdf_kg and mass_kg = clip(URDF mass, 0.05, 0.5), which the patched own test reads.

Run from anywhere with the dexonomy env. Usage: agent4_cap16_prep.py <manifest.json> <meshprocess_src> <out_root> <shard i> <n shards> <n proc>
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import manifold3d
import numpy as np
import trimesh

manifest_path, mp_src, out_root, shard, n_shard, n_proc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, mp_src)
from util.proc.basic import export_urdf, get_basic_info  # noqa: E402
from util.proc.tabletop_pose import export_mjcf  # noqa: E402


def origin_matrix(xyz, rpy):  # same convention as agent3_exact_hulls.origin_matrix
    T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    T[:3, 3] = xyz
    return T


def load_piece(c):  # same as agent3_exact_hulls.load_geom, from manifest fields
    parts = trimesh.load(c["mesh"], force="scene", process=False, split_object=True, group_material=False)
    parts = [g.copy() for g in parts.dump()] if isinstance(parts, trimesh.Scene) else [parts]
    T = origin_matrix(c["xyz"], c["rpy"]) @ np.diag(list(c["scale"]) + [1.0])
    for p in parts:
        p.apply_transform(T)
    return parts


def prep(e):
    t0 = time.time()
    stem = e["stem"]
    d = f"{out_root}/processed_data/{stem}"
    os.makedirs(f"{d}/urdf/meshes", exist_ok=True)
    os.makedirs(f"{d}/mesh", exist_ok=True)
    os.makedirs(f"{d}/info", exist_ok=True)
    hulls, piece_vol = [], 0.0
    for i, c in enumerate(e["collision"]):
        pts = np.concatenate([p.vertices for p in load_piece(c)]).astype(np.float32)
        h = manifold3d.Manifold.hull_points(pts)
        assert h.status() == manifold3d.Error.NoError and h.volume() > 0, (stem, i)
        m = h.to_mesh()
        trimesh.Trimesh(m.vert_properties[:, :3], m.tri_verts, process=False).export(f"{d}/urdf/meshes/convex_piece_{i:03d}.obj")
        hulls.append(h)
        piece_vol += h.volume()
    union = manifold3d.Manifold.batch_boolean(hulls, manifold3d.OpType.Add)
    um = union.to_mesh()
    union_mesh = trimesh.Trimesh(um.vert_properties[:, :3], um.tri_verts, process=False)
    assert union_mesh.is_watertight, stem
    assert 0 < union.volume() <= piece_vol * (1 + 1e-6), (stem, union.volume(), piece_vol)
    union_mesh.export(f"{d}/mesh/simplified.obj")
    export_urdf({"input_path": f"{d}/urdf/meshes", "output_path": f"{d}/urdf/coacd.urdf"}, skip=False, debug=False)
    export_mjcf({"input_path": f"{d}/urdf/meshes", "output_path": f"{d}/urdf/coacd.xml"}, skip=False, debug=False)
    get_basic_info({"input_path": f"{d}/mesh/simplified.obj", "output_path": f"{d}/info/simplified.json"}, skip=False, debug=False)
    save_path = f"{out_root}/scene_cfg/{stem}/floating/real"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mass = float(e["mass_kg"])
    scene_cfg = {
        "scene": {stem: {"type": "rigid_object",
                         "file_path": f"{d}/mesh/simplified.obj", "xml_path": f"{d}/urdf/coacd.xml",
                         "urdf_path": f"{d}/urdf/coacd.urdf", "info_path": f"{d}/info/simplified.json",
                         "scale": np.array([1.0, 1.0, 1.0]), "pose": np.array([0.0, 0, 0, 1, 0, 0, 0]),
                         "mass_urdf_kg": mass, "mass_kg": float(np.clip(mass, 0.05, 0.5))}},
        "scene_id": f"{stem}/floating/real",
        "task": {"type": "force_closure", "obj_name": stem},
    }
    np.save(save_path, scene_cfg)
    return {"stem": stem, "pieces": len(hulls), "union_cm3": round(union.volume() * 1e6, 4), "pieces_cm3": round(piece_vol * 1e6, 4),
            "overlap_frac": round(1 - union.volume() / piece_vol, 5), "mass_urdf_kg": mass, "mass_kg": float(np.clip(mass, 0.05, 0.5)),
            "extent_m": np.round(union_mesh.extents, 4).tolist(), "wall_s": round(time.time() - t0, 3)}


if __name__ == "__main__":
    entries = json.load(open(manifest_path))
    mine = entries[shard::n_shard]
    t0 = time.time()
    with Pool(n_proc) as pool:
        rows = pool.map(prep, mine, chunksize=1)
    os.makedirs(f"{out_root}/prep_logs", exist_ok=True)
    json.dump({"shard": shard, "n_shard": n_shard, "n": len(rows), "wall_s": round(time.time() - t0, 1), "rows": rows},
              open(f"{out_root}/prep_logs/shard_{shard:02d}.json", "w"))
    print(f"shard {shard}: {len(rows)} entries in {time.time() - t0:.1f}s")
