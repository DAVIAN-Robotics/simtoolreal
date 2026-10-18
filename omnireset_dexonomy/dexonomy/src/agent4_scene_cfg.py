"""Write one Dexonomy floating scene cfg per object, at its real size and in its URDF root frame.

The dict layout is MeshProcess export_floating_scene_cfg (src/util/proc/basic.py). The one change is
scale and pose. MeshProcess mesh_normalize maps v to (v - center) / length with center the bbox
centre and length half the bbox diagonal, and its floating cfgs then pick a random scale at the
origin. Our objects have one real size and a fixed frame, so scale = length and pose = (center,
identity), which puts the processed mesh back where the raw mesh was. World equals object_root.

Also checks, per object, that normalized.obj maps back onto the raw vertices, and how far
simplified.obj (the mesh Dexonomy samples and queries) sits from the raw collision solid.

Usage: python agent4_scene_cfg.py <objects_dir>   (holds raw_mesh/<obj>.obj and processed_data/<obj>/)
"""
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

root = sys.argv[1]
rows = {}
for fname in sorted(os.listdir(os.path.join(root, "raw_mesh"))):
    obj = fname.removesuffix(".obj")
    proc = os.path.join(root, "processed_data", obj)
    raw = trimesh.load(os.path.join(root, "raw_mesh", fname), force="mesh")
    v = np.asarray(raw.vertices)
    center = (v.max(0) + v.min(0)) / 2
    length = float(np.linalg.norm(v.max(0) - v.min(0)) / 2)

    norm = trimesh.load(os.path.join(proc, "mesh", "normalized.obj"), force="mesh")
    back = np.asarray(norm.vertices) * length + center
    norm_err = max(cKDTree(v).query(back)[0].max(), cKDTree(back).query(v)[0].max())

    raw_solid = trimesh.load(os.path.join(root, "raw_mesh", fname), force="mesh", process=False)
    simp = trimesh.load(os.path.join(proc, "mesh", "simplified.obj"), force="mesh")
    simp.vertices = np.asarray(simp.vertices) * length + center
    s_pts = trimesh.sample.sample_surface(simp, 20000, seed=0)[0]
    r_pts, r_face = trimesh.sample.sample_surface(raw_solid, 20000, seed=0)
    d_s2r = trimesh.proximity.closest_point(raw_solid, s_pts)[1]
    d_r2s = trimesh.proximity.closest_point(simp, r_pts)[1]
    # The raw union keeps touching shells separate, so it has internal faces where they meet.
    # A raw sample on another shell's surface, or inside another shell, is internal.
    label = trimesh.graph.connected_component_labels(raw_solid.face_adjacency, node_count=len(raw_solid.faces))
    shell_of_pt = label[r_face]
    internal = np.zeros(len(r_pts), bool)
    for k in np.unique(label):
        other = trimesh.Trimesh(raw_solid.vertices, raw_solid.faces[label != k], process=False)
        mine = shell_of_pt == k
        if len(other.faces) == 0 or not mine.any():
            continue
        near = trimesh.proximity.closest_point(other, r_pts[mine])[1] < 2e-4
        internal[np.where(mine)[0][near]] = True
    d_r2s_outer = d_r2s[~internal]

    scene_dir = os.path.join(root, "scene_cfg", obj, "floating")
    save_path = os.path.join(scene_dir, "real")
    paths = {
        "file_path": os.path.join(proc, "mesh", "simplified.obj"),
        "xml_path": os.path.join(proc, "urdf", "coacd.xml"),
        "urdf_path": os.path.join(proc, "urdf", "coacd.urdf"),
        "info_path": os.path.join(proc, "info", "simplified.json"),
    }
    for p in paths.values():
        assert os.path.exists(p), p
    scene_cfg = {
        "scene": {
            obj: {
                "type": "rigid_object",
                **{k: os.path.relpath(p, os.path.dirname(save_path)) for k, p in paths.items()},
                "scale": np.array([length, length, length]),
                "pose": np.array([center[0], center[1], center[2], 1.0, 0, 0, 0]),
            }
        },
        "scene_id": f"{obj}/floating/real",
        "task": {"type": "force_closure", "obj_name": obj},
    }
    os.makedirs(scene_dir, exist_ok=True)
    np.save(save_path, scene_cfg)

    info = json.load(open(paths["info_path"]))
    rows[obj] = {
        "length_m": round(length, 5),
        "center_m": np.round(center, 5).tolist(),
        "normalized_back_to_raw_max_m": float(f"{norm_err:.2e}"),
        "simplified_to_raw_mm": {"mean": round(float(d_s2r.mean()) * 1000, 3), "max": round(float(d_s2r.max()) * 1000, 3)},
        "raw_to_simplified_mm": {"mean": round(float(d_r2s.mean()) * 1000, 3), "max": round(float(d_r2s.max()) * 1000, 3)},
        "raw_shells": int(len(np.unique(label))),
        "raw_internal_sample_frac": round(float(internal.mean()), 4),
        "raw_outer_to_simplified_mm": {"mean": round(float(d_r2s_outer.mean()) * 1000, 3), "max": round(float(d_r2s_outer.max()) * 1000, 3)},
        "coacd_pieces": len(os.listdir(os.path.join(proc, "urdf", "meshes"))),
        "simplified_vertices": len(simp.vertices),
        "info_simplified_json": info,
    }
    print(obj, json.dumps(rows[obj]))
json.dump(rows, open(os.path.join(root, "scene_cfg", "agent4_scene_cfg_report.json"), "w"), indent=1)
