"""Taxel model from Closing the Reality Gap, arXiv 2601.02778v2, sections at paper.tex:471-550.

Each object carries a surface point cloud with normals. Each vendor taxel rides its fingertip link
through forward kinematics, finds its nearest cloud point, and counts as touching when the offset
opposes that point's outward normal. Per finger the active taxels reduce to a summed depth and a
depth-weighted contact centre, which is 20 numbers for the hand.

    contact_ij  <=>  (p_ij - p_o) . n_o < 0
    d_ij        =    |p_ij - p_o|
    F_i         =    sum_j d_ij
    mu_i        =    sum_j d_ij p_ij / sum_j d_ij      the stiffness cancels, paper.tex:547-550

The cloud is sampled from the convex hulls the solver uses, so the taxel senses the surface the
physics pushes on. Stiffness is 1, because the paper never publishes one and a constant is absorbed
by the observation normaliser.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

NUM_TAXELS_PER_FINGER: int = 120
NUM_CLOUD_POINTS: int = 1024
# A sampled-cloud sign test is only reliable near the surface. Far from the object the nearest
# sample's normal can point away from the taxel and report a false contact, which a rollout showed
# as depths of several hundred millimetres. The paper states no such bound. Taxel pitch is 2.2 mm
# and the sampling error is a few millimetres, so 5 mm admits every real contact.
MAX_CONTACT_DEPTH: float = 0.005

# points_t16_transformed and points_t30_right_hand_transformed, already in the finger link2
# frames that fingertip_body_regex selects. Millimetres, 10 rows by 12 columns.
_TAXEL_DIR = "assets/tactile"
_T16 = "points_t16_transformed.json"
_T30 = "points_t30_right_hand_transformed.json"


@wp.kernel
def _nearest_cloud_point(
    cloud_pos: wp.array(dtype=wp.vec3),
    cloud_nrm: wp.array(dtype=wp.vec3),
    env_asset: wp.array(dtype=wp.int32),
    taxel_obj: wp.array2d(dtype=wp.vec3),
    k_pts: int,
    max_depth: float,
    depth: wp.array2d(dtype=wp.float32),
):
    e, j = wp.tid()
    base = env_asset[e] * k_pts
    p = taxel_obj[e, j]

    best = float(1.0e9)
    bi = int(0)
    for k in range(k_pts):
        d = wp.length_sq(p - cloud_pos[base + k])
        if d < best:
            best = d
            bi = k

    dist = wp.sqrt(best)
    if dist < max_depth and wp.dot(p - cloud_pos[base + bi], cloud_nrm[base + bi]) < 0.0:
        depth[e, j] = dist
    else:
        depth[e, j] = 0.0


def _sensor_file(body_name: str) -> str:
    return _T30 if "thumb" in body_name else _T16


def load_taxels(env) -> torch.Tensor:
    """(5, 120, 3) taxel positions in each fingertip's own link2 frame, in metres.

    Ordered to match env._fingertip_body_names, which is articulation-parse order and not the
    tuple order in robots.py.
    """
    root = Path(_TAXEL_DIR)
    cache: dict[str, np.ndarray] = {}
    per_finger = []
    for name in env._fingertip_body_names:
        fname = _sensor_file(name)
        if fname not in cache:
            pts = json.loads((root / fname).read_text())["measurement_points"]
            cache[fname] = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=np.float32) / 1000.0
        per_finger.append(cache[fname])
    stacked = np.stack(per_finger)
    if stacked.shape != (len(env._fingertip_body_names), NUM_TAXELS_PER_FINGER, 3):
        raise RuntimeError(f"taxel layout is {stacked.shape}, expected 5 by 120 by 3")
    return torch.tensor(stacked, device=env.device)


def _primitive(geom) -> trimesh.Trimesh:
    """A urdf primitive collision as a mesh."""
    box = geom.find("box")
    if box is not None:
        return trimesh.creation.box(extents=[float(v) for v in box.get("size").split()])
    cyl = geom.find("cylinder")
    if cyl is not None:
        return trimesh.creation.cylinder(radius=float(cyl.get("radius")),
                                         height=float(cyl.get("length")))
    sph = geom.find("sphere")
    if sph is not None:
        return trimesh.creation.icosphere(radius=float(sph.get("radius")))
    raise ValueError(f"collision geometry {[c.tag for c in geom]} is not a mesh or a primitive")


def _hull_union(urdf_path: str) -> trimesh.Trimesh:
    """The object's collision geometry as one mesh, which is what the solver pushes on.

    The procedural pool is all meshes with an explicit scale and an absolute filename. Authored
    assets are not: a part can be box primitives, and a mesh filename can be relative to its own
    urdf and carry no scale.
    """
    parts = []
    root = Path(urdf_path).parent
    for coll in ET.parse(urdf_path).getroot().iter("collision"):
        mesh_el = coll.find("geometry/mesh")
        if mesh_el is not None:
            fname = Path(mesh_el.get("filename"))
            m = trimesh.load_mesh(fname if fname.is_absolute() else root / fname, process=False)
            scale = mesh_el.get("scale")
            if scale is not None:
                m.apply_scale([float(v) for v in scale.split()])
        else:
            m = _primitive(coll.find("geometry"))
        origin = coll.find("origin")
        if origin is not None:
            t = trimesh.transformations.euler_matrix(
                *[float(v) for v in origin.get("rpy", "0 0 0").split()])
            t[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            m.apply_transform(t)
        parts.append(m)
    return trimesh.util.concatenate(parts)


def build_object_clouds(env, urdf_paths: list[str]) -> None:
    """Sample every pool object once and hold the clouds on device."""
    k = NUM_CLOUD_POINTS
    pos = np.empty((len(urdf_paths), k, 3), dtype=np.float32)
    nrm = np.empty((len(urdf_paths), k, 3), dtype=np.float32)
    for i, path in enumerate(urdf_paths):
        mesh = _hull_union(path)
        p, face = trimesh.sample.sample_surface(mesh, k)
        pos[i] = p
        nrm[i] = mesh.face_normals[face]

    env._tactile_cloud_pos = wp.from_torch(
        torch.tensor(pos.reshape(-1, 3), device=env.device), dtype=wp.vec3
    )
    env._tactile_cloud_nrm = wp.from_torch(
        torch.tensor(nrm.reshape(-1, 3), device=env.device), dtype=wp.vec3
    )


def _init_hand_buffers(env) -> None:
    """Deferred, because setup_scene runs before reset_utils resolves the fingertip bodies."""
    env._taxel_pos_local = load_taxels(env)
    env._tactile_depth = torch.zeros(
        (env.num_envs, len(env._fingertip_body_names) * NUM_TAXELS_PER_FINGER),
        device=env.device, dtype=torch.float32,
    )
    # Wrapped once. Rebuilding these views every step would allocate per frame.
    env._tactile_depth_wp = wp.from_torch(env._tactile_depth)
    env._tactile_env_asset = wp.from_torch(env._object_asset_index_per_env.to(torch.int32))


def compute_tactile(env, ft_pos_w, ft_quat_w, obj_pos_w, obj_quat_w, palm_pos, env_origins):
    """(N, 5) active taxel ratio and (N, 5, 3) contact centre relative to the palm.

    ft_pos_w and ft_quat_w are the raw link2 body poses, not the pad-centre offset ones, because
    the vendor coordinates are already absolute in that frame.
    """
    from isaaclab.utils.math import quat_apply, quat_conjugate

    if not hasattr(env, "_taxel_pos_local"):
        _init_hand_buffers(env)

    n_env, n_fing = env.num_envs, len(env._fingertip_body_names)
    n_tax = NUM_TAXELS_PER_FINGER

    quat = ft_quat_w.unsqueeze(2).expand(-1, -1, n_tax, -1).reshape(-1, 4)
    local = env._taxel_pos_local.unsqueeze(0).expand(n_env, -1, -1, -1).reshape(-1, 3)
    taxel_w = quat_apply(quat, local).reshape(n_env, n_fing, n_tax, 3) + ft_pos_w.unsqueeze(2)

    flat = taxel_w.reshape(n_env, n_fing * n_tax, 3)
    obj_q = quat_conjugate(obj_quat_w).unsqueeze(1).expand(-1, n_fing * n_tax, -1).reshape(-1, 4)
    taxel_obj = quat_apply(obj_q, (flat - obj_pos_w.unsqueeze(1)).reshape(-1, 3)).reshape(
        n_env, n_fing * n_tax, 3
    )

    wp.launch(
        _nearest_cloud_point,
        dim=(n_env, n_fing * n_tax),
        inputs=[
            env._tactile_cloud_pos,
            env._tactile_cloud_nrm,
            env._tactile_env_asset,
            wp.from_torch(taxel_obj.contiguous(), dtype=wp.vec3),
            NUM_CLOUD_POINTS,
            MAX_CONTACT_DEPTH,
            env._tactile_depth_wp,
        ],
        device=env._tactile_cloud_pos.device,
    )

    depth = env._tactile_depth.view(n_env, n_fing, n_tax)
    # The scalar is the fraction of that fingertip's taxels in contact. Summed depth was measured
    # against the solver's own force at r 0.187 and the ratio at 0.176, so the magnitude adds
    # nothing the count does not already carry. The ratio also transfers without a calibration,
    # since the real hand counts the same 120 sites above its noise floor.
    ratio = (depth > 0).to(depth.dtype).mean(dim=-1)
    total = depth.sum(dim=-1)
    weighted = (depth.unsqueeze(-1) * taxel_w).sum(dim=-2)
    touching = total > 0.0
    # The centre stays depth weighted. Dropping that weighting costs 25.1 down to 12.2 in the
    # paper's own ablation, so only the exposed scalar changes.
    centre_w = torch.where(touching.unsqueeze(-1), weighted / total.clamp_min(1e-12).unsqueeze(-1),
                           torch.zeros_like(weighted))
    # Same convention as fingertip_pos_rel_palm at obs_utils.py:302.
    centre = torch.where(
        touching.unsqueeze(-1),
        (centre_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1),
        torch.zeros_like(centre_w),
    )
    return ratio, centre
