"""Static MuJoCo renders of reset states, FR3 plus XHand from the training URDF's visual meshes, the object's
visual mesh at its pose, the table box and the base plate. Env-local frame, as the 050 scene.
"""
import os
import xml.etree.ElementTree as ET
from functools import lru_cache

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import trimesh

import reset_common as rc

U = rc.Urdf()
T_WB = np.eye(4)
T_WB[:3, :3] = rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)).numpy()
T_WB[:3, 3] = rc.BASE_POS
ARM_GREY, HAND_DARK, OBJ_BLUE = [0.86, 0.86, 0.84, 1.0], [0.18, 0.18, 0.2, 1.0], [0.25, 0.5, 0.95, 1.0]


@lru_cache(maxsize=None)
def link_meshes(link):
    out = []
    for v in U.links[link].findall("visual"):
        m = v.find("geometry/mesh")
        if m is None:
            continue
        f = (rc.URDF.parent / m.get("filename")).resolve()
        mesh = trimesh.load(f, force="mesh")
        mesh.apply_transform(rc.origin_T(v))
        out.append(mesh)
    return out


@lru_cache(maxsize=None)
def object_mesh():
    v = ET.parse(rc.OBJ_URDF).getroot().find("link/visual")
    mesh = trimesh.load(v.find("geometry/mesh").get("filename"), force="mesh")
    mesh.apply_scale([float(s) for s in v.find("geometry/mesh").get("scale").split()])
    mesh.apply_transform(rc.origin_T(v))
    return mesh


def robot_link_poses(q19):
    q = {j: torch.tensor([float(x)], dtype=torch.float64) for j, x in zip(rc.ARM_JOINTS + rc.HAND_JOINTS, q19)}
    T = U.fk_from("fr3_link0", q)
    return {l: T_WB @ t[0].numpy() for l, t in T.items()}


def add_mesh(spec, name, mesh, T, rgba):
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    spec.add_mesh(name=name, uservert=v.ravel().tolist(), userface=mesh.faces.ravel().tolist())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba, contype=0, conaffinity=0)


def build(robots, obj_pose=None, size=640, table=True, obj_rgba=OBJ_BLUE):
    """robots, list of (q19, alpha). obj_pose (pos3, quat wxyz4) env-local."""
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = spec.visual.global_.offheight = size
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.worldbody.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.2, 0.2, -1], diffuse=[0.35, 0.35, 0.35])
    if table:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[b / 2 for b in rc.TABLE_BOX],
                                pos=[rc.TABLE_OFF[0], rc.TABLE_OFF[1], rc.TABLE_Z], rgba=[0.82, 0.62, 0.45, 1], contype=0, conaffinity=0)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.125, 0.125, 0.15], pos=[0, 0.515, rc.TABLE_Z],
                                rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)
    for k, (q19, alpha) in enumerate(robots):
        for link, T in robot_link_poses(q19).items():
            base = HAND_DARK if not link.startswith("fr3_") else ARM_GREY
            for i, m in enumerate(link_meshes(link)):
                add_mesh(spec, f"r{k}_{link}_{i}", m, T, base[:3] + [alpha])
    if obj_pose is not None:
        T = np.eye(4)
        T[:3, :3] = rc.quat_to_mat(torch.as_tensor(np.asarray(obj_pose[1], dtype=np.float64))).numpy()
        T[:3, 3] = obj_pose[0]
        add_mesh(spec, "obj", object_mesh(), T, obj_rgba)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def render(m, d, lookat, distance, azimuth, elevation, size=640):
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
    with mujoco.Renderer(m, size, size) as r:
        r.update_scene(d, cam)
        return r.render().copy()
