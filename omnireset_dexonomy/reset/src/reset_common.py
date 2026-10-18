"""Shared geometry for the object reset states: the 050 scene, the training URDF, grasps and object hulls.

All scene numbers come from the 050 tree and are asserted against its source text, so a changed
tree fails here. Frames are env-local, the frame SimToolReal's goal box and base pose live in.
"""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(os.environ.get("RESET_ROOT", "/home/nas_main/sibeenkim/draftrec"))
TREE = Path(os.environ.get("RESET_TREE", ROOT / "repos/simtoolreal-050"))  # dj holds a copy of the files used here
URDF = TREE / "assets/urdf/fr3_xhand_description/fr3_xhand/fr3_xhand.urdf"
# The PoC object, the cap16 pool entry at the median own-test pass count, 3,984 passes.
POC_STEM = "core__" + "mug-28f1e7bc572a633cb9946438ed40eeb9_s008"  # its dataset id
OBJ_URDF = ROOT / f"assets_040/objects_cap16/{POC_STEM}.urdf"
OBJ_SCALE = (2.0238671672010096, 2.4749304404446675, 3.01670702)  # objects_cap16/pool.json
GRASPS = ROOT / f"_graspoc/cap16/poc_object/{POC_STEM}.npz"
SPHERES = ROOT / "_curobo2/fr3cmp/cricket_fr3_spheres.yml"

BASE_POS = (0.0, 0.48, 0.53)  # scene_utils.py:59
BASE_QUAT = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)  # robots.py fr3-xhand
TABLE_Z = 0.38  # simtoolreal_env_cfg.py table_reset_z, the table root
TABLE_BOX = (0.9, 0.9, 0.3)  # table_davian.urdf collision box
TABLE_OFF = (0.0, -0.09, 0.0)  # table_davian.urdf collision origin
TABLE_TOP = TABLE_Z + TABLE_BOX[2] / 2
BOX_MIN, BOX_MAX = (-0.35, -0.2, 0.6), (0.35, 0.2, 0.95)  # target_volume_mins, target_volume_maxs
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_JOINTS = ["thumb_joint0", "thumb_joint1", "thumb_joint2", "index_joint0", "index_joint1", "index_joint2",
               "middle_joint0", "middle_joint1", "ring_joint0", "ring_joint1", "pinky_joint0", "pinky_joint1"]
ARM_DEFAULT = [0.0, 0.2618, 0.0, -2.0071, 0.0, 2.3562, -0.7854]  # robots.py fr3-xhand arm_default_pos


def check_tree():
    su = (TREE / "isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py").read_text()
    assert "pos=(0.0, 0.48, 0.53)," in su
    cfg = (TREE / "isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py").read_text()
    for s in ('table_urdf: str = "assets/urdf/table_davian.urdf"', "table_reset_z: float = 0.38",
              "target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)",
              "target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)",
              "target_volume_region_scale: float = 1.0"):
        assert s in cfg, s
    rb = (TREE / "isaacsimenvs/tasks/simtoolreal/robots.py").read_text()
    assert "base_quat_wxyz=(0.7071067811865476, 0.0, 0.0, -0.7071067811865476)" in rb
    assert '"fr3_joint4": -2.0071, "fr3_joint5": 0.0, "fr3_joint6": 2.3562,' in rb
    for j in HAND_JOINTS:
        assert f'"{j}"' in rb, j
    tb = ET.parse(TREE / "assets/urdf/table_davian.urdf").getroot().find("link/collision")
    assert tb.find("geometry/box").get("size") == "0.9 0.9 0.3" and tb.find("origin").get("xyz") == "0 -0.09 0"


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def origin_T(el):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy_mat(*rpy), xyz
    return T


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def mat_to_quat(R):
    """wxyz, R (..., 3, 3)."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.stack([1 + t, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                     1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    i = q.argmax(-1)
    w = torch.stack([1 + t, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
    x = torch.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1)
    y = torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]], -1)
    z = torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    out = torch.stack([w, x, y, z], -2).gather(-2, i[..., None, None].expand(*i.shape, 1, 4)).squeeze(-2)
    out = torch.nn.functional.normalize(out, dim=-1)
    return out * torch.where(out[..., :1] < 0, -1.0, 1.0)


def homog(R, p):
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3], T[..., :3, 3], T[..., 3, 3] = R, p, 1.0
    return T


class Urdf:
    def __init__(self, path=URDF):
        self.root = ET.parse(path).getroot()
        self.links = {l.get("name"): l for l in self.root.findall("link")}
        self.joints = {j.get("name"): j for j in self.root.findall("joint")}
        self.child_joint = {j.find("child").get("link"): j for j in self.root.findall("joint")}

    def subtree(self, root):
        out, stack = [root], [root]
        while stack:
            p = stack.pop()
            for j in self.root.findall("joint"):
                if j.find("parent").get("link") == p:
                    c = j.find("child").get("link")
                    out.append(c)
                    stack.append(c)
        return out

    def fk_from(self, root, q):
        """Link frames relative to ``root`` at joint values q (dict name -> (B,) tensor). (B,4,4) per link."""
        B = next(iter(q.values())).shape[0]
        dev = next(iter(q.values())).device
        T = {root: torch.eye(4, dtype=torch.float64, device=dev).expand(B, 4, 4)}
        for l in self.subtree(root)[1:]:
            j = self.child_joint[l]
            Tj = T[j.find("parent").get("link")] @ torch.as_tensor(origin_T(j), device=dev)
            if j.get("type") == "revolute":
                a = torch.as_tensor([float(v) for v in j.find("axis").get("xyz").split()], dtype=torch.float64, device=dev)
                K = torch.zeros(3, 3, dtype=torch.float64, device=dev)
                K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
                th = q[j.get("name")].double()[:, None, None]
                R = torch.eye(3, dtype=torch.float64, device=dev) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
                Tj = Tj @ homog(R, torch.zeros(B, 3, dtype=torch.float64, device=dev))
            else:
                assert j.get("type") == "fixed", j.get("name")
            T[l] = Tj
        return T

    def collision_hulls(self, link):
        """Convex hull vertices of each <collision> mesh of a link, in the link frame."""
        out = []
        for c in self.links[link].findall("collision"):
            m = c.find("geometry/mesh")
            f = (URDF.parent / m.get("filename")).resolve()
            v = trimesh.load(f, force="mesh").convex_hull.vertices
            if m.get("scale"):
                v = v * np.array([float(s) for s in m.get("scale").split()])
            T = origin_T(c)
            out.append(v @ T[:3, :3].T + T[:3, 3])
        return out


def t_7_palm(u):
    T = origin_T(u.joints["fr3_joint8"]) @ origin_T(u.joints["hand_mount"])
    ref = np.array([[-2.597e-06, 0.707102885, 0.707110677, -0.02616295], [-2.597e-06, -0.707110677, 0.707102885, -0.02616295],
                    [1.0, -0.0, 3.673e-06, 0.149], [0, 0, 0, 1]])
    assert np.abs(T - ref).max() < 1e-5, T  # reference/xhand/hand_root.md
    return T


def object_pieces(urdf=OBJ_URDF):
    """The object's collision pieces in its URDF root frame, as trimesh convex hulls. cap16 objects have 1 to 16."""
    out = []
    for c in ET.parse(urdf).getroot().iter("collision"):
        m = c.find("geometry/mesh")
        mesh = trimesh.load(m.get("filename"), force="mesh")
        mesh.apply_scale([float(s) for s in m.get("scale").split()])
        mesh.apply_transform(origin_T(c))
        out.append(mesh.convex_hull)
    assert 1 <= len(out) <= 16, (urdf, len(out))
    return out


def load_grasps(dev):
    g = np.load(GRASPS)
    out = {k: torch.as_tensor(g[k], device=dev) for k in ("hand_pos", "hand_quat", "q_grasp", "q_squeeze", "q_pregrasp")}
    out["tmpl"] = g["tmpl"]
    assert out["hand_pos"].shape == (3984, 3)
    return out
