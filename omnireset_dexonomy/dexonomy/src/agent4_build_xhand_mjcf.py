"""Build the Dexonomy hand MJCF for the reference XHand (agent 1, _graspoc/reference/xhand).

Fixed-joint links are merged into their movable parent, as the training importer does
(merge_fixed_joints=True). Collision geoms are the reference convex hulls, one per collision STL,
listed with the body's own hull first. Visual geoms are the STLs. Gains follow the training drive,
kp 3.0 and damping 0.1, with forcerange equal to the URDF effort limit.

Usage: python agent4_build_xhand_mjcf.py <reference_xhand_dir> <out_hand_dir>
Writes <out_hand_dir>/right.xml, meshes/ (STL copies) and hulls/ (OBJ copies).
"""
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

KP = 3.0
DAMPING = 0.1

ref_dir, out_dir = sys.argv[1], sys.argv[2]
root = ET.parse(os.path.join(ref_dir, "xhand_right.urdf")).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")
child_joint = {j.find("child").get("link"): j for j in joints}
children = {}
for j in joints:
    children.setdefault(j.find("parent").get("link"), []).append(j)


def vec(s):
    return np.array([float(x) for x in s.split()])


def origin_T(elem):
    o = elem.find("origin")
    T = np.eye(4)
    if o is None:
        return T
    xyz = vec(o.get("xyz", "0 0 0"))
    rpy = vec(o.get("rpy", "0 0 0"))
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def mesh_stem(elem):
    return os.path.splitext(os.path.basename(elem.find("geometry/mesh").get("filename")))[0]


# Group every link under its movable body. merged[body] = [(link, T_body_link), ...], body's own link first.
merged = {}


def collect(body, link, T_body_link):
    merged.setdefault(body, []).append((link, T_body_link))
    for j in children.get(link, []):
        if j.get("type") == "fixed":
            collect(body, j.find("child").get("link"), T_body_link @ origin_T(j))


def fused_inertial(body):
    ms, coms, Is = [], [], []
    for link, T in merged[body]:
        inert = links[link].find("inertial")
        m = float(inert.find("mass").get("value"))
        Ti = T @ origin_T(inert)
        a = inert.find("inertia").attrib
        I = np.array(
            [
                [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
            ]
        )
        ms.append(m)
        coms.append(Ti[:3, 3])
        Is.append(Ti[:3, :3] @ I @ Ti[:3, :3].T)
    M = sum(ms)
    com = sum(m * c for m, c in zip(ms, coms)) / M
    I_tot = np.zeros((3, 3))
    for m, c, I in zip(ms, coms, Is):
        d = c - com
        I_tot += I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, com, I_tot


spec = mujoco.MjSpec()
spec.modelname = "xhand_right"
spec.compiler.degree = False
spec.compiler.autolimits = True
spec.meshdir = os.path.abspath(out_dir)
os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "hulls"), exist_ok=True)
added_mesh = set()


def add_mesh(name, src, dst_rel):
    if name in added_mesh:
        return
    shutil.copyfile(src, os.path.join(out_dir, dst_rel))
    spec.add_mesh(name=name, file=dst_rel)
    added_mesh.add(name)


actuated = []


def build(parent_spec_body, body, T_parent_body, joint):
    b = parent_spec_body.add_body(name=body, pos=T_parent_body[:3, 3], quat=mat2quat(T_parent_body[:3, :3]))
    collect(body, body, np.eye(4))
    M, com, I = fused_inertial(body)
    b.explicitinertial = True
    b.mass = M
    b.ipos = com
    b.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    if joint is not None:
        lim = joint.find("limit")
        jj = b.add_joint(
            name=joint.get("name"),
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=vec(joint.find("axis").get("xyz")),
            range=[float(lim.get("lower")), float(lim.get("upper"))],
            damping=DAMPING,
            armature=0.0,
            frictionloss=0.0,
        )
        actuated.append((joint.get("name"), float(lim.get("lower")), float(lim.get("upper")), float(lim.get("effort"))))
    for link, T in merged[body]:
        # collision first, so the body's first collision geom is its own hull (GraspADMM projects onto it)
        for c in links[link].findall("collision"):
            stem = mesh_stem(c)
            add_mesh("c_" + stem, os.path.join(ref_dir, "collision_hulls", stem + ".obj"), f"hulls/{stem}.obj")
            Tg = T @ origin_T(c)
            b.add_geom(
                name=f"{link}_collision",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="c_" + stem,
                pos=Tg[:3, 3],
                quat=mat2quat(Tg[:3, :3]),
                group=3,
                rgba=[0.6, 0.6, 0.6, 1],
            )
    for link, T in merged[body]:
        for v in links[link].findall("visual"):
            stem = mesh_stem(v)
            add_mesh("v_" + stem, os.path.join(ref_dir, "meshes", stem + ".STL"), f"meshes/{stem}.STL")
            Tg = T @ origin_T(v)
            b.add_geom(
                name=f"{link}_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="v_" + stem,
                pos=Tg[:3, 3],
                quat=mat2quat(Tg[:3, :3]),
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
                rgba=[0.8, 0.8, 0.8, 1],
            )
    for link, _ in merged[body]:
        for j in children.get(link, []):
            if j.get("type") == "revolute":
                T_link = dict(merged[body])[link]
                build(b, j.find("child").get("link"), T_link @ origin_T(j), j)


build(spec.worldbody, "palm", np.eye(4), None)
for name, lo, hi, eff in actuated:
    spec.add_actuator(
        name=name,
        target=name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[KP] + [0] * 9,
        biasprm=[0, -KP, 0] + [0] * 7,
        ctrlrange=[lo, hi],
        forcerange=[-eff, eff],
        ctrllimited=True,
        forcelimited=True,
    )

model = spec.compile()
xml = spec.to_xml().replace(os.path.abspath(out_dir), ".")
with open(os.path.join(out_dir, "right.xml"), "w") as f:
    f.write(xml)
print("bodies", [model.body(i).name for i in range(model.nbody)])
print("joints", [model.joint(i).name for i in range(model.njnt)])
print("nq", model.nq, "nu", model.nu, "ngeom", model.ngeom, "nmesh", model.nmesh)
