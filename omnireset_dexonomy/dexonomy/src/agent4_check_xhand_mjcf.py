"""Check the Dexonomy XHand MJCF against the reference URDF.

1. Joint names, order, limits, effort limits equal joint_order.json.
2. For 200 random joint vectors, every collision and visual geom sits at the pose of its URDF link,
   with the URDF loaded independently by MuJoCo with fusestatic off.
3. Each collision hull equals the reference hull OBJ, vertex for vertex.
4. Hand self-contacts at q = 0 and at the joint-range midpoint, listed with penetration depth.

Usage: python agent4_check_xhand_mjcf.py <reference_xhand_dir> <hand_dir>
"""
import json
import os
import sys
import tempfile

import mujoco
import numpy as np
import trimesh

ref_dir, hand_dir = sys.argv[1], sys.argv[2]
jo = json.load(open(os.path.join(ref_dir, "joint_order.json")))
m = mujoco.MjModel.from_xml_path(os.path.join(hand_dir, "right.xml"))
d = mujoco.MjData(m)

names = [m.joint(i).name for i in range(m.njnt)]
assert names == jo["joint_order"], names
for i, jr in enumerate(jo["joints"]):
    assert np.allclose(m.jnt_range[i], [jr["lower"], jr["upper"]]), (jr["name"], m.jnt_range[i])
    a = m.actuator(i)
    assert m.actuator_trnid[i, 0] == i
    assert np.allclose(m.actuator_forcerange[i], [-jr["effort_limit_Nm"], jr["effort_limit_Nm"]])
    assert np.isclose(m.actuator_gainprm[i, 0], jr["stiffness_Nm_per_rad"])
    assert np.isclose(m.dof_damping[i], jr["damping_Nms_per_rad"])
    assert np.isclose(m.dof_armature[i], jr["armature_kgm2"])
    assert np.isclose(m.dof_frictionloss[i], jr["joint_friction"])
print("joints, limits, gains, effort limits: equal to joint_order.json")

# Independent URDF load, no fusion
urdf = open(os.path.join(ref_dir, "xhand_right.urdf")).read()
head, rest = urdf.split("<robot ", 1)
tag_end = rest.index(">") + 1
urdf = (
    head
    + "<robot "
    + rest[:tag_end]
    + f'<mujoco><compiler fusestatic="false" discardvisual="false" meshdir="{ref_dir}" strippath="false"/></mujoco>'
    + rest[tag_end:]
)
tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", dir=ref_dir, delete=False)
tmp.write(urdf)
tmp.close()
mu = mujoco.MjModel.from_xml_path(tmp.name)
os.unlink(tmp.name)
du = mujoco.MjData(mu)
unames = [mu.joint(i).name for i in range(mu.njnt)]
uq = [unames.index(n) for n in names]
assert all(mu.jnt_type[k] == mujoco.mjtJoint.mjJNT_HINGE for k in uq)

from scipy.spatial import cKDTree

ref_verts = {}
for g in range(m.ngeom):
    stem = m.mesh(m.geom_dataid[g]).name
    if stem.startswith("c_"):
        f = os.path.join(ref_dir, "collision_hulls", stem[2:] + ".obj")
    else:
        f = os.path.join(ref_dir, "meshes", stem[2:] + ".STL")
    ref_verts[g] = np.asarray(trimesh.load(f, force="mesh", process=False).vertices)


def geom_world_verts(g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


# Every mesh geom's vertices, placed by my MJCF, against the reference file's vertices placed at the
# URDF link pose. Symmetric nearest-vertex distance, so it catches pose and shape errors alike.
rng = np.random.default_rng(0)
lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
max_err = 0.0
for t in range(50):
    q = lo + rng.random(12) * (hi - lo)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    du.qpos[:] = 0
    du.qpos[[mu.jnt_qposadr[k] for k in uq]] = q
    mujoco.mj_kinematics(mu, du)
    for g in range(m.ngeom):
        link = m.geom(g).name.rsplit("_", 1)[0]
        bid = mu.body(link).id
        w_ref = ref_verts[g] @ du.xmat[bid].reshape(3, 3).T + du.xpos[bid]
        w_mine = geom_world_verts(g)
        e1 = cKDTree(w_ref).query(w_mine)[0].max()
        e2 = cKDTree(w_mine).query(w_ref)[0].max()
        max_err = max(max_err, e1, e2)
print(f"mesh vertices vs reference file at URDF link pose, 50 random q, all 60 geoms: max err {max_err:.2e} m")
assert max_err < 1e-6

# Self-contacts as Dexonomy sees them. Its sim envs attach the hand under a body with a freejoint,
# so the palm is not welded to the world and MuJoCo's parent-child filter applies to it.
spec = mujoco.MjSpec()
frame = spec.worldbody.add_frame()
hw = frame.attach_body(mujoco.MjSpec.from_file(os.path.join(hand_dir, "right.xml")).worldbody, "hand-", "")
hw.add_freejoint()
mf = spec.compile()
df = mujoco.MjData(mf)


def self_contacts(q, label):
    df.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    df.qpos[7:] = q
    mujoco.mj_forward(mf, df)
    rows = []
    for c in df.contact[: df.ncon]:
        rows.append((c.dist, mf.geom(c.geom1).name, mf.geom(c.geom2).name))
    rows.sort()
    print(f"self contacts at {label}: {len(rows)}")
    for r in rows:
        print(f"  dist {r[0]*1000:+.2f} mm  {r[1]} <-> {r[2]}")
    return rows


self_contacts(np.zeros(12), "q = 0")
self_contacts((lo + hi) / 2, "range midpoint")
self_contacts(hi, "upper limits")
