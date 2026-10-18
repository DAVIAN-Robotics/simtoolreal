"""S4. Drop IK solutions whose arm spheres enter the object, with cuRobo's collision checker.

The object's 16 collision pieces are mesh obstacles with no activation distance, so a positive scene
distance means a sphere is inside a piece. Each object pose moves the pieces, then its solutions are
checked in one batch. A self test first proves the sign, a object placed on an arm sphere must collide
and a object 10 m away must not.
Usage: reset_arm_object.py <shard npz> <robot yml> <poses npz>
"""
import sys
import time

import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.scene import Mesh, Scene
from curobo.types import Pose

import reset_common as rc

shard, robot_yml, poses = sys.argv[1:4]
dev = torch.device("cuda")
t0 = time.time()
S, P = np.load(shard), np.load(poses)
robot = yaml.safe_load(open(robot_yml))
pieces = rc.object_pieces()
chk = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
    robot_config=robot, scene_model=Scene(mesh=[
        Mesh(name=f"obj{i}", pose=[0, 0, 10.0, 1, 0, 0, 0], vertices=p.vertices.tolist(), faces=p.faces.tolist())
        for i, p in enumerate(pieces)]),
    collision_activation_distance=0.0, n_meshes=16, n_cuboids=1))
T_wb = rc.homog(rc.quat_to_mat(torch.tensor(rc.BASE_QUAT, dtype=torch.float64)), torch.tensor(rc.BASE_POS, dtype=torch.float64)).to(dev)
T_bw = torch.linalg.inv(T_wb)


def place(T_bm):
    for i in range(16):
        chk.scene_model.update_obstacle_pose(f"obj{i}", Pose.from_matrix(T_bm.float()[None]))


def hits(q):
    out = torch.empty(len(q), dtype=torch.bool, device=dev)
    for s in range(0, len(q), 65536):
        d, _ = chk.get_scene_self_collision_distance_from_joints(q[s:s + 65536, None, :].contiguous())
        out[s:s + 65536] = d.reshape(d.shape[0], -1).max(-1).values > 0
    return out


q = torch.as_tensor(S["q"], device=dev)
# Self test on the first solution.
sp = chk.get_kinematics(q[:1, None]).robot_spheres[0, 0]
big = sp[sp[:, 3].argmax()]
c0 = torch.as_tensor(pieces[0].centroid, device=dev)
T = torch.eye(4, dtype=torch.float64, device=dev)
T[:3, 3] = big[:3].double() - c0
place(T)
assert hits(q[:1]).item(), "a object on an arm sphere reads collision free"
T[:3, 3] = torch.tensor([0.0, 0.0, 10.0], device=dev)
place(T)
assert not hits(q[:1]).item(), "a object 10 m away reads colliding"

pose = torch.as_tensor(S["pose"].astype(np.int64), device=dev)
obj_ok = torch.ones(len(q), dtype=torch.bool, device=dev)
for p in torch.unique(pose).tolist():
    idx = torch.nonzero(pose == p).squeeze(-1)
    T_wm = rc.homog(rc.quat_to_mat(torch.as_tensor(P["quat"][p], device=dev)), torch.as_tensor(P["pos"][p], device=dev))
    place(T_bw @ T_wm)
    obj_ok[idx] = ~hits(q[idx])
np.save(shard.replace(".npz", "_obj.npy"), obj_ok.cpu().numpy())
print(f"{shard} solutions {len(q)} object ok {int(obj_ok.sum())} in {time.time() - t0:.1f} s", flush=True)
