"""Pose-viewer pages, the wandb interactive media view, for the randomly picked passing and failing states.

Each page plays the recorded Isaac test of one state at 60 Hz, the arm and hand from the training URDF,
the object at its recorded pose, the table at its reset pose. The goal ghost is parked 5 m below.
Loads the viewer without Isaac Lab, as the validator's build_hold_html.py does.
Usage: reset_html.py <reset dir> <run dir> <out dir>
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

import reset_common as rc

D, RUN, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
for name, rel in [("isaacsimenvs", "isaacsimenvs"), ("isaacsimenvs.utils", "isaacsimenvs/utils")]:
    mod = types.ModuleType(name)
    mod.__path__ = [str(rc.TREE / rel)]
    sys.modules[name] = mod
spec = importlib.util.spec_from_file_location("pose_viewer", rc.TREE / "isaacsimenvs/tasks/simtoolreal/pose_viewer.py")
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)

pick = np.load(f"{D}/html_pick.npz", allow_pickle=True)
files = [str(f) for f in pick["files"]]
B = {int(np.load(f)["row"][0]): np.load(f) for f in files}
starts = np.array(sorted(B))
G = np.load(rc.GRASPS)
table_path = rc.TREE / "assets/urdf/table_davian.urdf"
xyzw = lambda q: np.array([q[1], q[2], q[3], q[0]], np.float32)
base = np.concatenate([np.array(rc.BASE_POS, np.float32), xyzw(rc.BASE_QUAT)])
table = np.array([0, 0, rc.TABLE_Z, 0, 0, 0, 1], np.float32)
goal = np.array([0, 0, -5.0, 0, 0, 0, 1], np.float32)
rows = []
for tag, idx in (("pass", pick["passing"]), ("fail", pick["failing"])):
    for k, r in enumerate(idx):
        b = B[int(starts[starts <= r].max())]
        j = int(r - b["row"][0])
        q, o = b["rec_q"][:, j].astype(np.float32), b["rec_o"][:, j].astype(np.float32)
        frames = [{"robot_joint_names": rc.ARM_JOINTS + rc.HAND_JOINTS, "robot_joint_pos": q[t], "robot_base_pose": base,
                   "object_pose": np.concatenate([o[t, :3], xyzw(o[t, 3:])]), "goal_pose": goal, "table_pose": table}
                  for t in range(len(q))]
        # The page names the object generically, its dataset id stays out of the page.
        html = pv.build_pose_viewer_html(frames=frames, robot_urdf_rel="assets/urdf/fr3_xhand_description/fr3_xhand/fr3_xhand.urdf",
                                         object_urdf_text=rc.OBJ_URDF.read_text(), table_urdf_text=table_path.read_text(),
                                         object_urdf_path=rc.OBJ_URDF, table_urdf_path=table_path)
        html = html.replace(rc.POC_STEM, "poc_object").replace(rc.POC_STEM.split("__")[1].rsplit("_", 1)[0], "poc_object")
        name = f"{tag}_{k:02d}_row{int(r)}.html"
        (OUT / name).write_text(html)
        rows.append((name, str(pick["reason"][r]), str(G["tmpl"][int(b["grasp"][j])]), len(q)))
        print(name, rows[-1][1], rows[-1][2], f"{len(html) / 1e6:.1f} MB", flush=True)
