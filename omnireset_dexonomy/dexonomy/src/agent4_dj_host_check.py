"""Per-host check of the dj dexonomy env: import paths, versions, the GPU torch sees, a torch matmul and a
warp mesh-query kernel on that GPU (the JIT path Dexonomy's init uses), and the xhand MuJoCo env on CPU.
Run with CUDA_VISIBLE_DEVICES set to one free GPU, from any cwd."""
import os
import socket

import mujoco
import numpy as np
import torch
import trimesh
import warp as wp

import dexonomy
import dexonomy.main  # noqa: F401
from dexonomy.sim import HandCfg, MuJoCo_OptEnv

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/Dexonomy"
# dexonomy has no top-level __init__, so the editable finder adds a path-hook entry next to the clone
paths = [p for p in dexonomy.__path__ if os.path.isdir(p)]
assert paths == [f"{D}/dexonomy"], list(dexonomy.__path__)
assert dexonomy.main.__file__.startswith(D), dexonomy.main.__file__
cap = torch.cuda.get_device_capability(0)
x = torch.randn(2048, 2048, device="cuda")
mm = float((x @ x).abs().sum())

wp.init()
box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
mesh = wp.Mesh(points=wp.array(box.vertices, dtype=wp.vec3, device="cuda:0"),
               indices=wp.array(box.faces.ravel(), dtype=int, device="cuda:0"))


@wp.kernel
def closest(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, pts[i], 1.0)
    out[i] = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - pts[i])


pts = wp.array(np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.3]]), dtype=wp.vec3, device="cuda:0")
out = wp.zeros(2, dtype=float, device="cuda:0")
wp.launch(closest, dim=2, inputs=[mesh.id, pts], outputs=[out], device="cuda:0")
d = out.numpy()
assert np.allclose(d, [0.15, 0.25], atol=1e-5), d

env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=f"{D}/assets/hand/xhand/right.xml", freejoint=True))
print(f"{socket.gethostname()} OK dexonomy={paths[0]} torch={torch.__version__} cuda_rt={torch.version.cuda} "
      f"gpu={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} visible={os.environ['CUDA_VISIBLE_DEVICES']} "
      f"matmul_ok={np.isfinite(mm)} warp={wp.config.version} warp_mesh_query={d.round(3).tolist()} "
      f"mujoco={mujoco.__version__} xhand_nq={env._model.nq}")
