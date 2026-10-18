"""Make the own test apply the capped mass exactly.

MuJoCo integrates every convex piece, so where pieces overlap the overlap counts twice, while upstream's density
formula divides by the union's volume (info mass at density 1). This writes collision_volume = the sum of the
piece volumes into each entry's info/simplified.json (atomic rename, the running own test reads these files), so
the patched eval_data.py can set density = mass_kg / collision_volume, which makes the MuJoCo mass equal mass_kg.
It then builds the eval env for every entry with that density and asserts the MuJoCo mass is the target to 1e-6.
Usage: agent4_cap16_fix_mass.py <prep_root> <shard> <n shards> <n proc>
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import trimesh

root, shard, n_shard, n_proc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
CLONE = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/Dexonomy"
sys.path.insert(0, CLONE)
from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv  # noqa: E402
from dexonomy.util.file_util import load_scene_cfg  # noqa: E402


def fix(p):
    sc = load_scene_cfg(p)
    (name, o), = sc["scene"].items()
    d = os.path.dirname(os.path.dirname(o["xml_path"]))
    vol = sum(trimesh.load(f, force="mesh", process=False).volume for f in sorted(glob.glob(f"{d}/urdf/meshes/convex_piece_*.obj")))
    info = json.load(open(o["info_path"]))
    info["collision_volume"] = vol
    tmp = os.path.join(os.path.dirname(o["info_path"]), ".simplified.json.tmp")
    json.dump(info, open(tmp, "w"), indent=1)
    os.rename(tmp, o["info_path"])
    o["density"] = o["mass_kg"] / (vol * np.prod(o["scale"]))
    env = MuJoCo_EvalEnv(hand_cfg=HandCfg(f"{CLONE}/assets/hand/xhand/right.xml", freejoint=True), scene_cfg=sc,
                         sim_cfg=MuJoCo_EvalCfg(miu_coef=[0.6, 0.02]), debug_render=False, debug_view=False, hard=False)
    m = float(env._model.body_subtreemass[env._obj_active_bodyid])
    assert abs(m / o["mass_kg"] - 1) < 1e-6, (name, m, o["mass_kg"])
    return name, vol / (info["mass"] / info["density"])


if __name__ == "__main__":
    files = sorted(glob.glob(f"{root}/scene_cfg/*/floating/real.npy"))[shard::n_shard]
    with Pool(n_proc) as pool:
        rows = pool.map(fix, files, chunksize=4)
    r = np.array([x for _, x in rows])
    print(f"shard {shard}: {len(rows)} info files updated, MuJoCo mass = target on all; pieces/union volume max {r.max():.4f}, over 1.01: {(r > 1.01).sum()}")
