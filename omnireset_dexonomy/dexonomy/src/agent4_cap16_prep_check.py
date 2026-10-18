"""Check every cap16 scene cfg through Dexonomy's own code: load_scene_cfg, then MuJoCo_EvalEnv with the xhand,
with the own test's density computed as the patched eval_data.py does it (mass_kg / (info volume)), and
confirm the object's MuJoCo mass. Also checks mass_kg = clip(mass_urdf_kg, 0.05, 0.5) and absolute paths.
Usage: agent4_cap16_prep_check.py <prep_root> <shard> <n shards> <n proc> <out.json>"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

root, shard, n_shard, n_proc, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
CLONE = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/Dexonomy"
sys.path.insert(0, CLONE)
from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv  # noqa: E402
from dexonomy.util.file_util import load_json, load_scene_cfg  # noqa: E402


def check(p):
    sc = load_scene_cfg(p)
    (name, o), = sc["scene"].items()
    for k in ["file_path", "xml_path", "urdf_path", "info_path"]:
        assert os.path.isabs(o[k]) and os.path.exists(o[k]), (p, k)
    assert np.isclose(o["mass_kg"], np.clip(o["mass_urdf_kg"], 0.05, 0.5)), p
    info = load_json(o["info_path"])
    o_coef = info["mass"] / (info["density"] * (info["scale"] ** 3))
    o["density"] = o["mass_kg"] / (o_coef * np.prod(o["scale"]))  # as the patched eval_data.py
    env = MuJoCo_EvalEnv(hand_cfg=HandCfg(f"{CLONE}/assets/hand/xhand/right.xml", freejoint=True), scene_cfg=sc,
                         sim_cfg=MuJoCo_EvalCfg(miu_coef=[0.6, 0.02]), debug_render=False, debug_view=False, hard=False)
    mj_mass = float(env._model.body_subtreemass[env._obj_active_bodyid])
    return {"stem": name, "mass_kg": o["mass_kg"], "mass_urdf_kg": o["mass_urdf_kg"], "mujoco_mass_kg": round(mj_mass, 5),
            "mujoco_over_target": round(mj_mass / o["mass_kg"], 4), "pose": env.get_active_obj_pose().round(6).tolist()}


if __name__ == "__main__":
    files = sorted(glob.glob(f"{root}/scene_cfg/*/floating/real.npy"))[shard::n_shard]
    with Pool(n_proc) as pool:
        rows = pool.map(check, files, chunksize=4)
    json.dump(rows, open(out, "w"))
    r = np.array([x["mujoco_over_target"] for x in rows])
    print(f"shard {shard}: {len(rows)} ok, mujoco/target mass min {r.min():.3f} median {np.median(r):.3f} max {r.max():.3f}")
