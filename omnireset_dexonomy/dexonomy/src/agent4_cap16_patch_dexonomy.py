"""Apply the two cap16 changes to the dj Dexonomy clone, each an exact one-place edit that fails loud if the
upstream line is not found.

1. eval_data.py: the own test sets the object's density from its scene cfg's mass_kg (clip(URDF mass, 0.05,
   0.5) kg, Sibeen's ruling) instead of the global cfg.obj_mass. Scaling density keeps the centre of mass.
2. hand_loader.py: the template-library producer re-globs new_tmpl_dir every 60 s instead of every 1 s.
   With about 150 init processes sharing one new_tmpl_dir on the nas4 NFS, a recursive glob per process per
   second would saturate the NAS. The selection logic is unchanged, only the refresh cadence.
Usage: agent4_cap16_patch_dexonomy.py <Dexonomy clone root>
"""
import sys

root = sys.argv[1]
edits = [
    (f"{root}/dexonomy/op/eval_data.py",
     '            o_cfg["density"] = cfg.obj_mass / (o_coef * np.prod(o_cfg["scale"]))\n',
     '            # 051 cap16: per-object mass clip(URDF mass, 0.05, 0.5) kg from the scene cfg, not cfg.obj_mass\n'
     '            o_cfg["density"] = o_cfg["mass_kg"] / (o_coef * np.prod(o_cfg["scale"]))\n'),
    (f"{root}/dexonomy/data/hand_loader.py",
     "            time.sleep(1.0)  # Control production rate\n",
     "            time.sleep(60.0)  # 051 cap16: 1 s upstream; 150 processes share new_tmpl_dir on NFS\n"),
]
for path, old, new in edits:
    s = open(path).read()
    assert s.count(old) == 1, (path, s.count(old))
    open(path, "w").write(s.replace(old, new))
    print("patched", path)
