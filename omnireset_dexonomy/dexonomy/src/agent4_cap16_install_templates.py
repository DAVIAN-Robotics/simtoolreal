"""Install astra's approved 31 XHand templates into the dj Dexonomy clone, with sha256 checks.

1. Every package file listed in the artifacts json matches its sha256 in the extracted package.
2. The clone's hand model files (right.xml, keypoint/skeleton/body_group yaml, hulls/*, meshes/*) are
   compared byte for byte with the package's. Differences are reported and nothing is overwritten.
3. The current raw_anno/ and init_tmpl/ are copied to a backup dir, which must not exist yet.
4. raw_anno/ and init_tmpl/ are replaced by the package's 31, then every installed file is re-verified.
Usage: python3 agent4_cap16_install_templates.py <artifacts.json> <extracted astra_rules_corrected_xhand_port dir> <clone xhand dir> <backup dir>
"""
import hashlib
import json
import os
import shutil
import sys

art_path, pkg, xhand, backup = sys.argv[1:5]
art = json.load(open(art_path))
PREFIX = "src/astra_rules_corrected_xhand_port/"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


entries = {k[len(PREFIX):]: v["sha256"] for k, v in art.items() if k.startswith(PREFIX)}
bad = [rel for rel, h in entries.items() if sha(os.path.join(pkg, rel)) != h]
assert not bad, ("package files not matching artifacts json", bad)
print(f"package: {len(entries)} files match artifacts json sha256")

model = ["right.xml", "keypoint.yaml", "skeleton.yaml", "body_group.yaml"] + \
        [f"hulls/{f}" for f in sorted(os.listdir(os.path.join(pkg, "hulls")))] + \
        [f"meshes/{f}" for f in sorted(os.listdir(os.path.join(pkg, "meshes")))]
differ = [m for m in model if sha(os.path.join(pkg, m)) != sha(os.path.join(xhand, m))]
print(f"hand model files compared: {len(model)}, differing: {differ}")
assert not differ, "a clone model file differs from the package; not overwriting"

assert not os.path.exists(backup), backup
os.makedirs(backup)
for sub in ["raw_anno", "init_tmpl"]:
    shutil.copytree(os.path.join(xhand, sub), os.path.join(backup, sub))
print(f"backup: raw_anno {len(os.listdir(os.path.join(backup, 'raw_anno')))}, init_tmpl {len(os.listdir(os.path.join(backup, 'init_tmpl')))} -> {backup}")

for sub in ["raw_anno", "init_tmpl"]:
    shutil.rmtree(os.path.join(xhand, sub))
    shutil.copytree(os.path.join(pkg, sub), os.path.join(xhand, sub))

installed = {}
for sub in ["raw_anno", "init_tmpl"]:
    for f in sorted(os.listdir(os.path.join(xhand, sub))):
        rel = f"{sub}/{f}"
        assert rel in entries, ("installed file not in artifacts json", rel)
        installed[rel] = sha(os.path.join(xhand, rel)) == entries[rel]
n_raw = sum(r.startswith("raw_anno/") for r in installed)
n_init = sum(r.startswith("init_tmpl/") for r in installed)
assert all(installed.values()), [r for r, ok in installed.items() if not ok]
assert n_raw == 31 and n_init == 31, (n_raw, n_init)
names = sorted(f.split(".")[0] for f in os.listdir(os.path.join(xhand, "init_tmpl")))
assert not any(n.split("_")[0] in ("19", "21") for n in names), names
print(f"installed: raw_anno {n_raw}, init_tmpl {n_init}, all sha256 match; types 19 and 21 absent")
print("templates:", " ".join(sorted(names, key=lambda n: int(n.split("_")[0]))))
