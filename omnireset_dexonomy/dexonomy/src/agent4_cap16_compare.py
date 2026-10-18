"""Compare two cap16 pool manifests (agent4_cap16_manifest_side.py output), key by key.

Checks per pool.json key: URDF stem, URDF mass and pool.json mass, and for collision and visual in order:
file content sha256, byte size, origin xyz and rpy, mesh scale, and the mesh path below meshdatav3_scaled/.
Prints counts of every kind of difference and up to 10 examples each. Exit status 1 if any collision or
mass difference exists.
Usage: python agent4_cap16_compare.py <a.json> <b.json>
"""
import collections
import json
import sys

a = {e["key"]: e for e in json.load(open(sys.argv[1]))["entries"]}
b = {e["key"]: e for e in json.load(open(sys.argv[2]))["entries"]}
diff = collections.defaultdict(list)
if set(a) != set(b):
    diff["key_set"].append((sorted(set(a) - set(b))[:10], sorted(set(b) - set(a))[:10]))


def tail(p):
    return p.split("meshdatav3_scaled/", 1)[1] if "meshdatav3_scaled/" in p else p


for k in sorted(set(a) & set(b)):
    x, y = a[k], b[k]
    if x["stem"] != y["stem"]:
        diff["stem"].append((k, x["stem"], y["stem"]))
    for f in ["mass_kg", "pool_mass_kg"]:
        if x[f] != y[f]:
            diff[f].append((k, x[f], y[f]))
    if x["mass_kg"] != x["pool_mass_kg"]:
        diff["a_urdf_vs_pool_mass"].append((k, x["mass_kg"], x["pool_mass_kg"]))
    for kind in ["collision", "visual"]:
        if len(x[kind]) != len(y[kind]):
            diff[f"{kind}_count"].append((k, len(x[kind]), len(y[kind])))
            continue
        for i, (p, q) in enumerate(zip(x[kind], y[kind])):
            for f in ["sha256", "bytes", "xyz", "rpy", "scale"]:
                if p[f] != q[f]:
                    diff[f"{kind}_{f}"].append((k, i, p[f], q[f]))
            if tail(p["mesh"]) != tail(q["mesh"]):
                diff[f"{kind}_relpath"].append((k, i, tail(p["mesh"]), tail(q["mesh"])))
print(json.dumps({"keys_a": len(a), "keys_b": len(b), "common": len(set(a) & set(b)),
                  "difference_counts": {k: len(v) for k, v in diff.items()}}))
for k, v in diff.items():
    print(k, v[:10])
bad = [k for k in diff if k.startswith("collision") or "mass" in k or k in ("key_set", "stem")]
sys.exit(1 if bad else 0)
