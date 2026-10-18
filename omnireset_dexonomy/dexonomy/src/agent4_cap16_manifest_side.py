"""Per-side manifest of a cap16 pool: for every pool.json entry, its URDF stem, key, pool mass, URDF mass,
and every visual and collision mesh with origin, scale, byte size and sha256 of the file content.
Run on the host that holds the pool (yj-4, kait or dj). Mesh paths are absolute in the URDFs; with
--remap OLD=NEW the mesh path is rewritten before it is read (dj mirror).
Usage: python agent4_cap16_manifest_side.py <pool_dir> <out.json> [--remap OLD=NEW] [--files <out_files.txt>] [--workers N]
"""
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("pool_dir")
ap.add_argument("out")
ap.add_argument("--remap", default=None)
ap.add_argument("--files", default=None)
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()
old, new = args.remap.split("=") if args.remap else ("", "")


def vec(s, n):
    v = [float(x) for x in s.split()]
    assert len(v) == n, s
    return v


def elems(link, tag):
    out = []
    for el in link.findall(tag):
        o = el.find("origin")
        m = el.find("geometry/mesh")
        assert m is not None, (tag, ET.tostring(el))
        fn = m.get("filename")
        assert os.path.isabs(fn), fn
        out.append({"mesh": fn, "xyz": vec(o.get("xyz", "0 0 0"), 3) if o is not None else [0.0, 0.0, 0.0],
                    "rpy": vec(o.get("rpy", "0 0 0"), 3) if o is not None else [0.0, 0.0, 0.0],
                    "scale": vec(m.get("scale", "1 1 1"), 3)})
    return out


def parse(e):
    urdf = e["urdf"]
    stem = os.path.basename(urdf)[:-5]
    local = os.path.join(args.pool_dir, stem + ".urdf")
    r = ET.parse(local).getroot()
    links = r.findall("link")
    assert len(links) == 1, (stem, len(links))
    link = links[0]
    mass = float(link.find("inertial/mass").get("value"))
    return {"stem": stem, "key": e["key"], "pool_mass_kg": e["mass_kg"], "mass_kg": mass,
            "collision": elems(link, "collision"), "visual": elems(link, "visual")}


def file_info(p):
    q = p.replace(old, new, 1) if old else p
    b = open(q, "rb").read()
    return p, len(b), hashlib.sha256(b).hexdigest()


pool = json.load(open(os.path.join(args.pool_dir, "pool.json")))
assert len(pool) == 3200, len(pool)
with Pool(args.workers) as pl:
    ents = pl.map(parse, pool, chunksize=20)
files = sorted({x["mesh"] for e in ents for x in e["collision"] + e["visual"]})
with Pool(args.workers) as pl:
    info = {p: (n, h) for p, n, h in pl.map(file_info, files, chunksize=20)}
for e in ents:
    for x in e["collision"] + e["visual"]:
        x["bytes"], x["sha256"] = info[x["mesh"]]
json.dump({"pool_dir": args.pool_dir, "entries": ents}, open(args.out, "w"))
if args.files:
    open(args.files, "w").write("".join(p.lstrip("/") + "\n" for p in files))
coll = {x["mesh"] for e in ents for x in e["collision"]}
vis = {x["mesh"] for e in ents for x in e["visual"]}
print(json.dumps({"entries": len(ents), "unique_files": len(files), "collision_files": len(coll), "visual_files": len(vis),
                  "bytes_total": sum(info[p][0] for p in files),
                  "unique_meshes_dirs": len({os.path.dirname(os.path.dirname(p)) for p in coll})}))
