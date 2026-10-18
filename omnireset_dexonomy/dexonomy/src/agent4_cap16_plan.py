"""Split the 3,200 prepped cap16 entries into one chunk per dj host, sized to that host's capacity, and link each
chunk's scene cfgs into cap16/chunks/<host>/scene_cfg/<stem>/floating/real.npy (the cfgs hold absolute paths).

Capacity in objects per second is min(CPU side, GPU side), both from the kaist140 one-core profile:
  CPU: refinement 0.12 s per start plus own test 0.56 s per kept grasp (about half kept) = 0.40 s per start,
       31 templates x 5 epochs x 10 starts = 1,550 starts per object -> 620 CPU-s per object.
  GPU: init 5 s per 10-object batch per template-epoch -> 77.5 process-s per object, about 1.5x throughput
       from 3 processes per 24 GB GPU -> 52 GPU-s per object, and a TITAN RTX taken as 0.6 of a 3090.
Entries are shuffled with seed 0 before slicing so every chunk gets the same mix of piece counts.
Usage: agent4_cap16_plan.py <cap16 dir> <hosts json> <out plan.json>
hosts json: [{"host", "gpus": [..], "idle_threads", "titan": bool}, ...]
"""
import glob
import json
import os
import random
import sys

cap16, hosts_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
hosts = json.load(open(hosts_path))
stems = sorted(os.path.basename(os.path.dirname(os.path.dirname(p))) for p in glob.glob(f"{cap16}/prep/scene_cfg/*/floating/real.npy"))
assert len(stems) == 3200, len(stems)
random.seed(0)
random.shuffle(stems)
for h in hosts:
    kpar = 3 * len(h["gpus"])
    cpu_left = h["idle_threads"] - kpar
    h["kpar"] = kpar
    h["nwg"] = max(4, round(0.3 * cpu_left))  # refinement workers
    h["nwe"] = max(3, round(0.7 * cpu_left / 3))  # eval n_worker, its pool is 3x this
    h["cap_obj_per_s"] = min(cpu_left / 620, len(h["gpus"]) * (0.6 if h["titan"] else 1.0) / 52)
tot = sum(h["cap_obj_per_s"] for h in hosts)
start = 0
for i, h in enumerate(hosts):
    n = round(3200 * h["cap_obj_per_s"] / tot) if i < len(hosts) - 1 else 3200 - start
    h["n_objects"] = n
    h["stems"] = stems[start:start + n]
    start += n
    d = f"{cap16}/chunks/{h['host']}/scene_cfg"
    for s in h["stems"]:
        os.makedirs(f"{d}/{s}/floating", exist_ok=True)
        os.symlink(f"{cap16}/prep/scene_cfg/{s}/floating/real.npy", f"{d}/{s}/floating/real.npy")
    h["glob"] = f"{d}/**/floating/*.npy"
assert start == 3200 and len({s for h in hosts for s in h["stems"]}) == 3200
json.dump({"hosts": hosts, "total_cap_obj_per_s": tot, "projected_min": round(3200 / tot / 60, 1)}, open(out, "w"), indent=1)
for h in hosts:
    print(h["host"], "gpus", h["gpus"], "objects", h["n_objects"], "kpar", h["kpar"], "nwg", h["nwg"], "nwe", h["nwe"], "cap", round(h["cap_obj_per_s"], 3))
print("projected minutes", round(3200 / tot / 60, 1))
