"""The one delivery of the cap16 Dexonomy run, after generation: every entry's own-test passes as <entry>.npz.

Run once every chunk's own test is done (EVAL_DONE on all 8 chunks and all 496 pairs final), as one shard per host so
the ~1.7M grasp files are read over 8 NFS clients. Each shard takes the URDF stems whose index in the sorted
manifest is its shard modulo the shard count, collects their passes from every pass list (gsched/pass, one line
per own-test pass, the grasp_data path relative to the clone), and writes <stem>.npz in the 051 common format
(hand_pos, hand_quat wxyz, q_grasp, q_squeeze, q_pregrasp, float32) plus extra/<stem>.npz (template name and source
path per row), each to a dot-prefixed temp name in the relay dir then renamed. Entries with no pass get no file.
Each shard writes its per-entry row counts to cap16/deliver_final/shard_<i>.json. The shard that finds all shard
files present writes DONE last, holding the totals. A relaunched shard keeps the entries already on disk (A4-D104).
Usage: agent4_cap16_deliver_final.py <shard> <n shards> <n proc>
"""
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy"
CLONE = f"{D}/Dexonomy"
CLONE_REAL = os.path.realpath(CLONE)
S = f"{D}/cap16/gsched"
DEL = f"{D}/cap16_deliver/dexonomy"
OUTD = f"{D}/cap16/deliver_final"
HOSTS = ["kaist67", "kaist121", "kaist140", "kaist143", "kaist144", "kaist145", "kaist148", "kaist149"]
shard, n_shard, nproc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


def savez_atomic(path, **arrays):
    tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.rename(tmp, path)


def write_entry(job):
    stem, rels = job
    main = f"{DEL}/{stem}.npz"
    # A relaunch keeps entries already on disk: the main file is renamed into place after its extra file, so its
    # presence means both are complete. Its row count must match this entry's passes.
    if os.path.exists(main):
        n = np.load(main)["hand_pos"].shape[0]
        assert n == len(rels) and os.path.exists(f"{DEL}/extra/{stem}.npz"), (stem, n, len(rels))
        return stem, n
    rows = collections.defaultdict(list)
    for rel in rels:
        f = os.path.join(CLONE, rel)
        g = np.load(f, allow_pickle=True).item()
        gq, sq, pq = g["grasp_qpos"], g["squeeze_qpos"], g["pregrasp_qpos"]
        assert gq.shape == (1, 19) and sq.shape == (1, 19) and pq.ndim == 2 and pq.shape[1] == 19, f
        assert np.allclose(sq[0, :7], gq[0, :7]), f
        assert os.path.normpath(g["scene_path"]).split(os.sep)[-3] == stem, (f, g["scene_path"])
        rows["hand_pos"].append(gq[0, :3])
        rows["hand_quat"].append(gq[0, 3:7] / np.linalg.norm(gq[0, 3:7]))
        rows["q_grasp"].append(gq[0, 7:])
        rows["q_squeeze"].append(sq[0, 7:])
        rows["q_pregrasp"].append(pq[0, 7:])
        rows["tmpl"].append(g["tmpl_name"])
        rows["src"].append(os.path.join(CLONE_REAL, rel))
    out = {k: np.asarray(rows[k], dtype=np.float32) for k in ["hand_pos", "hand_quat", "q_grasp", "q_squeeze", "q_pregrasp"]}
    for k, v in out.items():
        assert np.isfinite(v).all(), (stem, k)
    savez_atomic(f"{DEL}/extra/{stem}.npz", tmpl=np.asarray(rows["tmpl"]), src=np.asarray(rows["src"]))
    savez_atomic(f"{DEL}/{stem}.npz", **out)
    return stem, len(rels)


if __name__ == "__main__":
    for h in HOSTS:
        assert os.path.exists(f"{CLONE}/output/cap16_{h}_xhand/driver/EVAL_DONE"), h
    assert len(glob.glob(f"{S}/final/*")) == 2 * 8 * 31, len(glob.glob(f"{S}/final/*"))
    stems = sorted(x["stem"] for x in json.load(open(f"{D}/cap16/manifest.json")))
    assert len(stems) == 3200
    mine = set(stems[shard::n_shard])
    by = collections.defaultdict(set)
    for pf in glob.glob(f"{S}/pass/*.txt"):
        raw = open(pf).read()
        assert raw.endswith("\n"), pf
        for rel in raw.split("\n")[:-1]:
            stem = rel.split("/")[4]
            if stem in mine:
                by[stem].add(rel)
    os.makedirs(f"{DEL}/extra", exist_ok=True)
    os.makedirs(OUTD, exist_ok=True)
    with Pool(nproc) as pool:
        res = dict(pool.map(write_entry, [(s, sorted(r)) for s, r in sorted(by.items())], chunksize=1))
    counts = {s: res.get(s, 0) for s in sorted(mine)}
    tmp = f"{OUTD}/.shard_{shard}.json.tmp"
    json.dump(counts, open(tmp, "w"))
    os.rename(tmp, f"{OUTD}/shard_{shard}.json")
    print(f"shard {shard}: {len(res)} entries written, {sum(res.values())} grasps, {len(mine) - len(res)} entries without a pass")
    shards = sorted(glob.glob(f"{OUTD}/shard_*.json"))
    if len(shards) == n_shard and not os.path.exists(f"{DEL}/DONE"):
        allc = {}
        for f in shards:
            allc.update(json.load(open(f)))
        assert len(allc) == 3200
        n_files = len([f for f in os.listdir(DEL) if f.endswith(".npz")])
        with_pass = sum(1 for v in allc.values() if v)
        assert n_files == with_pass, (n_files, with_pass)
        open(f"{DEL}/DONE", "w").write(
            f"cap16 Dexonomy finished and delivered: {with_pass} of 3200 entries with own-test passes, "
            f"{sum(allc.values())} grasps in total, {3200 - with_pass} entries without a pass (no file)\n")
        print("DONE written")
