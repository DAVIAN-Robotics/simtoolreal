"""Patch the dj Dexonomy clone's gen_init.py (after A4-D79 and A4-D82) so each batch's init grasps are one npz (A4-D92).

Upstream writes each valid grasp of a batch as its own <init_dir>/<tmpl>/<scene_id>/<epoch>_<k>_grasp.npy, up to
100 files per batch. Here the saver writes the batch as <init_dir>/<tmpl>/_batch/<epoch>/<j>.npz, j the batch's
index in the epoch (every batch counts, as the indexer counts its 'Get' and 'Skip' lines). Each member is named by
the file's path under <init_dir>/<tmpl> without .npy, and np.savez writes each member with the same np.save call on
the same dict, so every member's bytes equal the file upstream would write. The npz is written under a dot-prefixed
temp name and renamed, and the saver logs 'Saved N npz <path under init_dir>'. A resumed job skips a batch whose npz
exists, as well as one upstream's glob finds files for (a try before this patch wrote files).
Usage: agent4_cap16_patch_npzbatch.py <clone>
"""
import os
import sys

p = f"{sys.argv[1]}/dexonomy/op/gen_init.py"
s = open(p).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub("        def save_batch(rows):\n"
    "            for grasp_dir in dict.fromkeys(r[0] for r in rows):  # each distinct dir once, not once per file\n"
    "                os.makedirs(grasp_dir, exist_ok=True)\n"
    "            for grasp_dir, path, data in rows:\n"
    "                np.save(path, data)\n"
    "            logging.info(f\"Saved {len(rows)}\")\n",
    "        # 051 cap16: one npz per batch, each member the bytes np.save writes; see agent4_cap16_patch_npzbatch.py\n"
    "        tmpl_base = os.path.join(cfg.init_dir, tmpl_name) + os.sep\n"
    "\n"
    "        def batch_npz(e, j):\n"
    "            return os.path.join(cfg.init_dir, tmpl_name, \"_batch\", str(e), f\"{j}.npz\")\n"
    "\n"
    "        def save_batch(rows, npz):\n"
    "            members = {}\n"
    "            for grasp_dir, path, data in rows:\n"
    "                assert path.startswith(tmpl_base) and path.endswith(\".npy\"), path\n"
    "                members[path[len(tmpl_base):-len(\".npy\")]] = data\n"
    "            os.makedirs(os.path.dirname(npz), exist_ok=True)\n"
    "            tmp = os.path.join(os.path.dirname(npz), f\".{os.path.basename(npz)}.tmp\")\n"
    "            with open(tmp, \"wb\") as f:\n"
    "                np.savez(f, **members)\n"
    "            os.replace(tmp, npz)\n"
    "            logging.info(f\"Saved {len(rows)} npz {os.path.relpath(npz, cfg.init_dir)}\")\n")

sub("            for obj_sample in obj_loader:\n"
    "                if cfg.skip_done and resume_check:\n"
    "                    # Check if the result already exists\n"
    "                    continue_flag = False\n",
    "            for bj, obj_sample in enumerate(obj_loader):\n"
    "                if cfg.skip_done and resume_check:\n"
    "                    # Check if the result already exists\n"
    "                    continue_flag = os.path.exists(batch_npz(eee, bj))  # 051 cap16: saved as one npz\n")

sub("                pending.append(saver.submit(save_batch, rows))\n",
    "                pending.append(saver.submit(save_batch, rows, batch_npz(eee, bj)))\n")
tmp = f"{p}.npzbatch.tmp"  # jobs start at any time; they must import the old file or the new one, never half
open(tmp, "w").write(s)
os.replace(tmp, p)
print("patched gen_init.py")
