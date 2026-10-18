"""Patch the dj Dexonomy clone's gen_init.py (after A4-D79) with two NFS cuts for new init jobs (A4-D82).

1. The saver makes each distinct directory once per batch before writing, instead of os.makedirs before every one of
   the batch's up to 100 files (10 distinct object dirs). Same files, same bytes.
2. upstream's skip_done globs every object dir of every batch (about 1 s per batch on cold NFS dirs). Files of epoch e
   for this (chunk, template) can only exist if some process logged "Epoch e" to this pair's init log before writing
   them. So the job reads its own log once at start: if no earlier "Epoch e" line exists for any epoch it will run,
   every glob would find nothing and they are skipped; otherwise (an earlier try) upstream's check runs unchanged.
   The decision is logged as "Resume check on" or "Resume check off".
Usage: agent4_cap16_patch_initcuts.py <clone>
"""
import sys

p = f"{sys.argv[1]}/dexonomy/op/gen_init.py"
s = open(p).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub("        def save_batch(rows):\n"
    "            for grasp_dir, path, data in rows:\n"
    "                os.makedirs(grasp_dir, exist_ok=True)\n"
    "                np.save(path, data)\n",
    "        def save_batch(rows):\n"
    "            for grasp_dir in dict.fromkeys(r[0] for r in rows):  # each distinct dir once, not once per file\n"
    "                os.makedirs(grasp_dir, exist_ok=True)\n"
    "            for grasp_dir, path, data in rows:\n"
    "                np.save(path, data)\n")

sub("        saver, pending = ThreadPoolExecutor(max_workers=1), []\n"
    "        logging.info(\"Async saves\")\n",
    "        saver, pending = ThreadPoolExecutor(max_workers=1), []\n"
    "        logging.info(\"Async saves\")\n"
    "        # 051 cap16: see agent4_cap16_patch_initcuts.py; epoch-e files of this pair exist only after an \"Epoch e\"\n"
    "        # line in this pair's log, so without one for this job's epochs every skip_done glob finds nothing\n"
    "        main_log = os.path.join(cfg.log_dir, \"main.log\")\n"
    "        prior = open(main_log).read() if os.path.exists(main_log) else \"\"\n"
    "        resume_check = any(f\"] - Epoch {e}\\n\" in prior for e in range(epoch_start, op_cfg.epoch))\n"
    "        logging.info(f\"Resume check {'on' if resume_check else 'off'}\")\n")

sub("                if cfg.skip_done:\n"
    "                    # Check if the result already exists\n",
    "                if cfg.skip_done and resume_check:\n"
    "                    # Check if the result already exists\n")
open(p, "w").write(s)
print("patched gen_init.py")
