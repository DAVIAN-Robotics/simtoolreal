"""Patch the dj Dexonomy clone's gen_init.py so a batch's init files are written by one background thread (A4-D79).

Upstream writes up to 100 small npy files per batch to NFS in the main thread, between GPU phases. Here the main
thread builds each file's dict with upstream's own expressions (same data, same names), hands the batch to a single
saver thread and goes on to the next batch. At most 2 batches wait for saving; a save error is raised in the main
thread; all saves finish before 'Finish initialization.' is logged. The process logs 'Async saves' once before its
epoch loop and 'Saved N' after each batch's N files are written, so the indexer knows when a batch is on disk
(it can no longer take the next main-flow line as that sign).
Usage: agent4_cap16_patch_asyncsave.py <clone>
"""
import sys

p = f"{sys.argv[1]}/dexonomy/op/gen_init.py"
s = open(p).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub("        for eee in range(epoch_start, op_cfg.epoch):\n",
    "        # 051 cap16: saves on one background thread; see agent4_cap16_patch_asyncsave.py\n"
    "        from concurrent.futures import ThreadPoolExecutor\n"
    "\n"
    "        def save_batch(rows):\n"
    "            for grasp_dir, path, data in rows:\n"
    "                os.makedirs(grasp_dir, exist_ok=True)\n"
    "                np.save(path, data)\n"
    "            logging.info(f\"Saved {len(rows)}\")\n"
    "\n"
    "        saver, pending = ThreadPoolExecutor(max_workers=1), []\n"
    "        logging.info(\"Async saves\")\n"
    "        for eee in range(epoch_start, op_cfg.epoch):\n")

sub("                for k, v in ret_dict.items():\n"
    "                    ret_dict[k] = v.cpu().numpy()\n"
    "\n"
    "                for count, o_id in enumerate(ret_dict[\"obj_id\"]):\n",
    "                for k, v in ret_dict.items():\n"
    "                    ret_dict[k] = v.cpu().numpy()\n"
    "\n"
    "                rows = []\n"
    "                for count, o_id in enumerate(ret_dict[\"obj_id\"]):\n")

sub("                    os.makedirs(grasp_dir, exist_ok=True)\n"
    "                    np.save(\n"
    "                        f\"{grasp_dir}/{eee}_{count}_grasp.npy\",\n"
    "                        {\n",
    "                    rows.append((\n"
    "                        grasp_dir,\n"
    "                        f\"{grasp_dir}/{eee}_{count}_grasp.npy\",\n"
    "                        {\n")

sub("                            \"obj_scale\": ret_dict[\"obj_scale\"][count],\n"
    "                        },\n"
    "                    )\n"
    "        logging.info(\"Finish initialization.\")\n",
    "                            \"obj_scale\": ret_dict[\"obj_scale\"][count],\n"
    "                        },\n"
    "                    ))\n"
    "                while pending and (pending[0].done() or len(pending) >= 2):\n"
    "                    pending.pop(0).result()  # raises a save error here\n"
    "                pending.append(saver.submit(save_batch, rows))\n"
    "        for f in pending:\n"
    "            f.result()\n"
    "        saver.shutdown(wait=True)\n"
    "        logging.info(\"Finish initialization.\")\n")
open(p, "w").write(s)
print("patched gen_init.py")
