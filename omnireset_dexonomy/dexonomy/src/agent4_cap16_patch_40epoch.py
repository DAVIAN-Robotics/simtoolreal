"""Patch the dj Dexonomy clone for the 40-epoch continuation of cap16 (A4-D72 to A4-D74).

gen_init.py: a job runs epochs [op.epoch_start, op.epoch) instead of walking the done epochs, and reseeds numpy,
torch and random with 12 + epoch_start after the seed-12 object order is fixed, so a fresh process does not redraw
epoch 0's start poses (agent4_cap16_rng_test: two seed-12 processes draw identical poses).
hand_loader.py: the pool's template paths come from the own-test workers' per-(chunk, template) index files in
cap16/gsched/pool, read incrementally, instead of a recursive glob of new_tmpl_dir/<tmpl> on NFS every 60 s. Same
paths, same strings (agent4_cap16_pool_index checks the index against the glob).
Usage: agent4_cap16_patch_40epoch.py <clone>
"""
import sys

clone = sys.argv[1]


def patch(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old[:60])
    open(path, "w").write(s.replace(old, new))


patch(f"{clone}/dexonomy/op/gen_init.py",
      "        for eee in range(op_cfg.epoch):\n",
      "        # 051 cap16: a job runs epochs [epoch_start, epoch). A fresh process restarts at seed 12, and the loader's\n"
      "        # worker seeds come from the main torch generator, so it would redraw epoch 0's start poses; reseed with\n"
      "        # 12 + epoch_start here, after the seed-12 object order above is fixed.\n"
      "        epoch_start = op_cfg.get(\"epoch_start\", 0)\n"
      "        if epoch_start:\n"
      "            np.random.seed(12 + epoch_start)\n"
      "            torch.manual_seed(12 + epoch_start)\n"
      "            random.seed(12 + epoch_start)\n"
      "        for eee in range(epoch_start, op_cfg.epoch):\n")

patch(f"{clone}/dexonomy/data/hand_loader.py",
      "        self.new_tmpl_dir = new_tmpl_dir\n",
      "        self.new_tmpl_dir = new_tmpl_dir\n"
      "        # 051 cap16: pool paths read incrementally from the own-test workers' index files, see _update_data_path\n"
      "        self._pool_paths, self._pool_off = set(), {}\n")

patch(f"{clone}/dexonomy/data/hand_loader.py",
      "                template_lst = glob(\n"
      "                    os.path.join(self.new_tmpl_dir, self.tmpl_name, \"**/*.npy\"),\n"
      "                    recursive=True,\n"
      "                )\n",
      "                # 051 cap16: the paths the recursive glob of new_tmpl_dir/<tmpl> would return, from the index files\n"
      "                # the own-test workers append per (chunk, template), instead of listing the pool on NFS every 60 s\n"
      "                index_dir = os.path.join(os.path.dirname(self.new_tmpl_dir), \"gsched\", \"pool\")\n"
      "                for f in glob(os.path.join(index_dir, f\"*__{self.tmpl_name}.txt\")):\n"
      "                    off = self._pool_off.get(f, 0)\n"
      "                    with open(f) as fh:\n"
      "                        fh.seek(off)\n"
      "                        buf = fh.read()\n"
      "                    end = buf.rfind(\"\\n\") + 1\n"
      "                    self._pool_paths.update(buf[:end].split(\"\\n\")[:-1])\n"
      "                    self._pool_off[f] = off + end\n"
      "                template_lst = list(self._pool_paths)\n")
print("patched gen_init.py and hand_loader.py")
