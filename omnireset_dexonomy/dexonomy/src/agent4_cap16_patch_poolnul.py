"""Patch the dj Dexonomy clone's hand loader to skip pool-index lines containing NUL bytes (A4-D77).

Appends from several hosts to one index file, while ~96 loaders on all hosts kept reading it, left NUL holes on NFS
(8 files, 630 to 2,361 NUL bytes each, between complete lines). np.load of such a path raises 'embedded null byte'
in a consumer thread. The workers now write one index file per writing host; this skips the damaged lines.
Usage: agent4_cap16_patch_poolnul.py <clone>
"""
import sys

p = f"{sys.argv[1]}/dexonomy/data/hand_loader.py"
s = open(p).read()
old = "                    self._pool_paths.update(buf[:end].split(\"\\n\")[:-1])\n"
new = "                    self._pool_paths.update(x for x in buf[:end].split(\"\\n\")[:-1] if \"\\x00\" not in x)\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched hand_loader.py")
