"""Patch the dj Dexonomy clone's object dataloader to start its workers through a forkserver (A4-D76).

Init's hand loader runs a producer and consumer threads. The object dataloader forks its 4 workers at every epoch;
a fork while a consumer thread holds a lock leaves the child blocked on it forever (seen: a worker with one thread
in futex_wait_queue since its fork, the main process waiting on its first batch). Upstream hid this because its
recursive glob of the pool kept the consumers idle for minutes at start. Forkserver workers fork from a clean
single-threaded server. Worker seeds still come from the main torch generator, so the draws do not change
(agent4_cap16_rng_test with the context).
Usage: agent4_cap16_patch_forkserver.py <clone>
"""
import sys

p = f"{sys.argv[1]}/dexonomy/data/obj_loader.py"
s = open(p).read()
old = "        pin_memory=True,\n    )\n    return dataloader"
new = ("        pin_memory=True,\n"
       "        multiprocessing_context=\"forkserver\",  # 051 cap16: no fork while the hand loader's threads hold locks\n"
       "    )\n    return dataloader")
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched obj_loader.py")
