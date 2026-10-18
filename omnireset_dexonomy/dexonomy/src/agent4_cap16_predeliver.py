"""Gate before the one cap16 delivery, run once every pair is final.

1. Items whose upstream call raised were swallowed by upstream's safe_wrapper as WARNING tracebacks in the worker
   logs and counted done without a result (the 02:33 stale negative lookup on kaist143, A4-D80). Each such
   traceback must name its input file; the item is run again the way the worker runs it: an own-test item with
   _single_eval, and a pass is appended to its pass list (and its pool link to the host's pool index), as finish()
   does; a refinement item with _single_grasp, and a kept grasp is then own-tested the same way.
2. Pass lists hold no line twice (the indexer's crash trim, A4-D93, must have left no double queue lines).
3. For 20,000 random pass lines, the refined grasp file and its succ link exist.
Any traceback without a file, a duplicate, or a missing file raises. Writes cap16/predeliver.json.
Usage: agent4_cap16_predeliver.py
"""
import collections
import glob
import json
import os
import random
import re
import sys
import time

sys.argv = [sys.argv[0], "predeliver", "1", "1", "reuse"]
sys.path.insert(0, "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy/cap16")
import agent4_cap16_gworker as W  # noqa: E402

S, HOST = W.S, os.uname().nodename.replace("159-", "kaist")
TB = re.compile(r"\]\[WARNING\] - Traceback.*?(?=\n\[\d{4}-|\Z)", re.S)
MISSING = re.compile(r"No such file or directory: '([^']+)'")

rerun = []
for log in sorted(glob.glob(f"{S}/log/kaist*.log")):
    for tb in TB.findall(open(log).read()):
        m = MISSING.search(tb)
        assert m, (log, tb[-400:])  # a swallowed error that names no input: look at it by hand
        kind = "eval" if "_single_eval" in tb else "grasp"
        assert kind == "eval" or "_single_grasp" in tb, (log, tb[-400:])
        rerun.append((kind, m.group(1), os.path.basename(log)))


def own_test(p, c, t):
    cfg = W.CFG[("eval", c)]
    assert os.path.exists(p), p  # upstream would swallow a missing input again and report no pass
    assert W.FUNC["eval"]((p, cfg)) == p
    made = os.path.lexists(p.replace(cfg.grasp_dir, cfg.succ_grasp_dir))
    if made:
        pf = f"{S}/pass/{c}__{t}.txt"
        if p not in W.lines(pf):
            with open(pf, "a") as f:
                f.write(p + "\n")
        link = p.replace(cfg.grasp_dir, cfg.new_tmpl_dir)
        if os.path.lexists(link):
            with open(f"{S}/pool/{c}__w{HOST}__{t}.txt", "a") as f:
                f.write(link + "\n")
    return made


results = []
for kind, p, log in rerun:
    c, t = p.split("/")[1][len("cap16_"):-len("_xhand")], p.split("/")[3]
    if kind == "eval":
        results.append({"kind": kind, "item": p, "log": log, "passed": own_test(p, c, t)})
    else:
        cg = W.CFG[("grasp", c)]
        assert os.path.exists(p), p  # an npz-queued init names a path with no file: handle by hand
        assert W.FUNC["grasp"]((p, cg)) == p
        g = p.replace(cg.init_dir, cg.grasp_dir)
        kept = os.path.exists(g)
        results.append({"kind": kind, "item": p, "log": log, "kept": kept, "passed": kept and own_test(g, c, t)})

dups, n_pass, sample = {}, 0, []
rng = random.Random(0)
for pf in sorted(glob.glob(f"{S}/pass/*.txt")):
    ls = open(pf).read().split("\n")[:-1]
    d = [x for x, k in collections.Counter(ls).items() if k > 1]
    if d:
        dups[os.path.basename(pf)] = len(d)
    for x in ls:  # reservoir sample of 20,000 over all pass lines
        n_pass += 1
        if len(sample) < 20000:
            sample.append(x)
        elif rng.randrange(n_pass) < 20000:
            sample[rng.randrange(20000)] = x
assert not dups, dups
missing = []
for p in sample:
    c = p.split("/")[1][len("cap16_"):-len("_xhand")]
    cfg = W.CFG[("eval", c)]
    if not (os.path.exists(p) and os.path.lexists(p.replace(cfg.grasp_dir, cfg.succ_grasp_dir))):
        missing.append(p)
assert not missing, missing[:5]

out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rerun": results, "pass_lines": n_pass,
       "pass_duplicates": 0, "sampled_pass_lines_checked": len(sample)}
json.dump(out, open(f"{W.D}/cap16/predeliver.json", "w"), indent=1)
print(json.dumps(out))
