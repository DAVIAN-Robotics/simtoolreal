"""Progress of the full-set IK run on dj, for the page. Reads the worker logs and two directory listings.
Writes status.json and ik_reached_hist.png into ik_all/.
"""
import glob
import json
import os
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/home/nas4_user/sibeenkim/work/_graspoc"
DEL = f"{D}/dexonomy/cap16_deliver/dexonomy"
OUT = f"{D}/reset/ik_all"
PASS = 0.832  # the PoC object's Isaac pass rate under Franka's limits, used to project states
line = re.compile(r"\[w(\S+)\] (\S+) grasps (\d+) hand_ok (\d+) reached (\d+) in ([\d.]+) s")
rows, errors, workers = {}, [], {}
for f in glob.glob(f"{OUT}/logs/*.log"):
    txt = open(f).read()
    w = os.path.basename(f)[:-4]
    since = txt[txt.rfind("] ready"):] if "] ready" in txt else txt  # errors after the worker's latest start
    workers[w] = {"objects": 0, "traceback": "Traceback" in since, "last": txt.strip().splitlines()[-1][:160] if txt.strip() else ""}
    for m in line.finditer(txt):
        rows[m.group(2)] = dict(worker=m.group(1), grasps=int(m.group(3)), hand_ok=int(m.group(4)), reached=int(m.group(5)), seconds=float(m.group(6)))
        workers[w]["objects"] += 1
    if "Traceback" in since:
        errors.append((w, since[since.rindex("Traceback"):][-600:]))
delivered = sum(1 for f in os.listdir(DEL) if f.endswith(".npz") and not f.startswith("."))
done = sum(1 for f in os.listdir(OUT) if f.endswith(".npz") and not f.startswith("."))
reached = np.array([r["reached"] for r in rows.values()]) if rows else np.zeros(0)
secs = np.array([r["seconds"] for r in rows.values()]) if rows else np.zeros(0)
short = sorted((r["reached"], s) for s, r in rows.items() if r["reached"] * PASS < 10000)
st = dict(time_utc=time.strftime("%H:%M", time.gmtime()), delivered=delivered, ik_done=done, logged=len(rows),
          reached_min=int(reached.min()) if len(reached) else None, reached_median=float(np.median(reached)) if len(reached) else None,
          reached_max=int(reached.max()) if len(reached) else None, seconds_median=float(np.median(secs)) if len(secs) else None,
          projected_short=len(short), short_list=short[:20], workers=workers, errors=errors[:5])
json.dump(st, open(f"{OUT}/.status.{os.getpid()}.tmp", "w"), indent=1)
os.rename(f"{OUT}/.status.{os.getpid()}.tmp", f"{OUT}/status.json")
if len(reached):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(reached, bins=60, color="#3a7d44")
    ax.axvline(10000 / PASS, color="k", lw=1)
    ax.set_xlabel("IK-reached wrists per object, of 100,000 poses")
    ax.set_ylabel("objects")
    ax.set_title(f"{len(reached)} objects so far, black line at {10000 / PASS:,.0f}, the count that projects to 10,000 states")
    fig.savefig(f"{OUT}/.hist.{os.getpid()}.tmp.png", dpi=110, bbox_inches="tight")
    os.rename(f"{OUT}/.hist.{os.getpid()}.tmp.png", f"{OUT}/ik_reached_hist.png")
print(json.dumps({k: v for k, v in st.items() if k not in ("workers", "errors", "short_list")}))
