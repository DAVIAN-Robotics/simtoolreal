"""Shared refinement and own-test worker for the cap16 run, one per host, replacing every chunk's own grasp and
eval loops so any host with free CPU works on any chunk. It never lists a directory.

Work is leased per (kind, chunk, template) with mkdir locks under cap16/gsched/lock. Each pair has a queue,
gsched/todo/<kind>__<chunk>__<tmpl>.txt, and a done list, gsched/done/<kind>__<chunk>__<tmpl>.txt. The grasp
queue is written by agent4_cap16_indexer from the init logs. The eval queue is written by the grasp leases of the
same pair, one line per refined grasp whose file has been fully written, so the own test never reads a file that
is still being written. A lease takes up to LEASE queued items not yet done and runs upstream's _single_grasp or
_single_eval on them in this host's pool, grouped by object. With <cache> reuse, agent4_cap16_envcache (proven
bit-identical by agent4_cap16_envcache_test) lets one compiled model serve every start of an object, refinement
and own test.
The done list is appended after the pool returns, as upstream appends its path_log. Own-test passes go to
gsched/pass/<chunk>__<tmpl>.txt. Each object group is its own pool task, and leases are added while fewer than two
groups per pool process are outstanding, so the pool never drains behind one long group. Finished leases are closed
in any order. Own-test pairs are offered before refinement
pairs, so threads go to the slower stage while it has work. Pairs whose inputs this host wrote come before other
hosts' pairs (gsched/aff, A4-D81).
gsched/seen/<pair>@<size> holds the queue size up to which everything is done, so a pair is read again only when its
queue has grown. A grasp pair is final when the indexer has marked its init final (gsched/indexed) and nothing
queued is left. An eval pair is final when its grasp pair is final and nothing queued is left. When all 31 pairs of
a chunk are final, the chunk's GRASP_DONE or EVAL_DONE marker is written. Each finished lease appends one counter
line to gsched/count/<host>.txt: time, kind, chunk, template, items, outputs (grasps kept, or own-test passes).
Usage: agent4_cap16_gworker.py <host> <n proc> <lease items> <cache: reuse | rebuild>
"""
import glob
import logging
import os
import random
import shutil
import socket
import sys
import time
from multiprocessing import Pool

D = "/home/nas4_user/sibeenkim/work/_graspoc/dexonomy"
CLONE = f"{D}/Dexonomy"
POOL = f"{D}/cap16/new_tmpl_pool"
S = f"{D}/cap16/gsched"
CHUNKS = ["kaist67", "kaist121", "kaist140", "kaist143", "kaist144", "kaist145", "kaist148", "kaist149"]
HOST, NPROC, LEASE, CACHE = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert CACHE in ("reuse", "rebuild"), CACHE
os.chdir(CLONE)
sys.path.insert(0, CLONE)
logging.basicConfig(filename=f"{S}/log/{HOST}.log", level=logging.INFO,
                    format="[%(asctime)s][%(process)d][%(levelname)s] - %(message)s")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from dexonomy.op.eval_data import _single_eval  # noqa: E402
from dexonomy.op.gen_grasp import _single_grasp  # noqa: E402

# same seeding as dexonomy/main.py
np.random.seed(12)
torch.manual_seed(12)
random.seed(12)

TMPLS = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{CLONE}/assets/hand/xhand/init_tmpl/*.npy"))
assert len(TMPLS) == 31, TMPLS
FUNC = {"grasp": _single_grasp, "eval": _single_eval}


def job_cfg(kind, chunk):
    """The resolved job config of `python -m dexonomy.main op=<kind> ...` as the chunk driver called it."""
    ov = ["hand=xhand", f"exp_name=cap16_{chunk}", f"new_tmpl_dir={POOL}", f"n_worker={NPROC}", f"op={kind}"]
    if kind == "eval":
        ov += ["log_id=0", "skip_traj=True"]
    with initialize_config_dir(config_dir=f"{CLONE}/dexonomy/config", version_base=None):
        cfg = compose("base", overrides=ov, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            del cfg["hydra"]  # the ${hydra:...} resolver reads the singleton set above
        c = OmegaConf.to_container(cfg, resolve=True)
    return OmegaConf.create(c)


CFG = {(k, c): job_cfg(k, c) for k in FUNC for c in CHUNKS}


def lines(p):
    return set(open(p).read().split("\n")) - {""} if os.path.exists(p) else set()


def listing(root, tmpl):
    """Same strings as upstream's glob(os.path.join(root, '**/*.npy')) restricted to one template. Tests only."""
    out = []
    base = os.path.join(root, tmpl)
    if not os.path.isdir(base):
        return out
    for stem in os.listdir(base):
        d = os.path.join(base, stem, "floating", "real")
        if os.path.isdir(d):  # makedirs creates the levels one by one, real/ can lag its stem dir
            out += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy")]
    return out


def pending(kind, c, t, q_off, d_off):
    """Queued items not yet done, in queue order, and the queue size read (whole lines only). Everything before the
    queue offset q_off was done when the done list was d_off bytes long, so only the queue from q_off and the done
    list from d_off are read; a later item cannot have been done before it was queued."""
    q = f"{S}/todo/{kind}__{c}__{t}.txt"
    if not os.path.exists(q):
        return [], 0
    with open(q) as f:
        f.seek(q_off)
        raw = f.read()
    end = raw.rfind("\n") + 1
    df = f"{S}/done/{kind}__{c}__{t}.txt"
    done = set()
    if os.path.exists(df):
        with open(df) as f:
            f.seek(d_off)
            done = set(f.read().split("\n")) - {""}
    out, seen = [], set()
    for p in raw[:end].split("\n")[:-1]:
        if p not in done and p not in seen:
            out.append(p)
            seen.add(p)
    return out, q_off + end


def final_ready(kind, c, t):
    return os.path.exists(f"{S}/indexed/grasp__{c}__{t}" if kind == "grasp" else f"{S}/final/grasp__{c}__{t}")


def scan_state():
    """Every pair's state from four directory listings, not a stat per pair: final markers, indexer markers, seen
    sizes (held in the file name <pair>@<size>) and queue sizes."""
    final = set(os.listdir(f"{S}/final"))
    indexed = set(os.listdir(f"{S}/indexed"))
    seen = {}  # pair -> (queue offset, done-list offset); files written before offsets existed read as (size, 0)
    for n in os.listdir(f"{S}/seen"):
        if "@" in n:
            pair, _, v = n.rpartition("@")
            q_off, _, d_off = v.partition("_")
            cur = (int(q_off), int(d_off) if d_off else 0)
            if cur > seen.get(pair, (-1, 0)):
                seen[pair] = cur
    sizes = {e.name[:-4]: e.stat().st_size for e in os.scandir(f"{S}/todo") if e.name.endswith(".txt")}
    aff = dict(n.rsplit("@", 1) for n in os.listdir(f"{S}/aff"))
    return final, indexed, seen, sizes, aff


def set_aff(pair, host):
    """Mark a pair with the host that writes its inputs (gsched/aff/<pair>@<host>); only its lease holder writes."""
    new = f"{S}/aff/{pair}@{host}"
    if os.path.exists(new):
        return
    open(new, "w").close()
    for f in glob.glob(f"{S}/aff/{pair}@*"):
        if f != new:
            os.remove(f)


def write_seen(kind, c, t, size, d_size):
    """The queue size up to which the pair's queue is done and the done list's size at that point, as an empty file
    <pair>@<size>_<d_size>, so one listing reads them all. Only the pair's lock holder writes it; older values are
    removed after the new one exists."""
    pair = f"{kind}__{c}__{t}"
    new = f"{S}/seen/{pair}@{size}_{d_size}"
    open(new, "w").close()
    for f in glob.glob(f"{S}/seen/{pair}@*"):
        if f != new:
            os.remove(f)


def mark_final(kind, c, t):
    open(f"{S}/final/{kind}__{c}__{t}", "w").close()
    if all(os.path.exists(f"{S}/final/{kind}__{c}__{x}") for x in TMPLS):
        open(f"{CLONE}/output/cap16_{c}_xhand/driver/{'GRASP_DONE' if kind == 'grasp' else 'EVAL_DONE'}", "w").close()
        logging.info(f"{kind} of chunk {c} final, marker written")


_NP_LOAD = np.load
_NPZ = {}  # the running group's init grasps queued from a batch npz: path upstream's file would have -> (npz, member)


def _np_load(file, *a, **k):
    """np.load in the pool (A4-D92). An init grasp that init saved inside its batch's npz is queued as the path its
    own file would have, a tab, and the npz. The member holds the bytes that file would hold, so upstream's
    _single_grasp loads the same dict under the same path and writes the same outputs."""
    if isinstance(file, str) and file in _NPZ:
        npz, key = _NPZ[file]
        return npz[key]
    return _NP_LOAD(file, *a, **k)


def run_group(args):
    """One object's items of a lease, in one pool process. Returns (queue line, output written, pool link) per item:
    for an own-test pass, the template-pool link upstream's eval made or found (tmpl_upd_mode hybrid, n_evo <= 2)."""
    kind, lines, cfg = args
    items = [x.partition("\t")[::2] for x in lines]  # (path, npz or "")
    # The inputs were written by another host. A stale negative lookup cached on this host made one own test raise
    # FileNotFoundError (swallowed by upstream's safe_wrapper as a WARNING); listing the input's directory reads it
    # from the server first. A group's files sit in its object's directory, its npz members in their batches' dirs.
    for d in dict.fromkeys(os.path.dirname(n or p) for p, n in items):
        os.listdir(d)
    opened = {n: _NP_LOAD(n, allow_pickle=True) for n in dict.fromkeys(n for _, n in items if n)}
    for p, n in items:
        if n:
            assert p.startswith(cfg.init_dir + "/") and p.endswith(".npy"), p
            key = p[len(cfg.init_dir) + 1:].split("/", 1)[1][:-len(".npy")]  # under <init_dir>/<tmpl>, as init named it
            assert key in opened[n].files, (p, n)  # here, not inside upstream's safe_wrapper, which would swallow it
            _NPZ[p] = (opened[n], key)
    out = []
    for line, (p, _) in zip(lines, items):
        r = FUNC[kind]((p, cfg))
        assert r == p, (r, p)
        if kind == "grasp":
            out.append((line, os.path.exists(p.replace(cfg.init_dir, cfg.grasp_dir)), None))
        else:
            made = os.path.lexists(p.replace(cfg.grasp_dir, cfg.succ_grasp_dir))
            link = p.replace(cfg.grasp_dir, cfg.new_tmpl_dir)
            out.append((line, made, link if made and os.path.lexists(link) else None))
    _NPZ.clear()
    for f in opened.values():
        f.close()
    return out


def groups(kind, c, todo):
    by = {}
    for p in todo:
        by.setdefault(p.split("/")[4], []).append(p)
    return [(kind, ps, CFG[(kind, c)]) for ps in by.values()]


def find_lease(rng):
    got = find_leases(rng, 1)
    return got[0] if got else None


def find_leases(rng, want):
    """Up to want leases from one scan of the pair state, so a pool far below its target fills without a scan per
    lease. Own-test pairs first: it costs about 1.3 s per grasp against 0.1 s per refinement start."""
    final, indexed, seen, sizes, aff = scan_state()
    out = []
    # Pairs whose inputs this host wrote come first (A4-D81): reading a file another host just created makes the NAS
    # recall that host's delegation, which stalls both. Other hosts' pairs are taken when there is no local work.
    pairs = []
    for local in (True, False):
        for k in ["eval", "grasp"]:
            ps = [(k, c, t) for c in CHUNKS for t in TMPLS
                  if f"{k}__{c}__{t}" not in final and (aff.get(f"{k}__{c}__{t}") == HOST) == local]
            rng.shuffle(ps)
            pairs += ps
    for k, c, t in pairs:
        ready_marker = f"grasp__{c}__{t}" in (indexed if k == "grasp" else final)
        q_off, d_off = seen.get(f"{k}__{c}__{t}", (0, 0))
        if sizes.get(f"{k}__{c}__{t}", 0) == q_off and f"{k}__{c}__{t}" in seen and not ready_marker:
            continue
        lk = f"{S}/lock/{k}__{c}__{t}"
        try:
            os.mkdir(lk)  # the lock: exactly one host holds a pair
        except FileExistsError:
            continue
        open(f"{lk}/owner", "w").write(f"{HOST} {os.getpid()}\n")
        ready = final_ready(k, c, t)  # before reading the queue, whose last lines precede the marker
        todo, size = pending(k, c, t, q_off, d_off)
        if todo:
            out.append((lk, k, c, t, todo[:LEASE], size, len(todo) <= LEASE))
            if len(out) >= want:
                return out
            continue
        df = f"{S}/done/{k}__{c}__{t}.txt"
        write_seen(k, c, t, size, os.path.getsize(df) if os.path.exists(df) else 0)
        if ready:
            mark_final(k, c, t)
        shutil.rmtree(lk)
    return out


def submit(pool, lease):
    """One apply_async per object group, so the main loop sees how many groups are still outstanding."""
    g = groups(lease[1], lease[2], lease[4])
    L = {"meta": lease, "t0": time.time(), "left": len(g), "out": [], "err": []}

    def done(r):
        L["out"].extend(r)
        L["left"] -= 1

    for grp in g:
        pool.apply_async(run_group, (grp,), callback=done, error_callback=L["err"].append)
    return L


def finish(L):
    (lk, k, c, t, todo, size, exhausted), t0, out = L["meta"], L["t0"], L["out"]
    assert sorted(r for r, _, _ in out) == sorted(todo), (k, c, t)
    made = [r for r, m, _ in out if m]
    pooled = [x for _, _, x in out if x]
    if k == "grasp" and made:
        cfg = CFG[("grasp", c)]
        with open(f"{S}/todo/eval__{c}__{t}.txt", "a") as f:
            f.write("\n".join(p.partition("\t")[0].replace(cfg.init_dir, cfg.grasp_dir) for p in made) + "\n")
        set_aff(f"eval__{c}__{t}", HOST)
    if k == "eval" and made:
        with open(f"{S}/pass/{c}__{t}.txt", "a") as f:
            f.write("\n".join(made) + "\n")
    if pooled:  # the hand loader reads these instead of globbing the pool. One file per writing host: appends from
        # several hosts to one file that ~96 loaders keep reading left NUL holes on NFS (A4-D77)
        with open(f"{S}/pool/{c}__w{HOST}__{t}.txt", "a") as f:
            f.write("\n".join(pooled) + "\n")
    df = f"{S}/done/{k}__{c}__{t}.txt"
    with open(df, "a") as f:
        f.write("\n".join(r for r, _, _ in out) + "\n")
    if exhausted:
        write_seen(k, c, t, size, os.path.getsize(df))
    shutil.rmtree(lk)
    with open(f"{S}/count/{HOST}.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {k} {c} {t} {len(todo)} {len(made)}\n")
    logging.info(f"{k} {c} {t} {len(todo)} items, {len(made)} outputs, {time.time() - t0:.1f} s in flight")


def all_final():
    return len(set(os.listdir(f"{S}/final"))) == 2 * len(CHUNKS) * len(TMPLS)


if __name__ == "__main__":
    # a restarted worker drops the locks its previous run on this host held; a lock another host is taking right
    # now has no owner line yet, and cannot be this host's since this worker is not running yet
    for d in glob.glob(f"{S}/lock/*"):
        o = open(f"{d}/owner").read().split() if os.path.exists(f"{d}/owner") else []
        if o and o[0] == HOST:
            shutil.rmtree(d)
    if CACHE == "reuse":
        import agent4_cap16_envcache
        agent4_cap16_envcache.install()
    np.load = _np_load  # before the pool forks, so every pool process reads npz-queued inits (A4-D92)
    logging.info(f"worker start host={socket.gethostname()} pid={os.getpid()} n_proc={NPROC} lease={LEASE} cache={CACHE}")
    rng = random.Random(HOST)
    inflight = []
    next_scan = 0.0
    with Pool(NPROC) as pool:
        while True:
            stop = os.path.exists(f"{S}/stop/{HOST}")
            for L in inflight:
                if L["err"]:
                    raise L["err"][0]
            for L in [L for L in inflight if L["left"] == 0]:
                finish(L)
                inflight.remove(L)
            # keep about two outstanding object groups per pool process; one long group must not hold the rest
            outstanding = sum(L["left"] for L in inflight)
            if not stop and outstanding < 2 * NPROC and time.time() >= next_scan:
                # several leases per round when the pool is far below its target (A4-D83): one lease per 2 s left
                # 130 of kaist140's 180 processes idle once affinity made most hosts' work local
                got = find_leases(rng, max(1, min(8, (2 * NPROC - outstanding) // 20)))
                for nxt in got:
                    inflight.append(submit(pool, nxt))
                next_scan = time.time() + (1 if got else 10)  # nothing to lease, do not rescan for 10 s
                if got:
                    continue
            if inflight:
                time.sleep(0.5)
            elif stop:
                logging.info("stop file found, exiting")
                break
            elif all_final():
                logging.info("every pair final, exiting")
                break
            else:
                time.sleep(10)
