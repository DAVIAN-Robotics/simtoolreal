"""Reuse one compiled MuJoCo model per object inside a worker process instead of rebuilding it for every start.

install() swaps the MuJoCo_OptEnv name that upstream's _single_grasp looks up, and the MuJoCo_EvalEnv name that
_single_eval looks up, for factories with the same signature. The first call with a given set of constructor
arguments builds the env the upstream way and keeps it untouched as the pristine copy. Every call then returns a
new env object whose model is a full copy of the pristine model (mj_copyModel), whose sim cfg is a deep copy, and
whose data is a new MjData put through the constructor's own reset. So each start begins from the model and cfg
a fresh build gives, and the upstream flow's own reset_qpos then resets the data from the keyframe as before.
Only the model is shared, never mutated. The key is the pickled constructor arguments, so any difference in hand,
scene, mass or margins builds a new env. One pristine env is kept per process, since tasks are grouped by object.
"""
import collections
import copy
import os
import pickle

import mujoco

import dexonomy.op.eval_data as E
import dexonomy.op.gen_grasp as G
import dexonomy.sim.mujoco_env as M

_UPSTREAM = {"opt": G.MuJoCo_OptEnv, "eval": E.MuJoCo_EvalEnv}
_PRISTINE = {}
STATS = {"build": 0, "reuse": 0}
_SPEC = {"from_file": mujoco.MjSpec.from_file, "compile": mujoco.MjSpec.compile}
_XML, _HAND_BYTES = {}, {}
_OBJ_BYTES = collections.OrderedDict()  # object mesh path -> bytes, least recently used first (A4-D97)
_OBJ_CAP = 256 << 20
_OBJ_SIZE = [0]
# upstream's meshdir for every env (MuJoCo_BaseEnv.__init__), under which the hand's meshes resolve
_HAND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(M.__file__))), "assets", "hand") + os.sep


def _factory(kind):
    cls = _UPSTREAM[kind]

    def make(*args, **kwargs):
        key = (kind, pickle.dumps((args, sorted(kwargs.items()))))
        if key not in _PRISTINE:
            for k in [k for k in _PRISTINE if k[0] == kind]:
                del _PRISTINE[k]
            _PRISTINE[key] = cls(*args, **kwargs)
            STATS["build"] += 1
        else:
            STATS["reuse"] += 1
        src = _PRISTINE[key]
        env = copy.copy(src)
        env._cfg = copy.deepcopy(src._cfg)
        env._model = copy.deepcopy(src._model)
        env._data = mujoco.MjData(env._model)
        env.reset_qpos(env.get_hand_qpos(), set_ctrl=False)
        return env

    return make


def _cached_loader(fn):
    """Per-process cache of a read-only input file (scene cfg, info json), keyed by path. Each call gets a deep copy,
    since the own test writes the object's density into the scene cfg it is given."""
    memo = {}

    def load(path):
        if path not in memo:
            if len(memo) >= 64:
                memo.clear()
            memo[path] = fn(path)
        return copy.deepcopy(memo[path])

    return load


def _spec_from_file(path):
    if path not in _XML:
        _XML[path] = open(path).read()
    return mujoco.MjSpec.from_string(_XML[path])


def _mesh_bytes(full):
    if full.startswith(_HAND_DIR):
        if full not in _HAND_BYTES:
            _HAND_BYTES[full] = open(full, "rb").read()
        return _HAND_BYTES[full]
    b = _OBJ_BYTES.get(full)
    if b is None:
        b = _OBJ_BYTES[full] = open(full, "rb").read()
        _OBJ_SIZE[0] += len(b)
        while _OBJ_SIZE[0] > _OBJ_CAP:
            _OBJ_SIZE[0] -= len(_OBJ_BYTES.popitem(last=False)[1])
    else:
        _OBJ_BYTES.move_to_end(full)
    return b


def _spec_compile(spec):
    spec.assets = {os.path.join(spec.meshdir, m.file): _mesh_bytes(os.path.join(spec.meshdir, m.file))
                   for m in spec.meshes}
    return _SPEC["compile"](spec)


def install_assets():
    """A4-D88. A process builds an env for almost every group, since it keeps one pristine env and groups rotate
    objects. Each build read the hand XML and the object XML from NFS and stat'ed about 21 hand meshes (MuJoCo
    validates its mesh cache by file), about 18 NFS syscalls per item in the live trace (A4-D87). Here every XML is read once
    per process and parsed from its text (upstream rewrites each mesh path from the XML's own path right after, so
    the XML directory from_file would record is never used), and the hand's meshes, a fixed set, are handed to
    MuJoCo from memory. A4-D97: the object's pieces too, each read once per process and kept in a 256 MB LRU
    (median 3 pieces, 185 KB per object), since a process builds an object again whenever its groups come back.
    The compiled model is byte-identical to upstream's build from files (agent4_cap16_memassets_test.py)."""
    mujoco.MjSpec.from_file = staticmethod(_spec_from_file)
    mujoco.MjSpec.compile = _spec_compile


def uninstall_assets():
    mujoco.MjSpec.from_file = staticmethod(_SPEC["from_file"])
    mujoco.MjSpec.compile = _SPEC["compile"]


def install():
    """Refinement and own test. Both are proven bit-identical. The own test gains little CPU, since its time is
    simulation, but each fresh build reads the hand XML, 30 hand meshes and the object's pieces from NFS, and with
    hundreds of processes per NFS client those reads left most of kaist140's pool waiting in rpc_wait. The scene
    cfgs and info jsons, which never change during the run, are read once per process for the same reason."""
    G.MuJoCo_OptEnv = _factory("opt")
    install_eval()
    G.load_scene_cfg = _cached_loader(G.load_scene_cfg)
    E.load_scene_cfg = _cached_loader(E.load_scene_cfg)
    E.load_json = _cached_loader(E.load_json)
    install_assets()


def install_eval():
    E.MuJoCo_EvalEnv = _factory("eval")
