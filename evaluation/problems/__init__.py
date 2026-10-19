"""Unified registry for precise-assembly problem instances.

A `Problem` is the triple `(insertion_object, receptive_object, insert_poses)`
that fully specifies a precise-assembly env configuration. Each sub-package
(`peg`, `fabrica`, `furniture_bench`) registers its problems into
`PROBLEM_REGISTRY` on import; importing `evaluation.problems` triggers all of
them. The registry is then filtered down to the four problems released here
(see `_RELEASED_PROBLEMS` at the bottom of this file).
"""

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple


Pose7 = Tuple[float, float, float, float, float, float, float]


def make_pre_insert_sequence(
    final_pose: Pose7,
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    pre_insert_offset: float = 0.05,
) -> Tuple[Pose7, Pose7]:
    """Return the standard insertion sequence: pre-insert pose, then final pose."""
    x, y, z, qx, qy, qz, qw = (float(v) for v in final_pose)
    dx, dy, dz = (float(v) for v in insertion_direction)
    norm = sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 0.0:
        raise ValueError("insertion_direction must be non-zero")
    scale = float(pre_insert_offset) / norm
    pre_insert = (
        x - dx * scale,
        y - dy * scale,
        z - dz * scale,
        qx,
        qy,
        qz,
        qw,
    )
    final = (x, y, z, qx, qy, qz, qw)
    return pre_insert, final


def make_transport_pre_insert_sequence(
    final_pose: Pose7,
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    pre_insert_offset: float = 0.025,
    transport_offset: float = 0.10,
) -> Tuple[Pose7, Pose7, Pose7]:
    """Hole-frame trajectory tail: transport_above, pre_insert, final."""
    pre_insert, final = make_pre_insert_sequence(
        final_pose=final_pose,
        insertion_direction=insertion_direction,
        pre_insert_offset=pre_insert_offset,
    )
    px, py, pz, qx, qy, qz, qw = pre_insert
    dx, dy, dz = (float(v) for v in insertion_direction)
    norm = sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 0.0:
        raise ValueError("insertion_direction must be non-zero")
    scale = float(transport_offset) / norm
    transport_above = (
        px - dx * scale,
        py - dy * scale,
        pz - dz * scale,
        qx,
        qy,
        qz,
        qw,
    )
    return transport_above, pre_insert, final


@dataclass(frozen=True)
class Problem:
    """A single insertion problem.

    ``insert_pose_rel_receptive`` is the complete ordered insertion-subgoal
    sequence in the receptive frame. Plain insertion tasks use two poses:
    pre-insert, then assembled pose. Screw-style tasks can provide denser
    waypoints.
    """

    name: str                                          # registry key, e.g. "peg.tol0p5mm"
    insertion_object_name: str                         # key into NAME_TO_OBJECT
    receptive_urdf: str                                # path relative to assets/urdf/
    insert_pose_rel_receptive: Tuple[Pose7, ...]
    # World-frame Z offset for placing the receptive URDF root above the table
    # top. Typically half the receiver's canonical Z extent (so the receiver's
    # bottom face sits on the table top). The receptive URDF root is the
    # receiver's canonical-mesh origin (centered at its centroid by canonical-
    # mesh convention), so loading at z = table_top_z + hole_z_offset puts the
    # bottom face flush with the table.
    hole_z_offset: float = 0.0
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    pre_insert_offset: float = 0.05
    # If set, env constructs a per-episode prelude of [lift_in_place, over_hole]
    # at this Z lift above the start pose (with pre-insert orientation), prepended
    # to the hole-frame insertion sequence. Only honored when goal_mode allows it.
    prelude_lift_offset: float = 0.0

    def __post_init__(self) -> None:
        poses = self._normalize_pose_sequence(self.insert_pose_rel_receptive)
        object.__setattr__(self, "insert_pose_rel_receptive", poses)

    @staticmethod
    def _normalize_pose_sequence(raw) -> Tuple[Pose7, ...]:
        if len(raw) == 0:
            raise ValueError("insert_pose_rel_receptive must contain at least one pose")

        poses = []
        for pose in raw:
            if len(pose) != 7:
                raise ValueError(
                    "each insert_pose_rel_receptive entry must be length 7; "
                    f"got {len(pose)} in problem pose {pose!r}"
                )
            poses.append(tuple(float(v) for v in pose))
        return tuple(poses)

    @property
    def final_insert_pose_rel_receptive(self) -> Pose7:
        return self.insert_pose_rel_receptive[-1]


PROBLEM_REGISTRY: Dict[str, Problem] = {}


# Trigger registration of per-domain problems by importing the subpackages —
# each one's __init__.py imports its `problems` module for the side effect of
# populating PROBLEM_REGISTRY.
from . import peg               # noqa: E402, F401
from . import fabrica           # noqa: E402, F401
from . import furniture_bench   # noqa: E402, F401


# The data-driven registration above populates PROBLEM_REGISTRY with every
# problem variant the shipped assets support. This release exposes only the four
# assembly tasks from the paper, under their public (branded) names. We keep the
# internal registration logic intact and remap the four kept entries here.
import dataclasses as _dc  # noqa: E402

# branded public name -> internal registry key produced by the registration code
_RELEASED_PROBLEMS = {
    "tight_insertion": "Lpeg_matchedmass.tol0p5mm",
    # Beam assembly order: part_2 is inserted first (step 1), part_0 second (step 2).
    "beam_assembly_step1": "fabrica.beam_3x.part_2_matchedmass_sdf_hybrid",
    "beam_assembly_step2": "fabrica.beam_3x.part_0_matchedmass_sdf_hybrid",
    "screwing": "furniture_bench.one_leg_matchedmass_sdf_hybrid_super_dense",
}

_missing = [k for k in _RELEASED_PROBLEMS.values() if k not in PROBLEM_REGISTRY]
if _missing:
    raise RuntimeError(
        "evaluation.problems: expected problems are missing from PROBLEM_REGISTRY: "
        f"{_missing}. Check that the corresponding assets under assets/urdf/ are present."
    )

# Rebuild the registry keyed by branded name (Problem.name updated to match).
_branded = {
    public: _dc.replace(PROBLEM_REGISTRY[internal], name=public)
    for public, internal in _RELEASED_PROBLEMS.items()
}
PROBLEM_REGISTRY.clear()
PROBLEM_REGISTRY.update(_branded)


from pathlib import Path as _Path  # noqa: E402
_REPO_ROOT_FOR_BODY3X = _Path(__file__).resolve().parents[2]


# --- body-3x screw leg -------------------------------------------------------------------
# The released FurnitureBench leg is the original 1x part at 87.8 mm. The paper's Screw-Leg uses a
# 3x-longer leg, and the fixed reward box, 141 mm at keypoint_scale 1.5 so 211 mm, is sized for
# that one rather than for the 88 mm part. The body is stretched 3x along the leg axis, the thread
# is copied untouched so it still mates with the same socket, and the part is recentred on its new
# bounds. Recentring moves the object frame by `_BODY3X_SHIFT`, so every insertion pose moves by
# -R(q) @ shift to leave the thread landing in the same place.
import math as _math  # noqa: E402

from evaluation.objects import NAME_TO_OBJECT as _NAME_TO_OBJECT, Object as _Object  # noqa: E402

_BODY3X_URDF = (
    _REPO_ROOT_FOR_BODY3X
    / "assets/urdf/furniture_bench/square_table/square_table_leg4_body3x/sdf_hybrid"
    / "square_table_leg4_body3x_matchedmass_sdf_hybrid.urdf"
)
_BODY3X_KEY = "furniture_bench_square_table_leg4_body3x_matchedmass_sdf_hybrid"
_BODY3X_SCALE = (5.3555, 0.75, 0.75)
_BODY3X_SHIFT = (-0.063359, 0.0, 0.0)


def _quat_rotate_xyzw(q, v):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    rot = (
        (1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)),
    )
    return tuple(sum(rot[i][j] * v[j] for j in range(3)) for i in range(3))


if _BODY3X_URDF.is_file():
    _NAME_TO_OBJECT[_BODY3X_KEY] = _Object(
        urdf_path=_BODY3X_URDF, scale=_BODY3X_SCALE, need_vhacd=False,
    )
    _screw = PROBLEM_REGISTRY["screwing"]
    _moved = []
    for _p in _screw.insert_pose_rel_receptive:
        _d = _quat_rotate_xyzw(_p[3:7], _BODY3X_SHIFT)
        _moved.append((_p[0] - _d[0], _p[1] - _d[1], _p[2] - _d[2], _p[3], _p[4], _p[5], _p[6]))
    PROBLEM_REGISTRY["screwing"] = _dc.replace(
        _screw,
        insertion_object_name=_BODY3X_KEY,
        insert_pose_rel_receptive=tuple(_moved),
    )


# --- FMB single-object board 1 -------------------------------------------------------------
# FMB ships CAD and nothing else, so the assets and the goal poses are generated from its STEP
# files. The nine peg board is nine problems, one per opening, filled far row first and left to
# right, and problem k receives the board with pegs 0 to k-1 already seated. Two goals each, the
# peg resting on the board face and then seated on the board floor.
import json as _json  # noqa: E402

_FMB_DIR = _REPO_ROOT_FOR_BODY3X / "assets/urdf/fmb/board1"
_FMB_SPEC = _FMB_DIR / "fmb_board1.json"

if _FMB_SPEC.is_file():
    _fmb = _json.loads(_FMB_SPEC.read_text())
    for _fp in _fmb["problems"]:
        _ftag = _fp["tag"]
        _fkey = "fmb_board1_" + _ftag
        _NAME_TO_OBJECT[_fkey] = _Object(
            urdf_path=_FMB_DIR / (_ftag + ".urdf"),
            scale=tuple(_fp["scale"]),
            need_vhacd=False,
        )
        PROBLEM_REGISTRY[_fkey] = Problem(
            name=_fkey,
            insertion_object_name=_fkey,
            receptive_urdf="urdf/fmb/board1/fixture_" + _ftag + ".urdf",
            insert_pose_rel_receptive=tuple(tuple(_g) for _g in _fp["goals"]),
            hole_z_offset=_fmb["hole_z_offset"],
        )


# --- FMB multi-object sets -----------------------------------------------------------------
# Board N.step ships the four parts already assembled, so each goal is read off the CAD. Part k in
# the fixed order is one problem and receives the board with the parts before it standing. Parts
# that must first clear a standing part get a third goal ahead of the usual two.
for _sspec in sorted((_REPO_ROOT_FOR_BODY3X / "assets/urdf/fmb").glob("set*/fmb_set*.json")):
    _sd = _json.loads(_sspec.read_text())
    for _sp in _sd["problems"]:
        _skey = "fmb_set" + str(_sd["set"]) + "_" + _sp["tag"]
        _NAME_TO_OBJECT[_skey] = _Object(
            urdf_path=_sspec.parent / (_sp["tag"] + ".urdf"),
            scale=tuple(_sp["scale"]),
            need_vhacd=False,
        )
        PROBLEM_REGISTRY[_skey] = Problem(
            name=_skey,
            insertion_object_name=_skey,
            receptive_urdf="urdf/fmb/" + _sspec.parent.name + "/fixture_" + _sp["tag"] + ".urdf",
            insert_pose_rel_receptive=tuple(tuple(_g["pose"]) for _g in _sp["goals"]),
            hole_z_offset=_sd["hole_z_offset"],
        )


# --- DexVerse contact-rich ports -----------------------------------------------------------
# Geometry for both comes out of the shipped USDs rather than the config constants, whose half
# widths are named radii. In both the first goal stands the part at the mouth and the second is the
# assembled pose, so the pre-insert offset is the insertion depth plus a clearance rather than the
# fixed 50 mm our shallow holes use.
#
# insert-peg is ManiSkill PegInsertionSide. The socket is a 200 mm cube with a 46 mm square bore
# straight through it. DexVerse runs that bore along local x, which our env would place left to
# right across the robot, leaving it facing a solid wall, so the asset is re-authored with the bore
# on local y and the entry mouth at +y. Seated is their success pose, 85 of the peg's 200 mm inside.
#
# insert-pen stands a 185 mm pen in a 120 mm cup. The cup's own hull would seal the cavity, so its
# collision is the pieces from the learned decomposer, probed open to r 25 mm over z 15 to 115 mm.
# Seated rests the tip on the cup floor at 8 mm, 112 mm of insertion.
#
# The admissible receptacle xy is not free either. Goal world position is hole_pos + goal_local, and
# the pretrained goal box bounds the goal root, so the range is the intersection over both goals of
# target_volume_mins/maxs minus goal_local. That is set per run on the launch line, since the task
# carries one hole_x_range for whichever problem it is training.
_DXV = _REPO_ROOT_FOR_BODY3X / "assets/urdf"

if (_DXV / "dexverse_insertpeg/peg.urdf").is_file():
    _NAME_TO_OBJECT["dexverse_insertpeg"] = _Object(
        urdf_path=_DXV / "dexverse_insertpeg/peg.urdf",
        scale=(5.0, 1.0, 1.0),  # 200 x 40 x 40 mm over object_base_size 0.04
        need_vhacd=False,
    )
    PROBLEM_REGISTRY["dexverse_insertpeg"] = Problem(
        name="dexverse_insertpeg",
        insertion_object_name="dexverse_insertpeg",
        receptive_urdf="urdf/dexverse_insertpeg/socket.urdf",
        insert_pose_rel_receptive=make_pre_insert_sequence(
            (0.0, 0.115, 0.0, 0.0, 0.0, -0.70710678, 0.70710678),
            insertion_direction=(0.0, -1.0, 0.0),
            pre_insert_offset=0.090,  # 85 mm of insertion plus 5 mm clear of the mouth
        ),
        hole_z_offset=0.10,
        insertion_direction=(0.0, -1.0, 0.0),
        pre_insert_offset=0.090,
    )

if (_DXV / "dexverse_insertpen/pen.urdf").is_file():
    _NAME_TO_OBJECT["dexverse_insertpen"] = _Object(
        urdf_path=_DXV / "dexverse_insertpen/pen.urdf",
        scale=(4.6269, 0.465, 0.5325),  # 185.1 x 18.6 x 21.3 mm over 0.04
        need_vhacd=False,  # the urdf already carries the decomposer's convex pieces
    )
    PROBLEM_REGISTRY["dexverse_insertpen"] = Problem(
        name="dexverse_insertpen",
        insertion_object_name="dexverse_insertpen",
        receptive_urdf="urdf/dexverse_insertpen/holder.urdf",
        insert_pose_rel_receptive=make_pre_insert_sequence(
            (0.0095, 0.0, 0.098, 0.0, -0.70710678, 0.0, 0.70710678),
            insertion_direction=(0.0, 0.0, -1.0),
            pre_insert_offset=0.122,  # 112 mm of insertion plus 10 mm clear of the rim
        ),
        hole_z_offset=0.0,  # the cup's root is its own bottom face
        pre_insert_offset=0.122,
    )
