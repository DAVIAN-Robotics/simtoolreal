"""Procedural handle-head URDF generation for SimToolReal.

Clean port of the minimal subset of isaacgymenvs/tasks/simtoolreal/generate_objects.py
needed for the base goal-pose-reaching env: `generate_handle_urdf` (single cuboid
or cylinder) and `generate_handle_head_urdf` (handle + head composite with
variable densities + parallel-axis-adjusted inertia).

Every URDF is emitted in the object's own bounding-box frame. The primitives are
described once, as parts, and that one description builds both the mesh the box
is measured from and the geometry the URDF carries, so the two cannot disagree.

The top-level entry `generate_handle_head_urdfs(types, num_per_type, out_dir)`
mirrors legacy `env.py:_handle_head_primitives`: per-type sampling with
`np.random.seed(42)`, pool shuffle, and box extents normalized by
`object_base_size`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh

from dextoolbench.oriented_box import oriented_box, reexpress_origin

from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS, Scale2, Scale3

Scale = Union[Scale2, Scale3]

# Matches legacy env.py:1732.
_SEED = 42
# Matches legacy env.py:1731; kept module-level so call sites can override for tests.
_NUM_OBJECTS_PER_TYPE_DEFAULT = 100
# Reward-space normalization constant (same value the env uses when rescaling).
_OBJECT_BASE_SIZE = 0.04

# The oriented box of a tessellated cylinder depends on where the facets fall, so
# the section count is pinned rather than left to the trimesh default.
_CYLINDER_SECTIONS = 32


# ----------------------------------------------------------------------------
# Parts: one description, used for both the measured mesh and the emitted URDF
# ----------------------------------------------------------------------------


# All procedural object URDFs use this same link name so MultiUsdFileCfg-spawned
# prims have an identical internal structure across envs (env_*/Object/<LINK>).
# RigidObject's view regex is derived from env_0's structure and applied to all
# envs; if the link name varies, only matching envs are registered and
# set_transforms overflows the partial view.
_OBJECT_ROOT_LINK = "object_root"

_CYL_ALONG_X = (0.0, -math.pi / 2, 0.0)
_CYL_ALONG_Y = (-math.pi / 2, 0.0, 0.0)


def _box_part(extents, xyz, rpy):
    lx, ly, lz = extents
    return {
        "geometry": f'<geometry><box size="{lx} {ly} {lz}"/></geometry>',
        "mesh": trimesh.creation.box(extents=extents),
        "xyz": xyz,
        "rpy": rpy,
    }


def _cylinder_part(length, radius, xyz, rpy):
    return {
        "geometry": f'<geometry><cylinder length="{length}" radius="{radius}"/></geometry>',
        "mesh": trimesh.creation.cylinder(
            height=length, radius=radius, sections=_CYLINDER_SECTIONS
        ),
        "xyz": xyz,
        "rpy": rpy,
    }


def _handle_part(handle_scale: Scale):
    """The handle primitive, long axis along +x in the authored frame."""
    if len(handle_scale) == 3:
        return _box_part(handle_scale, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    height, diameter = handle_scale
    return _cylinder_part(height, diameter / 2, (0.0, 0.0, 0.0), _CYL_ALONG_X)


def _head_part(head_scale: Scale, handle_scale: Scale):
    """The head primitive, attached beyond the +x end of the handle."""
    if len(head_scale) == 3:
        x_offset = handle_scale[0] / 2 + head_scale[0] / 2
        return _box_part(head_scale, (x_offset, 0.0, 0.0), (0.0, 0.0, 0.0))
    height, diameter = head_scale
    radius = diameter / 2
    x_offset = handle_scale[0] / 2 + radius
    return _cylinder_part(height, radius, (x_offset, 0.0, 0.0), _CYL_ALONG_Y)


def _part_mesh(part) -> trimesh.Trimesh:
    mesh = part["mesh"].copy()
    mesh.apply_transform(_pose(part["xyz"], part["rpy"]))
    return mesh


def _pose(xyz, rpy) -> np.ndarray:
    m = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    m[:3, 3] = xyz
    return m


# ----------------------------------------------------------------------------
# Mass and inertia
# ----------------------------------------------------------------------------


def _compute_mass_and_inertia(scale: Scale, density: float):
    """Capsule-approximation for cylinders; exact for cuboids.

    Returns (m, ixx, iyy, izz) with scale-axis = z (caller flips if needed).
    """
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly * ly + lz * lz)
        iyy = (1 / 12) * m * (lx * lx + lz * lz)
        izz = (1 / 12) * m * (lx * lx + ly * ly)
        return m, ixx, iyy, izz
    if len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        # Capsule mass = cylinder + two hemispheres.
        m_c = density * math.pi * r * r * h
        m_h = density * (2 / 3) * math.pi * r ** 3
        m = m_c + 2 * m_h
        # Cylinder inertia about centroid (axis = z).
        i_c_axis = 0.5 * m_c * r * r
        i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
        # Hemisphere inertia about its own centroid.
        i_h_axis = (2 / 5) * m_h * r * r
        i_h_perp = (83 / 320) * m_h * r * r
        d_com = (h / 2) + (3 * r / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return m, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale}")


def _authored_frame_inertia(scale: Scale, density: float):
    """Mass and diagonal inertia expressed in the authored frame, x along the part."""
    if len(scale) == 3:
        return _compute_mass_and_inertia(scale, density)
    # Cylinder is modelled axis-along-z then rotated to +x, so ixx and izz swap.
    m, izz, iyy, ixx = _compute_mass_and_inertia(scale, density)
    return m, ixx, iyy, izz


# ----------------------------------------------------------------------------
# URDF emission, in the object's own box frame
# ----------------------------------------------------------------------------


def _write_urdf(path: Path, name: str, parts, mass, inertia, inertial_xyz, inertial_rpy):
    """Emit the URDF in the object's own box frame. Returns (path, box extents)."""
    mesh = trimesh.util.concatenate([_part_mesh(p) for p in parts])
    extents, box_in_mesh = oriented_box(mesh)
    mesh_in_box = np.linalg.inv(box_in_mesh)

    bodies = []
    materials = ('<material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>',
                 '<material name="gray"><color rgba="0.5 0.5 0.5 1.0"/></material>')
    for i, part in enumerate(parts):
        xyz, rpy = reexpress_origin(mesh_in_box, part["xyz"], part["rpy"])
        origin = f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>'
        bodies.append(
            f"    <visual>\n      {origin}\n      {part['geometry']}\n"
            f"      {materials[min(i, 1)]}\n    </visual>\n"
            f"    <collision>\n      {origin}\n      {part['geometry']}\n    </collision>"
        )

    ixyz, irpy = reexpress_origin(mesh_in_box, inertial_xyz, inertial_rpy)
    ixx, iyy, izz = inertia
    body_xml = "\n".join(bodies)
    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{_OBJECT_ROOT_LINK}">
{body_xml}
    <inertial>
      <origin xyz="{ixyz[0]} {ixyz[1]} {ixyz[2]}" rpy="{irpy[0]} {irpy[1]} {irpy[2]}"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path, extents


def _handle_urdf(path: Path, handle_scale: Scale, handle_density: float):
    """Emit a single-primitive object. Returns (path, box extents)."""
    part = _handle_part(handle_scale)
    # Standard URDF requires <mass> + <inertia>. Legacy isaacgym's URDF importer
    # accepts <density> as a Gazebo extension and back-computes mass from
    # density x volume; Isaac Sim's importer follows strict URDF spec and falls
    # back to 1 kg when <mass> is missing, which breaks erasers and any other
    # handle-only type.
    mass, ixx, iyy, izz = _authored_frame_inertia(handle_scale, handle_density)
    name = "cuboid" if len(handle_scale) == 3 else "cylinder"
    return _write_urdf(
        path, name, [part], mass, (ixx, iyy, izz), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )


def _handle_head_urdf(
    path: Path,
    handle_scale: Scale,
    head_scale: Scale,
    handle_density: float,
    head_density: float,
):
    handle = _handle_part(handle_scale)
    head = _head_part(head_scale, handle_scale)

    handle_mass, handle_ixx, handle_iyy, handle_izz = _authored_frame_inertia(
        handle_scale, handle_density
    )
    head_mass, head_ixx, head_iyy, head_izz = _authored_frame_inertia(
        head_scale, head_density
    )
    if len(head_scale) == 2:
        # Head cylinder lies along +y rather than +x, so its axis inertia is iyy.
        head_mass, head_ixx, head_izz, head_iyy = _compute_mass_and_inertia(
            head_scale, head_density
        )

    # Parallel-axis shift to composite COM (handle at origin, head at +x_offset).
    x_offset = head["xyz"][0]
    total_mass = handle_mass + head_mass
    com_x = (handle_mass * 0.0 + head_mass * x_offset) / total_mass
    d_handle = -com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle * d_handle) + (
        head_iyy + head_mass * d_head * d_head
    )
    izz = (handle_izz + handle_mass * d_handle * d_handle) + (
        head_izz + head_mass * d_head * d_head
    )

    return _write_urdf(
        path, "handle_head", [handle, head], total_mass, (ixx, iyy, izz),
        (com_x, 0.0, 0.0), (0.0, 0.0, 0.0),
    )


def _urdf_and_extents(
    path: Path,
    handle_scale: Scale,
    head_scale: Optional[Scale],
    handle_density: float,
    head_density: Optional[float],
):
    """Emit one object and return (path, its own box extents in metres)."""
    if head_scale is None and head_density is None:
        return _handle_urdf(path, handle_scale, handle_density)
    if head_scale is not None and head_density is not None:
        return _handle_head_urdf(
            path, handle_scale, head_scale, handle_density, head_density
        )
    raise ValueError(
        f"head_scale and head_density must both be set or both None (got "
        f"{head_scale} and {head_density})"
    )


def generate_handle_urdf(
    path: Path, handle_scale: Scale, handle_density: float = 400.0
) -> Path:
    """Emit a URDF for a single cuboid (len(scale)==3) or cylinder (len(scale)==2) handle."""
    return _handle_urdf(path, handle_scale, handle_density)[0]


def generate_handle_head_urdf(
    path: Path,
    handle_scale: Scale,
    head_scale: Optional[Scale],
    handle_density: float = 400.0,
    head_density: Optional[float] = 800.0,
) -> Path:
    """Emit a URDF for a handle (no head) or a handle+head composite.

    The composite is single-link with parallel-axis-adjusted inertia; the 2-link
    variant was unstable per legacy env.py:1723-1724 and is not ported.
    """
    return _urdf_and_extents(
        path, handle_scale, head_scale, handle_density, head_density
    )[0]


# ----------------------------------------------------------------------------
# Top-level: generate a pool of N×types URDFs for the env
# ----------------------------------------------------------------------------


def generate_handle_head_urdfs(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    out_dir: Union[str, Path] = "/tmp/simtoolreal_assets",
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Generate a pool of URDFs across the requested handle-head types.

    Mirrors legacy ``_handle_head_primitives`` at
    isaacgymenvs/tasks/simtoolreal/env.py:1706-1840:
      1. Seed NumPy with ``seed`` (deterministic pool per run).
      2. For each matching ``ObjectSizeDistribution``, sample ``num_per_type``
         (handle_scale, head_scale, handle_density, head_density) tuples.
      3. Emit one URDF per sample, each in that object's own bounding-box frame;
         return file paths paired with the box extents normalized by
         ``object_base_size``.
      4. Shuffle (paths, scales) in lockstep so env ``i`` gets pool entry
         ``i % len(pool)`` with uniform coverage over types.

    Returns
    -------
    urdf_paths : list[str]
        Absolute paths to generated URDFs. Length = ``num_per_type × len(matching types)``.
    object_scales_normalized : list[tuple[float, float, float]]
        Per-pool-entry box extents, normalized by ``object_base_size``. The
        policy reads these as the object descriptor and the reward measures
        against the corners they define.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

    np.random.seed(seed)

    type_set = set(handle_head_types)
    matching = [d for d in OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching ObjectSizeDistribution for handle_head_types={handle_head_types}. "
            f"Valid types: {sorted({d.type for d in OBJECT_SIZE_DISTRIBUTIONS})}"
        )

    paths: list[str] = []
    extents_m: list[np.ndarray] = []
    for dist in matching:
        # Sample order MUST match legacy env.py:1758-1772 (densities first,
        # scales second) — the two pools share seed=42 so first-asset URDFs
        # come out byte-identical across backends only when the np.random
        # draws line up step for step.
        handle_densities = dist.sample_handle_densities(num_per_type)
        head_densities = dist.sample_head_densities(num_per_type)
        handle_scales = dist.sample_handle_scales(num_per_type)
        head_scales = dist.sample_head_scales(num_per_type)

        for idx in range(num_per_type):
            h_scale = tuple(float(x) for x in handle_scales[idx])
            head = tuple(float(x) for x in head_scales[idx]) if head_scales is not None else None
            h_d = float(handle_densities[idx])
            head_d = float(head_densities[idx]) if head_densities is not None else None

            fname = (
                f"{idx:03d}_{dist.type}_handle_{h_scale}_head_{head}_d{h_d:.1f}_{head_d}"
                .replace(".", "-")
                + ".urdf"
            )
            urdf_path, extents = _urdf_and_extents(
                path=out_dir / fname,
                handle_scale=h_scale,
                head_scale=head,
                handle_density=h_d,
                head_density=head_d,
            )
            paths.append(str(urdf_path))
            extents_m.append(extents)

    scales_norm = [
        (float(e[0] / object_base_size), float(e[1] / object_base_size), float(e[2] / object_base_size))
        for e in extents_m
    ]

    # Shuffle (paths, scales) in lockstep so the per-type ordering doesn't bias
    # env-i-gets-asset-i%N. Matches legacy env.py:1824-1830. Disabled for
    # debug-parity runs (debug_differences/policy_rollout_isaacsim.py) where
    # we want pool[0] = first matching ObjectSizeDistribution.
    if shuffle:
        indices = np.arange(len(paths))
        np.random.shuffle(indices)
        paths = [paths[i] for i in indices]
        scales_norm = [scales_norm[i] for i in indices]

    return paths, scales_norm


__all__ = [
    "generate_handle_urdf",
    "generate_handle_head_urdf",
    "generate_handle_head_urdfs",
]
