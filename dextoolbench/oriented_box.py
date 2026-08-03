"""The object's own oriented bounding box, and the frame it defines.

The box is the frame. Origin at the box centre, axes the box axes, so a box
needs no centre field and no rotation field beside its three extents. Nothing
about a handle or a head enters here, which is the point: the same routine takes
a procedural primitive today and an arbitrary scanned mesh next.

Corner order is the full sign product with x outermost, so every consumer indexes
the same corner.
"""

import numpy as np
import trimesh
from trimesh import bounds

# trimesh deduplicates convex-hull normal directions before searching, controlled
# by angle_digits. Its default of 1 collapses distinct normals and misses
# orientations: over the 1200-object procedural pool it returns a box larger than
# the freely available axis-aligned one on 13 objects, worst case 6.5 percent
# oversized. 3 leaves one object 0.25 percent over, and 6 gains nothing further.
ANGLE_DIGITS = 3

CORNER_SIGNS = tuple(
    (sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)
)
NUM_CORNERS = len(CORNER_SIGNS)


def oriented_box(mesh: trimesh.Trimesh):
    """Return (extents, box_in_mesh) for the mesh's own oriented bounding box.

    ``box_in_mesh`` is the 4x4 placing the box frame in the mesh frame, so its
    inverse re-expresses mesh coordinates in the box frame.
    """
    to_origin, extents = bounds.oriented_bounds(mesh, angle_digits=ANGLE_DIGITS)
    box_in_mesh = np.linalg.inv(np.asarray(to_origin, dtype=float))
    return np.asarray(extents, dtype=float), box_in_mesh


def pose_matrix(xyz, rpy) -> np.ndarray:
    """URDF origin to 4x4. URDF rpy is fixed-axis xyz, which is trimesh's 'sxyz'."""
    m = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    m[:3, 3] = xyz
    return m


def reexpress_origin(mesh_in_box: np.ndarray, xyz, rpy):
    """Move one URDF origin from the mesh frame into the box frame."""
    m = mesh_in_box @ pose_matrix(xyz, rpy)
    r, p, y = trimesh.transformations.euler_from_matrix(m, axes="sxyz")
    return tuple(m[:3, 3]), (r, p, y)


def corner_offsets(extents) -> np.ndarray:
    """The eight corners of a box of these extents, in the box frame. Shape (8, 3)."""
    half = np.asarray(extents, dtype=float) * 0.5
    return np.asarray(CORNER_SIGNS, dtype=float) * half
