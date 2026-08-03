from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh

from dextoolbench.oriented_box import oriented_box
from isaacgymenvs.utils.utils import get_repo_root_dir

# Reward-space normalization, the same constant both backends use.
OBJECT_BASE_SIZE = 0.04


@dataclass
class Object:
    urdf_path: Path
    """Path to the object URDF file."""

    need_vhacd: bool
    """Legacy flag (now always False): collision is pre-decomposed offline by
    generate_collision_meshes.py into decomposed_urdf_path, so neither backend
    runs runtime V-HACD."""

    def __post_init__(self):
        assert self.urdf_path.exists(), f"Filepath {self.urdf_path} does not exist"

    @cached_property
    def scale(self) -> Tuple[float, float, float]:
        """The object's own bounding box, extents over object_base_size.

        Computed from the mesh by the same routine the procedural pool uses, so
        an eval object and a training object are described the same way. The
        URDF is authored in this box's frame, so the box needs no centre.
        """
        extents, _ = oriented_box(self.get_object_mesh())
        return tuple(float(v / OBJECT_BASE_SIZE) for v in extents)

    @property
    def decomposed_urdf_path(self) -> Path:
        """URDF whose collision is the offline CoACD convex parts (visual stays
        the original mesh, inertial is explicit). Loaded by both backends."""
        return self.urdf_path.with_name(f"{self.urdf_path.stem}_decomposed.urdf")

    def get_object_mesh_path_and_scale(self) -> Tuple[Path, np.ndarray]:
        from yourdfpy import URDF

        object_urdf_path = self.urdf_path

        assert object_urdf_path.exists(), object_urdf_path
        urdf = URDF.load(str(object_urdf_path))

        mesh_path_and_scale_list = []
        for link in urdf.robot.links:
            if len(link.collisions) == 0:
                continue

            for i, collision_link in enumerate(link.collisions):
                mesh_path = (
                    object_urdf_path.parent / collision_link.geometry.mesh.filename
                )
                assert mesh_path.exists(), mesh_path

                mesh_scale = (
                    np.array([1, 1, 1])
                    if collision_link.geometry.mesh.scale is None
                    else np.array(collision_link.geometry.mesh.scale)
                )
                mesh_path_and_scale_list.append((mesh_path, mesh_scale))

        # Assume urdf has only 1 link with only 1 collision mesh
        assert len(mesh_path_and_scale_list) == 1, (
            f"{mesh_path_and_scale_list} has len {len(mesh_path_and_scale_list)}"
        )

        mesh_path, mesh_scale = mesh_path_and_scale_list[0]
        return mesh_path, mesh_scale

    def get_object_mesh(self) -> trimesh.Trimesh:
        mesh_path, mesh_scale = self.get_object_mesh_path_and_scale()
        mesh = trimesh.load_mesh(str(mesh_path))
        mesh.apply_scale(mesh_scale)
        return mesh


NAME_TO_OBJECT: Dict[str, Object] = {}

HAMMER_NAME_TO_OBJECT = {
    "mallet_hammer": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.urdf"
        ),
        need_vhacd=False,
    ),
    "claw_hammer": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf"
        ),
        need_vhacd=False,
    ),
}

# overwrite NAME_TO_OBJECT with HAMMER_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(HAMMER_NAME_TO_OBJECT)

##SCREWDRIVERS
SCREWDRIVER_NAME_TO_OBJECT = {
    "long_screwdriver": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/screwdriver/long_screwdriver/long_screwdriver.urdf"
        ),
        need_vhacd=False,
    ),
    "short_screwdriver": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/screwdriver/short_screwdriver/short_screwdriver.urdf"
        ),
        need_vhacd=False,
    ),
}
NAME_TO_OBJECT.update(SCREWDRIVER_NAME_TO_OBJECT)

ERASER_NAME_TO_OBJECT = {
    "handle_eraser": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/eraser/handle_eraser/handle_eraser.urdf"
        ),
        need_vhacd=False,
    ),
    "flat_eraser": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.urdf"
        ),
        need_vhacd=False,
    ),
}

NAME_TO_OBJECT.update(ERASER_NAME_TO_OBJECT)

SPATULA_NAME_TO_OBJECT = {
    "flat_spatula": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/spatula/flat_spatula/flat_spatula.urdf"
        ),
        need_vhacd=False,
    ),
    "spoon_spatula": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/spatula/spoon_spatula/spoon_spatula.urdf"
        ),
        need_vhacd=False,
    ),
}
NAME_TO_OBJECT.update(SPATULA_NAME_TO_OBJECT)

MARKER_NAME_TO_OBJECT = {
    "sharpie_marker": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/marker/sharpie_marker/sharpie_marker.urdf"
        ),
        need_vhacd=False,
    ),
    "staples_marker": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/marker/staples_marker/staples_marker.urdf"
        ),
        need_vhacd=False,
    ),
}
NAME_TO_OBJECT.update(MARKER_NAME_TO_OBJECT)

BRUSH_NAME_TO_OBJECT = {
    "red_brush": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/brush/red_brush/red_brush.urdf"
        ),
        need_vhacd=False,
    ),
    "blue_brush": Object(
        urdf_path=(
            get_repo_root_dir()
            / "assets/urdf/dextoolbench/brush/blue_brush/blue_brush.urdf"
        ),
        need_vhacd=False,
    ),
}
NAME_TO_OBJECT.update(BRUSH_NAME_TO_OBJECT)
