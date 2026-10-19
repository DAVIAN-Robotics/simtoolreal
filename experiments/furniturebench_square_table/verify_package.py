"""Check published source/input hashes, reset identities, and required mesh paths."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    manifest = json.loads((HERE / "provenance/manifest.json").read_text())
    for relative, entry in manifest["source_files"].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == entry["published_sha256"], relative
    for name, entry in manifest["inputs"].items():
        assert hashlib.sha256((HERE / "inputs" / name).read_bytes()).hexdigest() == entry["sha256"], name
    for relative, entry in manifest["assets"].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == entry["sha256"], relative
    with np.load(HERE / "inputs/screw_resets.npz") as data:
        assert len(data["progress"]) == 636023
        assert np.bincount(data["leg_id"]).tolist() == [169683, 122620, 145486, 198234]
        assert set(data["progress"].tolist()) == set(range(8))
        assert data["arm_q"].shape == (636023, 7)
        assert data["hand_q"].shape == (636023, 12)
        assert data["hand_target"].shape == (636023, 12)
    spec = json.loads((HERE / "inputs/screw_goals.json").read_text())
    assert spec["objects"] == ["leg1", "leg2", "leg3", "leg4"]
    assert all(len(instance["events"]) == 9 for instance in spec["instances"])
    scene = spec["scene"]
    urdfs = [REPO / scene["robot_urdf"], REPO / scene["support_urdf"],
             HERE / "task/inputs" / scene["fixture_urdf"],
             HERE / "task/inputs" / scene["objects"][0]["urdf"]]
    mesh_count = 0
    for path in urdfs:
        for mesh in ET.parse(path).getroot().findall(".//mesh"):
            name = mesh.attrib["filename"]
            if name.startswith("package://"):
                target = REPO / name[len("package://"):]
            elif name.startswith("assets/"):
                target = REPO / name
            else:
                target = path.parent / name
            assert target.is_file(), (path, target)
            mesh_count += 1
    # Import the original registry too: it requires the accompanying released assets.
    from evaluation.problems import PROBLEM_REGISTRY
    assert "screwing" in PROBLEM_REGISTRY
    print(json.dumps({"status": "passed", "source_files": len(manifest["source_files"]),
                      "reset_rows": 636023, "legs": 4, "stages": 8,
                      "required_urdfs": len(urdfs), "mesh_references": mesh_count,
                      "asset_files": len(manifest["assets"])}))


if __name__ == "__main__":
    main()
