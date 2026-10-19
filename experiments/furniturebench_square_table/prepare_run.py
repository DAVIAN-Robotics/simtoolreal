"""Copy the exact saved inputs into a fresh, writable run directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    if work.exists():
        parser.error(f"Run directory already exists: {work}. Choose a fresh directory.")
    manifest = json.loads((HERE / "provenance/manifest.json").read_text())
    for name, entry in manifest["inputs"].items():
        path = HERE / "inputs" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
    work.mkdir(parents=True)
    for name in manifest["inputs"]:
        shutil.copy2(HERE / "inputs" / name, work / name)
    # The native W&B observer consumes this diff on workers without Git.
    try:
        diff = subprocess.run(
            ["git", "diff", manifest["parent_commit"], "--"],
            cwd=REPO, capture_output=True, text=True,
        )
        diff_text = diff.stdout if diff.returncode == 0 else None
    except FileNotFoundError:
        diff_text = None
    if diff_text is None:
        diff_text = (HERE / "provenance/portability.patch").read_text()
    (work / "source.patch").write_text(diff_text)
    shutil.copy2(HERE / "provenance/manifest.json", work / "source_manifest.json")
    (work / "logs").mkdir()
    print(work)


if __name__ == "__main__":
    main()
