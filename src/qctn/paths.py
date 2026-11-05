"""Locations of the gitignored third-party data, shared by all worktrees."""
import os
import subprocess
from pathlib import Path

AUTHORS_COMMIT = "936630aa161abb7bcab1557c91137696b3bc5f68"


def external_dir() -> Path:
    """`$QCTN_EXTERNAL`, or `external/` in the main checkout (see scripts/setup_external.sh)."""
    if env := os.environ.get("QCTN_EXTERNAL"):
        return Path(env)
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent / "external"


def authors_repo() -> Path:
    return external_dir() / "authors"


def authors_plans() -> Path:
    return external_dir() / "authors-data" / "plans" / "resultats"


def authors_circuits() -> Path:
    return external_dir() / "authors-data" / "circuits"
