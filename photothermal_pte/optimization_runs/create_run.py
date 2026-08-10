#!/usr/bin/env python3
"""Create the next immutable optimization run directory from the template."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def next_id(root: Path) -> int:
    values = []
    for entry in root.glob("run_[0-9][0-9][0-9]_*"):
        match = re.match(r"run_([0-9]{3})_", entry.name)
        if match:
            values.append(int(match.group(1)))
    return max(values, default=0) + 1


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", args.slug):
        raise SystemExit("slug must use lowercase letters, digits, underscores")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise SystemExit("source commit must be a full Git SHA")
    root = args.root.resolve()
    template = root / "_template"
    run_id = f"run_{next_id(root):03d}_{args.slug}"
    destination = root / run_id
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    shutil.copytree(template, destination)
    template_config = json.loads((destination / "run_config.template.json").read_text())
    config = deepcopy(template_config)
    config["run_id"] = run_id
    config["description"] = args.description
    config["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["source"]["git_commit"] = args.source_commit
    (destination / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "run_config.template.json").unlink()
    status = {
        "run_id": run_id,
        "status": "PLANNED",
        "last_updated_utc": config["created_at_utc"],
        "optimization_started": False,
        "message": "Created from the repository template; solver not launched.",
    }
    (destination / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    readme = (destination / "README.md").read_text().replace("RUN_ID", run_id)
    (destination / "README.md").write_text(readme)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
