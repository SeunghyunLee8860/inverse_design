#!/usr/bin/env python3
"""Refresh only repository PNG records in report artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


REPOSITORY = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dictionaries(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPOSITORY / "photothermal_pte" / "reports",
    )
    args = parser.parse_args()
    changed_records = 0
    changed_manifests = 0
    for manifest_path in sorted(args.report_root.rglob("RAW_ARTIFACT_MANIFEST.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_changed = False
        for record in dictionaries(payload):
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path.lower().endswith(".png"):
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            else:
                path = path.resolve()
            if not path.exists() or not path.is_relative_to(REPOSITORY):
                continue
            size = path.stat().st_size
            digest = sha256(path)
            size_key = "size_bytes" if "size_bytes" in record else "bytes"
            if record.get(size_key) != size or record.get("sha256") != digest:
                record[size_key] = size
                record["sha256"] = digest
                manifest_changed = True
                changed_records += 1
        if manifest_changed:
            manifest_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            changed_manifests += 1
    print(
        json.dumps(
            {
                "changed_manifests": changed_manifests,
                "changed_png_records": changed_records,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
