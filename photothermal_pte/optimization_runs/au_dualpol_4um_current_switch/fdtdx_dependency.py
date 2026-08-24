"""Fail-closed full-tree provenance for the fresh FDTDX source dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "fdtdx_dependency_lock.json"
DEFAULT_SOURCE = Path(
    "/home/seunghyun200/dependencies/fdtdx-f26f84b70a8cceec9b889553955a868624736bf1"
)


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("FDTDX dependency lock root must be an object")
    required = {
        "repository",
        "commit",
        "tree",
        "critical_files_sha256",
        "python_requires",
    }
    missing = required - set(value)
    if missing:
        raise RuntimeError(f"FDTDX dependency lock is missing {sorted(missing)}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(source), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalized_repository(value: str) -> str:
    return value.strip().removesuffix("/").removesuffix(".git")


def audit_source(source: Path) -> dict[str, Any]:
    """Audit a source checkout without importing or executing FDTDX."""

    source = Path(source).expanduser().resolve()
    lock = load_lock()
    checks: dict[str, bool] = {"source_directory_exists": source.is_dir()}
    errors: list[str] = []
    actual: dict[str, Any] = {"path": str(source)}
    if not checks["source_directory_exists"]:
        return {
            "status": "BLOCKED_PINNED_FDTDX_SOURCE_MISSING",
            "ready": False,
            "checks": checks,
            "errors": [f"missing source directory: {source}"],
            "expected": lock,
            "actual": actual,
        }

    try:
        actual.update(
            commit=_git(source, "rev-parse", "HEAD"),
            tree=_git(source, "rev-parse", "HEAD^{tree}"),
            repository=_git(source, "remote", "get-url", "origin"),
            dirty_porcelain=_git(source, "status", "--porcelain", "--untracked-files=all"),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"git provenance failed: {error}")
        actual.update(commit=None, tree=None, repository=None, dirty_porcelain=None)

    checks.update(
        commit_exact=actual.get("commit") == lock["commit"],
        tree_exact=actual.get("tree") == lock["tree"],
        repository_exact=(
            _normalized_repository(str(actual.get("repository") or ""))
            == _normalized_repository(str(lock["repository"]))
        ),
        worktree_clean=actual.get("dirty_porcelain") == "",
    )
    critical_actual: dict[str, str | None] = {}
    for relative, expected_hash in lock["critical_files_sha256"].items():
        path = source / relative
        actual_hash = sha256(path) if path.is_file() else None
        critical_actual[relative] = actual_hash
        checks[f"sha256:{relative}"] = actual_hash == expected_hash
    actual["critical_files_sha256"] = critical_actual
    checks["python_version_supported"] = (3, 11) <= sys.version_info[:2] < (3, 15)

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": (
            "VALIDATED_PINNED_FDTDX_SOURCE"
            if not failed and not errors
            else "BLOCKED_FDTDX_SOURCE_PROVENANCE"
        ),
        "ready": not failed and not errors,
        "checks": checks,
        "failed_checks": failed,
        "errors": errors,
        "expected": lock,
        "actual": actual,
    }


def require_source(source: Path) -> dict[str, Any]:
    result = audit_source(source)
    if result["ready"] is not True:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def configured_source() -> Path:
    configured = os.environ.get("FDTDX_SOURCE_DIR", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_SOURCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=configured_source())
    args = parser.parse_args()
    result = audit_source(args.source)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
