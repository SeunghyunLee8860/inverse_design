"""Auditable provenance checks for a non-editable local FDTDX install."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Callable
from urllib.parse import unquote, urlparse


GitRunner = Callable[[list[str], Path], str]


def _tree_sha256(root: Path) -> str | None:
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _installed_direct_url() -> str | None:
    try:
        raw = importlib.metadata.distribution("fdtdx").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if raw is None:
        return None
    try:
        value = json.loads(raw).get("url")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _file_url_path(url: str | None) -> Path | None:
    if url is None:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(unquote(parsed.path)).resolve()


def audit_runtime(
    source: Path,
    *,
    expected_commit: str,
    git_runner: GitRunner,
    resolved_module: Path | None = None,
) -> dict[str, object]:
    """Require an exact clean source and a byte-identical import package.

    A normal ``pip install /local/source`` copies the package into
    site-packages.  In that case, ``direct_url.json`` must identify the pinned
    source and the installed package tree must hash identically to
    ``source/src/fdtdx``.  An editable/direct source import also passes.
    """

    source = source.expanduser().resolve()
    commit: str | None = None
    dirty: str | None = None
    error: str | None = None
    if source.is_dir():
        try:
            commit = git_runner(["rev-parse", "HEAD"], source)
            dirty = git_runner(["status", "--porcelain"], source)
        except (OSError, subprocess.SubprocessError) as exc:
            error = str(exc)
    else:
        error = f"missing source directory: {source}"

    if resolved_module is None:
        spec = importlib.util.find_spec("fdtdx")
        if spec is not None and spec.origin is not None:
            resolved_module = Path(spec.origin).resolve()
    elif resolved_module is not None:
        resolved_module = resolved_module.resolve()

    direct_source_import = bool(
        resolved_module is not None
        and (resolved_module == source or source in resolved_module.parents)
    )
    direct_url = None if direct_source_import else _installed_direct_url()
    provenance_source = None if direct_url is None else _file_url_path(direct_url)
    provenance_matches = provenance_source == source

    source_package = source / "src" / "fdtdx"
    installed_package = None if resolved_module is None else resolved_module.parent
    source_tree_hash = None
    installed_tree_hash = None
    installed_tree_matches = False
    if direct_source_import:
        installed_tree_matches = True
    elif provenance_matches and installed_package is not None:
        source_tree_hash = _tree_sha256(source_package)
        installed_tree_hash = _tree_sha256(installed_package)
        installed_tree_matches = bool(
            source_tree_hash is not None and source_tree_hash == installed_tree_hash
        )

    import_is_pinned = direct_source_import or (
        provenance_matches and installed_tree_matches
    )
    checks = {
        "source_exists": source.is_dir(),
        "commit_is_pinned": commit == expected_commit,
        "source_tree_is_clean": dirty == "",
        "import_resolves_under_pinned_source": import_is_pinned,
    }
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "expected_commit": expected_commit,
        "observed_commit": commit,
        "source": str(source),
        "resolved_module": None if resolved_module is None else str(resolved_module),
        "direct_source_import": direct_source_import,
        "installed_direct_url": direct_url,
        "installed_provenance_source": (
            None if provenance_source is None else str(provenance_source)
        ),
        "source_package_sha256": source_tree_hash,
        "installed_package_sha256": installed_tree_hash,
        "installed_package_matches_source": installed_tree_matches,
        "error": error,
    }
