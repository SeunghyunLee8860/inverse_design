"""Version-discovered Lumerical DEVICE helpers for photothermal stage 1."""

from __future__ import annotations

import contextlib
import datetime as _datetime
import importlib.util
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import config_stage1 as config


class Stage1Error(RuntimeError):
    """A fail-closed stage-1 validation error."""


class MissingPropertyError(Stage1Error):
    """A required, probed Lumerical property is unavailable."""


def jsonable(value: Any) -> Any:
    """Convert lumapi/numpy values to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "shape": list(value.shape),
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2) + "\n")
    temporary.replace(path)


def utc_timestamp() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_output_directory(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory {path}")
    else:
        path = config.OUTPUT_ROOT / utc_timestamp()
        if path.exists():
            raise FileExistsError(f"timestamp output already exists: {path}")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
    return path


@dataclass(frozen=True)
class LumericalInstallation:
    version_key: str
    root: Path
    lumapi_path: Path
    device_executable: Path


def discover_installations() -> dict[str, LumericalInstallation]:
    found: dict[str, LumericalInstallation] = {}
    for key in config.LUMERICAL_ORDER:
        root = config.LUMERICAL_ROOTS[key]
        api = root / "api" / "python" / "lumapi.py"
        device = root / "bin" / "device"
        if api.is_file() and device.is_file():
            found[key] = LumericalInstallation(key, root.resolve(), api.resolve(), device.resolve())
    return found


def select_installation(requested: str = "auto") -> LumericalInstallation:
    requested = requested.lower()
    found = discover_installations()
    if requested == "auto":
        for key in config.LUMERICAL_ORDER:
            if key in found:
                return found[key]
    elif requested in found:
        return found[requested]
    elif requested not in {"v261", "v251"}:
        raise ValueError("Lumerical version must be auto, v261, or v251")
    raise FileNotFoundError(
        f"requested Lumerical {requested!r} unavailable; found {sorted(found)}"
    )


def load_lumapi(installation: LumericalInstallation):
    module_name = f"stage1_lumapi_{installation.version_key}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, installation.lumapi_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {installation.lumapi_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{installation.root / 'bin'}:{old_path}"
    try:
        spec.loader.exec_module(module)
    finally:
        os.environ["PATH"] = old_path
    if not hasattr(module, "DEVICE"):
        raise ImportError(f"{installation.lumapi_path} has no DEVICE class")
    return module


@contextlib.contextmanager
def open_device(
    installation: LumericalInstallation,
    *,
    hide: bool = True,
) -> Iterator[Any]:
    lumapi = load_lumapi(installation)
    attempts = [
        {"hide": hide, "serverArgs": {"platform": "offscreen"}},
        {"hide": hide},
        {"hide": True},
    ]
    errors: list[str] = []
    session = None
    for kwargs in attempts:
        try:
            session = lumapi.DEVICE(**kwargs)
            break
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    if session is None:
        raise Stage1Error("unable to open DEVICE: " + " | ".join(errors))
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception:
            pass


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.splitlines() if item.strip()]
    array = np.asarray(value, dtype=object).reshape(-1)
    result: list[str] = []
    for item in array:
        text = str(item)
        result.extend(part.strip() for part in text.splitlines() if part.strip())
    return result


def selected_properties(session: Any) -> dict[str, Any]:
    """Read every property of the currently selected object."""
    names = flatten_strings(session.get())
    result: dict[str, Any] = {}
    for name in names:
        try:
            result[name] = jsonable(session.get(name))
        except Exception as exc:
            result[name] = {
                "__read_error__": f"{type(exc).__name__}: {exc}",
            }
    return result


def add_and_probe(
    session: Any,
    command: str,
    *,
    prerequisite_region: bool = False,
) -> dict[str, Any]:
    """Invoke one official add command and capture the selected object."""
    if prerequisite_region:
        try:
            session.addsimulationregion()
        except Exception:
            pass
    try:
        returned = getattr(session, command)()
        properties = selected_properties(session)
        return {
            "command": command,
            "success": True,
            "returned_type": type(returned).__name__,
            "properties": properties,
        }
    except Exception as exc:
        return {
            "command": command,
            "success": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }


def property_lookup(properties: dict[str, Any], *candidates: str) -> str:
    """Resolve an exact probed property using normalized candidate spelling."""
    normalized = {
        " ".join(key.lower().replace("_", " ").split()): key for key in properties
    }
    for candidate in candidates:
        match = normalized.get(" ".join(candidate.lower().replace("_", " ").split()))
        if match is not None:
            return match
    raise MissingPropertyError(
        f"none of properties {candidates!r} were found; available={sorted(properties)}"
    )


def set_selected(
    session: Any,
    properties: dict[str, Any],
    candidates: tuple[str, ...],
    value: Any,
) -> str:
    name = property_lookup(properties, *candidates)
    session.set(name, value)
    return name


def git_environment(repository: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args, cwd=repository, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_porcelain": run("git", "status", "--short"),
    }


def environment_record(
    installation: LumericalInstallation,
    *,
    working_directory: Path,
) -> dict[str, Any]:
    return {
        "selected_version": installation.version_key,
        "lumerical_root": installation.root,
        "lumapi_path": installation.lumapi_path,
        "device_executable": installation.device_executable,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "working_directory": working_directory.resolve(),
        "utc_timestamp": utc_timestamp(),
        "git": git_environment(config.REPOSITORY_ROOT),
    }
