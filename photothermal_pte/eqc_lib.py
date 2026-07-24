"""Minimal Lumerical runtime for the certified TaIrTe4 volume-current path."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundle"
WL = 4.0e-6
FIELD_REGION = "cl_fieldregion"
MESH_REFINEMENT = os.environ.get("VC_MESH_REFINEMENT", "precise volume average")
AUTO_SHUTOFF_MIN = float(os.environ.get("VC_AUTO_SHUTOFF_MIN", "1e-8"))
SIM_TIME_S = float(os.environ.get("VC_SIM_TIME_S", "2e-12"))
GPU_DEVICE_DEFAULT = os.environ.get("CL_GPU_DEVICE", "GPU 1")


def _find_r12_root() -> Path:
    configured = os.environ.get("VC_LUMERICAL_ROOT") or os.environ.get("LUMERICAL_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/opt/lumerical/v261"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "api/python/lumapi.py").exists():
            return candidate.resolve()
    checked = ", ".join(str(v) for v in candidates if v is not None)
    raise RuntimeError(
        "Lumerical 2026 R1.02 (v261) was not found. Set VC_LUMERICAL_ROOT. "
        f"Checked: {checked}"
    )


R12_ROOT = _find_r12_root()
R12_API = R12_ROOT / "api/python"
CONTROL_BASE = ROOT / "runs/volume_current_control.fsp"
DATA = ROOT / "runs/data"


def bootstrap_env() -> None:
    """Pin all version/geometry settings before importing lumapi or the model."""
    os.environ.setdefault("EIDL_RUN_DIR", str(ROOT / "runs/model"))
    os.environ["TARGET_WL_UM"] = "4.0"
    os.environ.setdefault("PERIOD_UM", "6.0")
    os.environ["MSOPT_MESH_REFINEMENT_PIN"] = MESH_REFINEMENT
    os.environ["LUMERICAL_ROOT"] = str(R12_ROOT)
    os.environ.setdefault(
        "LUMERICAL_SESSION_GPU_DEVICE",
        os.environ.get("CL_GPU_DEVICE", GPU_DEVICE_DEFAULT),
    )
    os.environ.setdefault("FDTD_THREADS", "8")
    os.environ["ENABLE_LIVE_PLOTTING"] = "0"
    os.environ["PATH"] = f"{R12_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    for value in (str(R12_API), str(BUNDLE)):
        if value not in sys.path:
            sys.path.insert(0, value)


def load_model():
    bootstrap_env()
    import tairte4_volume_model as model
    return model


def epsfield(dataset):
    """Return solver index result as epsilon_r (..., frequency, xyz)."""
    components = []
    for axis in "xyz":
        value = np.asarray(dataset[f"index_{axis}"], np.complex128) ** 2
        if value.ndim == 3:
            value = value[..., None]
        components.append(value)
    return np.stack(components, axis=-1)


def wavelength_contract(fdtd, name):
    values = {}
    for prop in (
        "wavelength start", "wavelength stop", "wavelength center",
        "minimum wavelength", "maximum wavelength",
    ):
        try:
            values[prop] = float(fdtd.getnamed(name, prop))
        except Exception:
            pass
    return values


def assert_4um(values, label):
    for key, value in values.items():
        if not np.isclose(value, WL, rtol=1e-10, atol=1e-15):
            raise RuntimeError(f"{label}.{key}={value:.9e}, expected {WL:.9e}")


def configure_session_resources(fdtd):
    gpu = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", GPU_DEVICE_DEFAULT)
    threads = os.environ.get("FDTD_THREADS", "8")
    try:
        fdtd.setresource("FDTD", 1, "active", 1)
        fdtd.setresource("FDTD", 1, "processes", "1")
        fdtd.setresource("FDTD", 1, "threads", str(threads))
    except Exception:
        pass
    try:
        fdtd.setresource("FDTD", 2, "active", 1)
        fdtd.setresource("FDTD", 2, "processes", "1")
        fdtd.setresource("FDTD", 2, "threads", str(threads))
        fdtd.setresource("FDTD", 2, "device type", gpu)
        fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
    except Exception:
        pass


def run_session(fdtd, run_name):
    errors = []
    for resource in ("Local GPU", "Local Host", "localhost", "Local Computer"):
        try:
            print(f"[run] {run_name} on {resource!r}", flush=True)
            fdtd.run("FDTD", "GPU", resource)
            return resource
        except Exception as exc:
            errors.append(f"{resource}: {exc}")
    raise RuntimeError("session run failed: " + " | ".join(errors))


def run_project(fdtd, run_name, getters, retries=3, retry_delay=15.0):
    """Run the open project and fail closed when any required result is absent."""
    workdir = Path(DATA).parent
    workdir.mkdir(parents=True, exist_ok=True)
    fsp = workdir / f"{run_name}.fsp"
    fdtd.save(str(fsp))
    last_error = None
    for attempt in range(retries):
        try:
            configure_session_resources(fdtd)
            run_session(fdtd, run_name)
            result = {}
            for key, getter in getters.items():
                value = getter(fdtd)
                if value is None:
                    raise RuntimeError(f"getter {key} returned None")
                result[key] = value
            return result
        except Exception as exc:
            last_error = exc
            print(f"[retry] {run_name} attempt {attempt+1}: {exc}", flush=True)
            if attempt >= retries - 1:
                break
            time.sleep(retry_delay)
            try:
                fdtd.switchtolayout()
            except Exception:
                pass
            fdtd.load(str(fsp))
    raise RuntimeError(f"{run_name} failed after {retries} attempts: {last_error}")


def physical_seed(model):
    rho2 = np.asarray(model.x0, float).reshape(model.Nx, model.Ny).copy()
    rho2[-1, :] = rho2[0, :]
    rho2[:, -1] = rho2[:, 0]
    return np.repeat(rho2[..., None], model.Nz, axis=2)


def pin_solver(fdtd):
    fdtd.setnamed("FDTD", "mesh refinement", MESH_REFINEMENT)
    fdtd.setnamed("FDTD", "simulation time", SIM_TIME_S)
    fdtd.setnamed("FDTD", "auto shutoff min", AUTO_SHUTOFF_MIN)


def build_control_base(model, force=False, incident_polarization="x"):
    """Build Case-X forward geometry and add one 3-D FieldRegion."""
    sim = model.build_case(incident_polarization)
    if CONTROL_BASE.exists() and not force:
        try:
            sim.fdtd.close()
        except Exception:
            pass
        return sim
    fdtd = sim.fdtd
    fdtd.switchtolayout()
    pin_solver(fdtd)
    if fdtd.getnamednumber(FIELD_REGION):
        fdtd.select(FIELD_REGION); fdtd.delete()
    fdtd.addfieldregion()
    fdtd.set("name", FIELD_REGION)
    fdtd.set("monitor type", "3D")
    for axis, center, span in zip("xyz", model.fom_c, model.fom_s):
        fdtd.set(axis, center * 1e-6)
        fdtd.set(f"{axis} span", span * 1e-6)
    fdtd.set("override global monitor settings", True)
    fdtd.set("use source limits", False)
    fdtd.set("use wavelength spacing", True)
    fdtd.set("wavelength center", WL)
    fdtd.set("wavelength span", 0.0)
    fdtd.set("frequency points", 1)
    for monitor in (sim.design_monitor_name, sim.design_index_monitor_name):
        try:
            fdtd.setnamed(monitor, "spatial interpolation", "none")
        except Exception:
            pass
        # R1.02 exposes some wavelength controls as inactive depending on the
        # monitor type/current override state. The global single-frequency
        # contract remains authoritative, so unsupported per-monitor controls
        # are intentionally skipped.
        for prop, value in (
            ("override global monitor settings", True),
            ("use source limits", False),
            ("use wavelength spacing", True),
            ("wavelength center", WL),
            ("wavelength span", 0.0),
            ("frequency points", 1),
        ):
            try:
                fdtd.setnamed(monitor, prop, value)
            except Exception:
                pass
    try:
        fdtd.setnamed(sim.design_index_monitor_name, "record conformal mesh when possible", True)
    except Exception:
        pass
    fdtd.setglobalsource("wavelength start", WL)
    fdtd.setglobalsource("wavelength stop", WL)
    assert_4um(wavelength_contract(fdtd, "source"), "forward_source")
    assert_4um(wavelength_contract(fdtd, FIELD_REGION), "fieldregion")
    CONTROL_BASE.parent.mkdir(parents=True, exist_ok=True)
    fdtd.save(str(CONTROL_BASE))
    fdtd.close()
    return sim


def open_control(fsp=None):
    bootstrap_env()
    import lumapi
    path = Path(fsp) if fsp is not None else CONTROL_BASE
    if not path.exists():
        raise RuntimeError(f"missing control project {path}; call prepare() first")
    return lumapi.FDTD(filename=str(path), hide=True)
