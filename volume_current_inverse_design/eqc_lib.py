"""Minimal Lumerical runtime for the certified TaIrTe4 volume-current path."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundle"
C0 = 299792458.0
WL = 4.0e-6
SOURCE_WL_START = 3.0e-6
SOURCE_WL_STOP = 6.0e-6
MATERIAL_FIT_START = 2.7e-6
MATERIAL_FIT_STOP = 13.2e-6
MATERIAL_SAMPLE_COUNT = 600
FIELD_REGION = "cl_fieldregion"
MATERIAL_NAME = "TaIrTe4_ani"
MESH_TYPE = "auto non-uniform"
MESH_REFINEMENT = "conformal variant 1"
MESH_ACCURACY = 5
FLAKE_DZ_M = 5.0e-9
AUTO_SHUTOFF_MIN = 1.0e-8
SIM_TIME_S = 4.0e-12
EXPECTED_DT_S = 1.5888123913e-17
GPU_DEVICE_DEFAULT = os.environ.get("CL_GPU_DEVICE", "GPU 1")


def _find_r12_root() -> Path:
    configured = os.environ.get("VC_LUMERICAL_ROOT") or os.environ.get("LUMERICAL_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/opt/lumerical/v261"),
        Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261"),
        Path("/home/eidl/lumerical_r12_runtime/opt/lumerical/v261"),
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
    # Production optical contract. These are intentionally separate: source
    # bandwidth, analysis wavelength, and material-fit support are not aliases.
    os.environ["TARGET_WL_UM"] = "4.0"
    os.environ["SOURCE_WL_START_UM"] = "3.0"
    os.environ["SOURCE_WL_STOP_UM"] = "6.0"
    os.environ["MATERIAL_FIT_START_UM"] = "2.7"
    os.environ["MATERIAL_FIT_STOP_UM"] = "13.2"
    os.environ["MATERIAL_SAMPLE_COUNT"] = str(MATERIAL_SAMPLE_COUNT)
    os.environ["BULK_MESH_MODE"] = "auto"
    os.environ["MESH_ACCURACY"] = str(MESH_ACCURACY)
    os.environ["FILM_DZ_NM"] = "5.0"
    os.environ["VC_MESH_REFINEMENT"] = MESH_REFINEMENT
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


def _scalar(value, label):
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: shape={array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def configure_session_resources(fdtd):
    """Configure and read back the exact CPU/GPU resources used by v261."""
    gpu = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", GPU_DEVICE_DEFAULT)
    threads = os.environ.get("FDTD_THREADS", "8")
    fdtd.setresource("FDTD", 1, "active", 1)
    fdtd.setresource("FDTD", 1, "processes", "1")
    fdtd.setresource("FDTD", 1, "threads", str(threads))
    fdtd.setresource("FDTD", 2, "active", 1)
    fdtd.setresource("FDTD", 2, "processes", "1")
    fdtd.setresource("FDTD", 2, "threads", str(threads))
    fdtd.setresource("FDTD", 2, "device type", gpu)
    fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
    return resource_contract(fdtd)


def resource_contract(fdtd, *, validate=True):
    properties = (
        "active", "device type", "processes", "threads",
        "solver extra command line options",
    )
    result = {}
    for index in (1, 2):
        result[str(index)] = {
            prop: str(fdtd.getresource("FDTD", index, prop))
            for prop in properties
        }
    requested_gpu = os.environ.get(
        "LUMERICAL_SESSION_GPU_DEVICE", GPU_DEVICE_DEFAULT
    )
    cpu = result["1"]
    gpu = result["2"]
    if validate and cpu["device type"].strip().upper() != "CPU":
        raise RuntimeError(f"resource 1 is not CPU: {cpu}")
    if validate and gpu["active"].strip() != "1":
        raise RuntimeError(f"GPU resource is inactive: {gpu}")
    if validate and gpu["device type"].strip() != requested_gpu:
        raise RuntimeError(
            f"GPU ID={gpu['device type']!r}, expected {requested_gpu!r}"
        )
    if validate and "-gpu" not in gpu["solver extra command line options"]:
        raise RuntimeError(f"GPU resource lacks -gpu option: {gpu}")
    return result


def _material_contract(fdtd):
    table = np.asarray(fdtd.getmaterial(MATERIAL_NAME, "sampled data"))
    if table.ndim != 2 or table.shape[1] < 4:
        raise RuntimeError(f"invalid {MATERIAL_NAME} sampled data shape {table.shape}")
    wavelengths = C0 / np.asarray(table[:, 0].real, float)
    result = {
        "name": MATERIAL_NAME,
        "sample_count": int(table.shape[0]),
        "wavelength_min_m": float(np.min(wavelengths)),
        "wavelength_max_m": float(np.max(wavelengths)),
    }
    if result["sample_count"] != MATERIAL_SAMPLE_COUNT:
        raise RuntimeError(
            f"material samples={result['sample_count']}, expected {MATERIAL_SAMPLE_COUNT}"
        )
    if not np.isclose(
        result["wavelength_min_m"], MATERIAL_FIT_START, rtol=0, atol=1e-15
    ):
        raise RuntimeError(f"material fit start mismatch: {result}")
    if not np.isclose(
        result["wavelength_max_m"], MATERIAL_FIT_STOP, rtol=0, atol=1e-15
    ):
        raise RuntimeError(f"material fit stop mismatch: {result}")
    return result


def _monitor_contract(fdtd, name):
    if int(fdtd.getnamednumber(name)) != 1:
        raise RuntimeError(f"required monitor {name!r} is missing or duplicated")
    raw_frequency_points = int(round(_scalar(
        fdtd.getnamed(name, "frequency points"), f"{name}.frequency points"
    )))
    wavelengths = wavelength_contract(fdtd, name)
    try:
        use_wavelength_spacing = bool(round(_scalar(
            fdtd.getnamed(name, "use wavelength spacing"),
            f"{name}.use wavelength spacing",
        )))
    except Exception:
        use_wavelength_spacing = True
    try:
        sample_spacing = str(fdtd.getnamed(name, "sample spacing")).strip().lower()
    except Exception:
        sample_spacing = None
    active_custom_spacing = (
        sample_spacing == "custom" and not use_wavelength_spacing
    )
    checked = []
    if not active_custom_spacing:
        for key, value in wavelengths.items():
            if not np.isclose(value, WL, rtol=1e-10, atol=1e-15):
                raise RuntimeError(f"{name}.{key}={value:.9e}, expected {WL:.9e}")
            checked.append(key)
    custom_wavelengths = None
    if sample_spacing == "custom":
        custom_frequencies = np.asarray(
            fdtd.getnamed(name, "custom frequency samples"), float
        ).reshape(-1)
        custom_wavelengths = C0 / custom_frequencies
        if not np.allclose(custom_wavelengths, WL, rtol=1e-10, atol=1e-15):
            raise RuntimeError(
                f"{name}.custom frequency samples are not the 4 um point"
            )
    effective_frequency_points = (
        int(custom_wavelengths.size)
        if custom_wavelengths is not None else raw_frequency_points
    )
    if effective_frequency_points != 1:
        raise RuntimeError(
            f"{name}.effective frequency points={effective_frequency_points}, expected 1"
        )
    if not checked and custom_wavelengths is None:
        raise RuntimeError(f"cannot read wavelength contract for monitor {name!r}")
    return {
        "raw_frequency_points_property": raw_frequency_points,
        "effective_frequency_points": effective_frequency_points,
        "use_wavelength_spacing": use_wavelength_spacing,
        "sample_spacing": sample_spacing,
        "wavelength_properties_m": wavelengths,
        "custom_wavelengths_m": (
            None if custom_wavelengths is None else custom_wavelengths.tolist()
        ),
    }


def assert_production_contract(
    fdtd, monitor_names=(), run_setup=True, validate_resources=True
):
    """Read the realized FSP and fail before a solve on any contract drift."""
    source_named = wavelength_contract(fdtd, "source")
    source_start = _scalar(
        fdtd.getnamed("source", "wavelength start"), "source.wavelength start"
    )
    source_stop = _scalar(
        fdtd.getnamed("source", "wavelength stop"), "source.wavelength stop"
    )
    global_start = _scalar(
        fdtd.getglobalsource("wavelength start"), "global source start"
    )
    global_stop = _scalar(
        fdtd.getglobalsource("wavelength stop"), "global source stop"
    )
    for label, actual, expected in (
        ("source start", source_start, SOURCE_WL_START),
        ("source stop", source_stop, SOURCE_WL_STOP),
        ("global source start", global_start, SOURCE_WL_START),
        ("global source stop", global_stop, SOURCE_WL_STOP),
    ):
        if not np.isclose(actual, expected, rtol=0, atol=1e-15):
            raise RuntimeError(f"{label}={actual:.9e}, expected {expected:.9e}")
    pulse_type = str(fdtd.getnamed("source", "pulse type")).strip().lower()
    if pulse_type != "broadband":
        raise RuntimeError(f"source pulse type={pulse_type!r}, expected 'broadband'")

    mesh_type = str(fdtd.getnamed("FDTD", "mesh type")).strip().lower()
    refinement = str(fdtd.getnamed("FDTD", "mesh refinement")).strip().lower()
    accuracy = int(round(_scalar(
        fdtd.getnamed("FDTD", "mesh accuracy"), "FDTD.mesh accuracy"
    )))
    if mesh_type != MESH_TYPE:
        raise RuntimeError(f"mesh type={mesh_type!r}, expected {MESH_TYPE!r}")
    if refinement != MESH_REFINEMENT:
        raise RuntimeError(
            f"mesh refinement={refinement!r}, expected {MESH_REFINEMENT!r}"
        )
    if accuracy != MESH_ACCURACY:
        raise RuntimeError(f"mesh accuracy={accuracy}, expected {MESH_ACCURACY}")
    global_mesh_count = int(fdtd.getnamednumber("global_uniform_mesh"))
    if global_mesh_count != 0:
        raise RuntimeError(f"global_uniform_mesh count={global_mesh_count}, expected 0")
    if int(fdtd.getnamednumber("flake_mesh")) != 1:
        raise RuntimeError("flake_mesh is missing or duplicated")
    flake_dz = _scalar(fdtd.getnamed("flake_mesh", "dz"), "flake_mesh.dz")
    if not np.isclose(flake_dz, FLAKE_DZ_M, rtol=0, atol=1e-15):
        raise RuntimeError(f"flake dz={flake_dz:.9e}, expected {FLAKE_DZ_M:.9e}")

    version = str(fdtd.version())
    if R12_ROOT.name != "v261" or not version.startswith("8.35"):
        raise RuntimeError(
            f"solver is not the pinned v261 build: root={R12_ROOT}, version={version}"
        )
    resources = resource_contract(fdtd, validate=validate_resources)
    material = _material_contract(fdtd)
    monitors = {name: _monitor_contract(fdtd, name) for name in monitor_names}
    if run_setup:
        fdtd.runsetup()
    dt = _scalar(fdtd.getnamed("FDTD", "dt"), "FDTD.dt")
    if not np.isclose(dt, EXPECTED_DT_S, rtol=5e-10, atol=1e-24):
        raise RuntimeError(f"FDTD.dt={dt:.13e}, expected {EXPECTED_DT_S:.13e}")
    return {
        "source": {
            "wavelength_start_m": source_start,
            "wavelength_stop_m": source_stop,
            "global_wavelength_start_m": global_start,
            "global_wavelength_stop_m": global_stop,
            "pulse_type": pulse_type,
            "named_properties_m": source_named,
        },
        "analysis_wavelength_m": WL,
        "material": material,
        "monitors": monitors,
        "mesh": {
            "type": mesh_type,
            "refinement": refinement,
            "accuracy": accuracy,
            "global_uniform_mesh_count": global_mesh_count,
            "flake_dz_m": flake_dz,
            "dt_s": dt,
        },
        "solver": {
            "root": str(R12_ROOT),
            "version": version,
            "resources": resources,
        },
    }


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
    """Assert the realized FSP, then run; configuration drift is not retried."""
    workdir = Path(DATA).parent
    workdir.mkdir(parents=True, exist_ok=True)
    fsp = workdir / f"{run_name}.fsp"
    required_monitors = tuple(
        name for name in (
            FIELD_REGION, "design_monitor", "design_index_monitor"
        ) if int(fdtd.getnamednumber(name)) > 0
    )
    configure_session_resources(fdtd)
    assert_production_contract(fdtd, required_monitors, run_setup=True)
    fdtd.save(str(fsp))

    last_error = None
    for attempt in range(retries):
        try:
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
            fdtd.switchtolayout()
            fdtd.load(str(fsp))
            configure_session_resources(fdtd)
            assert_production_contract(fdtd, required_monitors, run_setup=True)
    raise RuntimeError(f"{run_name} failed after {retries} attempts: {last_error}")


def physical_seed(model, beta=1.0):
    """Physical (Nx,Ny,Nz) density from the model seed.

    Handles both representations: the production 240x240 unique latent (mapped
    to physical through model.mapping) and the legacy 241x241 latent (fencepost
    wrapped + z-extruded).  Fixes the 240-grid reshape crash.
    """
    x0 = np.asarray(model.x0, float).reshape(-1)
    nux = int(model.Nx) - 1
    nuy = int(model.Ny) - 1
    if x0.size == nux * nuy:
        return np.asarray(
            model.mapping(x0, beta), float
        ).reshape(model.Nx, model.Ny, model.Nz)
    rho2 = x0.reshape(model.Nx, model.Ny).copy()
    rho2[-1, :] = rho2[0, :]
    rho2[:, -1] = rho2[:, 0]
    return np.repeat(rho2[..., None], model.Nz, axis=2)


def pin_solver(fdtd):
    fdtd.setnamed("FDTD", "mesh type", MESH_TYPE)
    fdtd.setnamed("FDTD", "mesh refinement", MESH_REFINEMENT)
    fdtd.setnamed("FDTD", "mesh accuracy", MESH_ACCURACY)
    fdtd.setnamed("FDTD", "simulation time", SIM_TIME_S)
    fdtd.setnamed("FDTD", "auto shutoff min", AUTO_SHUTOFF_MIN)
    if int(fdtd.getnamednumber("global_uniform_mesh")) != 0:
        raise RuntimeError("global_uniform_mesh exists in the production FSP")


def build_control_base(model, force=False, incident_polarization="x"):
    """Build Case-X forward geometry and add one 3-D FieldRegion."""
    sim = model.build_case(incident_polarization)
    if CONTROL_BASE.exists() and not force:
        try:
            sim.fdtd.close()
        except Exception:
            pass
        with open_control(CONTROL_BASE) as existing:
            configure_session_resources(existing)
            assert_production_contract(
                existing,
                (FIELD_REGION, sim.design_monitor_name, sim.design_index_monitor_name),
                run_setup=True,
            )
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
    fdtd.setglobalmonitor("use source limits", False)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("wavelength center", WL)
    fdtd.setglobalmonitor("wavelength span", 0.0)
    fdtd.setglobalmonitor("frequency points", 1)
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
    # The design DFT monitor exposes an inactive raw "frequency points"
    # property when custom spacing is selected. Pin the actual custom sample
    # explicitly to the one 4 um analysis frequency.
    fdtd.setnamed(sim.design_monitor_name, "use wavelength spacing", False)
    fdtd.setnamed(sim.design_monitor_name, "sample spacing", "custom")
    fdtd.setnamed(
        sim.design_monitor_name,
        "custom frequency samples",
        np.asarray([C0 / WL]),
    )
    try:
        fdtd.setnamed(sim.design_index_monitor_name, "record conformal mesh when possible", True)
    except Exception:
        pass
    fdtd.setnamed("source", "use global source settings", True)
    fdtd.setnamed("source", "override global source settings", False)
    fdtd.setglobalsource("wavelength start", SOURCE_WL_START)
    fdtd.setglobalsource("wavelength stop", SOURCE_WL_STOP)
    configure_session_resources(fdtd)
    assert_production_contract(
        fdtd,
        (FIELD_REGION, sim.design_monitor_name, sim.design_index_monitor_name),
        run_setup=True,
    )
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
