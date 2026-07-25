"""Reusable coherent-FOM evaluator using a native FieldRegion volume current.

One gradient evaluation performs one forward EM solve, two component-separated
volume-current adjoint solves (Ex and Ez), and 27 colored *layout-only* probes
of Lumerical's actual conformal rho->epsilon map.  No dipole object is created.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import epsilon_0

import eqc_lib as lib
from collocated_coherent_fom import (
    collocated_coherent_fom_and_wirtinger,
    collocated_coherent_fom_and_wirtinger_y,
    fieldregion_periodic_source_right_inverse,
)
from volume_current_adjoint_core import as_e5, fieldregion_source_profile
from volume_current_colored_jacobian import (
    accumulate_color_transpose,
    color_direction,
    colors,
    expand_periodic_unique,
    owner_axes,
    unique_from_periodic,
)
from volume_current_yee_metric import component_periodic_yee_volumes


SOURCE_NAME = "source"


@dataclass
class Evaluation:
    fom: float
    gradient_physical: np.ndarray
    metadata: dict[str, Any]


def _set_nuttall(fdtd, enabled: bool = False) -> None:
    try:
        fdtd.setnamed(lib.FIELD_REGION, "nuttall window pulse", bool(enabled))
    except Exception as exc:
        raise RuntimeError(f"cannot pin FieldRegion Nuttall pulse: {exc}") from exc


def _e5(dataset: dict[str, Any]) -> np.ndarray:
    return as_e5(np.asarray(dataset["E"], np.complex128))


def _eps5(dataset: dict[str, Any]) -> np.ndarray:
    return lib.epsfield(dataset)


def import_fieldregion_profile(fdtd, grid: dict[str, np.ndarray], profile: np.ndarray) -> float:
    """Import one vector volume-current profile and verify an exact round trip."""
    value = np.ascontiguousarray(as_e5(profile))
    expected = (grid["x"].size, grid["y"].size, grid["z"].size, 1, 3)
    if value.shape != expected:
        raise ValueError(f"profile {value.shape} != FieldRegion grid {expected}")
    fdtd.putv("vc_x", np.asarray(grid["x"], float))
    fdtd.putv("vc_y", np.asarray(grid["y"], float))
    fdtd.putv("vc_z", np.asarray(grid["z"], float))
    fdtd.putv("vc_f", float(np.asarray(grid["f"]).reshape(-1)[0]))
    fdtd.putv("vc_Ex", np.ascontiguousarray(value[..., 0]))
    fdtd.putv("vc_Ey", np.ascontiguousarray(value[..., 1]))
    fdtd.putv("vc_Ez", np.ascontiguousarray(value[..., 2]))
    fdtd.eval(
        'vc_ds=rectilineardataset("EM fields",vc_x,vc_y,vc_z);'
        'vc_ds.addparameter("lambda",c/vc_f,"f",vc_f);'
        'vc_ds.addattribute("E",vc_Ex,vc_Ey,vc_Ez);'
        f'select("{lib.FIELD_REGION}");importdataset(vc_ds);'
    )
    back = _e5(fdtd.getresult(lib.FIELD_REGION, "source profile"))
    error = float(np.max(np.abs(back - value)))
    if error != 0.0:
        raise RuntimeError(f"FieldRegion profile round-trip error {error:.3e}")
    return error


class VolumeCurrentEvaluator:
    """One-channel Ex/Ez or Ey/Ez production evaluator.

    The returned gradient is with respect to the full periodic physical density
    array (Nx, Ny, Nz).  It is a valid representative for the mapping VJP:
    unique-node derivatives occupy ``[:-1, :-1, :]`` and fencepost entries are
    zero, so the mapping's exact seam transpose performs the required sum.
    """

    def __init__(
        self, workdir: str | Path | None = None, rho_step: float = 0.0025,
        incident_polarization: str = "x",
    ):
        self.root = Path(workdir or lib.ROOT / "runs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(exist_ok=True)
        if incident_polarization not in ("x", "y"):
            raise ValueError("incident_polarization must be x or y")
        self.incident_polarization = incident_polarization
        self.control_base = self.root / f"volume_current_control_{incident_polarization}.fsp"
        self._activate_paths()
        self.model = lib.load_model()
        self.rho_step = float(rho_step)
        # Solver-safe affine density layer.  The geometry mapping may output
        # rho_geom exactly at the 0/1 rails; during gradient evaluation we map it
        # to rho_solver = delta + (1-2*delta)*rho_geom so the centered 27-color
        # probe rho_solver +/- rho_step always stays strictly inside [0,1].  The
        # returned gradient is chain-ruled by (1-2*delta).  This decouples
        # manufacturing geometry (rails allowed) from Jacobian-probe safety and
        # removes the beta/rail deadlock (no margin gate, no step halving).
        self.safety_slack = float(
            os.environ.get("VC_SAFETY_SLACK", max(1e-6, 100 * np.finfo(float).eps))
        )
        self.delta = self.rho_step + self.safety_slack
        self.sim = None
        self._evaluation_index = 0
        self._assert_static_contract()

    def _activate_paths(self) -> None:
        lib.DATA = self.data_dir
        lib.CONTROL_BASE = self.control_base

    @property
    def physical_shape(self) -> tuple[int, int, int]:
        return (int(self.model.Nx), int(self.model.Ny), int(self.model.Nz))

    def _assert_static_contract(self) -> None:
        if not np.isclose(float(self.model.period_xy), 6.0):
            raise RuntimeError(f"period={self.model.period_xy} um; expected 6 um")
        if not np.allclose(np.asarray(self.model.target_wl, float), [4.0]):
            raise RuntimeError(f"wavelength={self.model.target_wl}; expected [4.0] um")
        if self.physical_shape != (241, 241, 13):
            raise RuntimeError(f"design grid={self.physical_shape}; expected (241,241,13)")
        if self.rho_step <= 0:
            raise ValueError("rho_step must be positive")

    def prepare(self, force_rebuild: bool = False):
        self._activate_paths()
        self.sim = lib.build_control_base(
            self.model, force=force_rebuild,
            incident_polarization=self.incident_polarization,
        )
        with lib.open_control(self.control_base) as fdtd:
            lib.configure_session_resources(fdtd)
            contract = lib.assert_production_contract(
                fdtd,
                (
                    lib.FIELD_REGION,
                    self.sim.design_monitor_name,
                    self.sim.design_index_monitor_name,
                ),
                run_setup=True,
            )
            px = float(fdtd.getnamed("FDTD", "x span"))
            py = float(fdtd.getnamed("FDTD", "y span"))
        if not np.allclose([px, py], [6e-6, 6e-6], rtol=0, atol=1e-14):
            raise RuntimeError(f"FDTD periodic span mismatch: {(px, py)}")
        return contract

    def _ensure_prepared(self) -> None:
        self._activate_paths()
        if self.sim is None:
            self.prepare()

    def _set_forward_state(self, fdtd, rho: np.ndarray) -> None:
        fdtd.switchtolayout()
        fdtd.setnamed(lib.FIELD_REGION, "source mode", False)
        fdtd.setnamed(SOURCE_NAME, "enabled", True)
        self.sim.fdtd = fdtd
        self.sim.update_design_density(np.asarray(rho, float))
        lib.pin_solver(fdtd)

    def _forward(self, fdtd, rho: np.ndarray, label: str, need_design: bool):
        self._set_forward_state(fdtd, rho)
        getters = {"fom": lambda f: f.getresult(lib.FIELD_REGION, "E")}
        if need_design:
            getters.update({
                "design": lambda f: f.getresult(self.sim.design_monitor_name, "E"),
                "index": lambda f: f.getresult(self.sim.design_index_monitor_name, "index"),
                "delta_x": lambda f: np.asarray(
                    f.getdata(self.sim.design_monitor_name, "delta_x"), float
                ).reshape(-1),
                "delta_y": lambda f: np.asarray(
                    f.getdata(self.sim.design_monitor_name, "delta_y"), float
                ).reshape(-1),
                "delta_z": lambda f: np.asarray(
                    f.getdata(self.sim.design_monitor_name, "delta_z"), float
                ).reshape(-1),
            })
        result = lib.run_project(fdtd, label, getters)
        fom_function = (
            collocated_coherent_fom_and_wirtinger
            if self.incident_polarization == "x"
            else collocated_coherent_fom_and_wirtinger_y
        )
        fom, overlap, q, contract = fom_function(
            _e5(result["fom"]), result["fom"]["x"], result["fom"]["y"],
            result["fom"]["z"]
        )
        return fom, overlap, q, contract, result

    def forward_fom(self, rho: np.ndarray, label: str = "forward_only") -> float:
        """Run only the physical forward problem; used by independent FD."""
        self._ensure_prepared()
        rho = self._validate_rho(rho, require_probe_margin=False)
        with lib.open_control(self.control_base) as fdtd:
            fom, _, _, _, _ = self._forward(fdtd, rho, label, need_design=False)
        return float(fom)

    def _adjoint(self, fdtd, grid, q_source, component: int, label: str):
        component_q = np.zeros_like(q_source)
        component_q[..., component] = q_source[..., component]
        profile, scale = fieldregion_source_profile(component_q)
        fdtd.switchtolayout()
        fdtd.setnamed(SOURCE_NAME, "enabled", False)
        fdtd.setnamed(lib.FIELD_REGION, "source mode", True)
        fdtd.setnamed("FDTD", "simulation time", lib.SIM_TIME_S)
        fdtd.setnamed("FDTD", "auto shutoff min", lib.AUTO_SHUTOFF_MIN)
        _set_nuttall(fdtd, False)
        roundtrip = import_fieldregion_profile(fdtd, grid, profile)
        base_amplitude = float(fdtd.getnamed(lib.FIELD_REGION, "base amplitude"))
        result = lib.run_project(
            fdtd, label,
            {"design": lambda f: f.getresult(self.sim.design_monitor_name, "E")},
        )
        return _e5(result["design"]), scale, base_amplitude, roundtrip

    def _validate_rho(self, rho: np.ndarray, require_probe_margin: bool) -> np.ndarray:
        value = np.asarray(rho, float)
        if value.shape != self.physical_shape:
            if value.size != int(np.prod(self.physical_shape)):
                raise ValueError(f"rho shape={value.shape}, expected {self.physical_shape}")
            value = value.reshape(self.physical_shape)
        unique_from_periodic(value)
        if np.min(value) < 0 or np.max(value) > 1:
            raise ValueError("rho is outside [0,1]")
        if require_probe_margin and (
            np.min(value) < self.rho_step or np.max(value) > 1 - self.rho_step
        ):
            raise ValueError(
                "centered measured rho->epsilon Jacobian needs rho in "
                f"[{self.rho_step}, {1-self.rho_step}]; reduce VC_RHO_STEP or beta"
            )
        return value

    def _measured_rho_transpose(self, fdtd, rho, fine_sensitivity, index0):
        unique = unique_from_periodic(rho)
        gradient = np.zeros_like(unique, dtype=float)
        base_eps = _eps5(index0)
        axes = {a: np.asarray(index0[a], float).reshape(-1) for a in "xyz"}
        design_z = np.asarray(self.sim.design_z, float).reshape(-1)
        max_owner_leakage = 0.0
        try:
            fdtd.switchtolayout()
            self.sim.fdtd = fdtd
            for color in colors():
                direction = color_direction(unique.shape, color)
                pair = []
                for sign in (1.0, -1.0):
                    self.sim.update_design_density(rho + sign * self.rho_step * direction)
                    index = fdtd.getresult(self.sim.design_index_monitor_name, "index")
                    for axis in "xyz":
                        if not np.array_equal(axes[axis], np.asarray(index[axis]).reshape(-1)):
                            raise RuntimeError(f"index {axis} grid changed for color {color}")
                    pair.append(_eps5(index))
                deps = (pair[0] - pair[1]) / (2 * self.rho_step)
                if deps.shape != base_eps.shape or deps.shape != fine_sensitivity.shape:
                    raise RuntimeError(
                        f"epsilon/sensitivity mismatch: {deps.shape}, "
                        f"{fine_sensitivity.shape}"
                    )
                owners = owner_axes(
                    unique.shape, deps.shape[:3], design_z, axes["z"], color
                )
                before = np.array(gradient, copy=True)
                accumulate_color_transpose(
                    gradient, fine_sensitivity, deps, owners
                )
                selected = direction[:-1, :-1].astype(bool)
                delta = gradient - before
                leakage = float(np.sum(np.abs(delta[~selected])))
                signal = float(np.sum(np.abs(delta[selected])))
                max_owner_leakage = max(
                    max_owner_leakage,
                    leakage / max(signal + leakage, 1e-300),
                )
        finally:
            self.sim.update_design_density(rho)
        return gradient, max_owner_leakage

    def value_and_gradient(
        self, rho: np.ndarray, label: str | None = None,
        density_mode: str = "probe_safe",
    ) -> Evaluation:
        """Return coherent FOM and exact-chain physical-density gradient.

        ``rho`` is the geometry (mapping) density and MAY sit exactly at the 0/1
        rails.  In ``probe_safe`` mode (default) it is mapped through the affine
        solver-safe layer rho_solver = delta + (1-2*delta)*rho_geom so the
        centered Jacobian probe never leaves [0,1]; the returned gradient is
        chain-ruled back to rho_geom by (1-2*delta).  No margin gate, no step
        halving.  ``exact`` mode (delta=0) requires a probe-margin-safe input and
        is only for validation against ``forward_fom``.
        """
        self._ensure_prepared()
        rho_geom = self._validate_rho(rho, require_probe_margin=False)
        if density_mode == "probe_safe":
            delta = self.delta
        elif density_mode == "exact":
            delta = 0.0
            self._validate_rho(rho_geom, require_probe_margin=True)
        else:
            raise ValueError(f"density_mode={density_mode!r} must be probe_safe or exact")
        chain = 1.0 - 2.0 * delta
        rho = delta + chain * rho_geom
        self._evaluation_index += 1
        prefix = label or f"eval_{self._evaluation_index:04d}"
        with lib.open_control(self.control_base) as fdtd:
            fom, overlap, q_observation, collocation, forward = self._forward(
                fdtd, rho, f"{prefix}_forward", need_design=True
            )
            q_source = fieldregion_periodic_source_right_inverse(q_observation)
            pairing = abs(
                np.vdot(q_source, _e5(forward["fom"]))
                - np.vdot(q_observation, _e5(forward["fom"]))
            ) / max(abs(np.vdot(q_observation, _e5(forward["fom"]))), 1e-300)
            if pairing > 1e-13:
                raise RuntimeError(f"periodic source pairing error {pairing:.3e}")
            grid = {
                a: np.asarray(forward["fom"][a]).reshape(-1) for a in "xyzf"
            }
            adjoints = {}
            source_meta = {}
            inplane_component = 0 if self.incident_polarization == "x" else 1
            inplane_name = self.incident_polarization
            for name, component in ((inplane_name, inplane_component), ("z", 2)):
                ea, scale, amp, roundtrip = self._adjoint(
                    fdtd, grid, q_source, component, f"{prefix}_adjoint_{name}"
                )
                adjoints[name] = ea
                source_meta[name] = {
                    "component": component,
                    "profile_scale": scale,
                    "base_amplitude": amp,
                    "roundtrip_max_abs_error": roundtrip,
                }

            ef = _e5(forward["design"])
            volumes = component_periodic_yee_volumes(
                forward["delta_x"], forward["delta_y"], forward["delta_z"]
            )
            fine_sensitivity = np.zeros_like(ef)
            for name in (inplane_name, "z"):
                meta = source_meta[name]
                ea = adjoints[name] * meta["profile_scale"]
                for c in range(3):
                    fine_sensitivity[..., c] += (
                        (2 * epsilon_0 / meta["base_amplitude"])
                        * volumes[c][..., None] * ef[..., c] * ea[..., c]
                    )
            fdtd.switchtolayout()
            gradient_unique, leakage = self._measured_rho_transpose(
                fdtd, rho, fine_sensitivity, forward["index"]
            )

        gradient_full = np.zeros(self.physical_shape, float)
        # chain rule for the affine solver-safe layer: dF/drho_geom = (1-2*delta)*dF/drho_solver
        gradient_full[:-1, :-1, :] = gradient_unique * chain
        metadata = {
            "source_type": "FieldRegion volumetric current only; zero dipoles",
            "incident_polarization": self.incident_polarization,
            "density_mode": density_mode,
            "solver_safe_affine": {
                "rho_step": self.rho_step,
                "safety_slack": self.safety_slack,
                "delta": delta,
                "chain_factor": chain,
                "formula": "rho_solver = delta + (1-2*delta)*rho_geom",
                "note": (
                    "optimisation FOM/gradient use probe-safe density; the exact "
                    "binary FOM is a separate forward_fom(delta=0) evaluation"
                ),
            },
            "fom": (
                f"|integral E{self.incident_polarization}_collocated "
                "* conj(Ez_collocated) dV|^2"
            ),
            "overlap": [float(np.real(overlap[0])), float(np.imag(overlap[0]))],
            "periodic_source_pairing_relative_error": float(pairing),
            "collocation": collocation,
            "source": source_meta,
            "rho_epsilon_transpose": {
                "method": "27-color measured centered layout-only Jacobian transpose",
                "step": self.rho_step,
                "electromagnetic_solves": 0,
                "max_owner_leakage_fraction": leakage,
            },
        }
        return Evaluation(float(fom), gradient_full, metadata)
