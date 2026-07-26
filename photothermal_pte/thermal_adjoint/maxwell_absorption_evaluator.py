"""v261 FieldRegion adjoint for a weighted TaIrTe4 absorption objective.

This reuses the already certified solver launch, source-profile round trip,
Yee metric, and 27-color measured conformal rho->epsilon transpose.  Only the
observation functional and its literal Wirtinger source differ from the
coherent-FOM evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.constants import c, epsilon_0

import eqc_lib as lib
from collocated_coherent_fom import fieldregion_periodic_source_right_inverse
from volume_current_adjoint_core import as_e5
from volume_current_colored_jacobian import unique_from_periodic
from volume_current_evaluator import (
    Evaluation,
    SOURCE_NAME,
    VolumeCurrentEvaluator,
)
from volume_current_yee_metric import component_periodic_yee_volumes

from yee_absorption_functional import (
    YeeAbsorptionEvaluation,
    component_shifted_trapezoid_volumes,
    weighted_yee_absorption_and_wirtinger,
)


WeightBuilder = Callable[
    [dict[str, np.ndarray], dict[int, np.ndarray]], np.ndarray
]


@dataclass(frozen=True)
class AbsorptionForward:
    value: float
    observation: YeeAbsorptionEvaluation
    q_observation: np.ndarray
    weight: np.ndarray
    field_result: dict[str, Any]
    incident_intensity_W_m2: float
    epsilon_imaginary: np.ndarray


def periodic_smooth_native_weight(
    grid: dict[str, np.ndarray],
    component_volumes: dict[int, np.ndarray],
) -> np.ndarray:
    """Deterministic nonconstant certificate weight on the native Yee grid."""
    x = np.asarray(grid["x"], float).reshape(-1)
    y = np.asarray(grid["y"], float).reshape(-1)
    z = np.asarray(grid["z"], float).reshape(-1)
    lx = float(x[-1] - x[0])
    ly = float(y[-1] - y[0])
    x_phase = 2.0 * np.pi * (x - x[0]) / lx
    y_phase = 2.0 * np.pi * (y - y[0]) / ly
    z_fraction = (z - z[0]) / max(float(z[-1] - z[0]), np.finfo(float).tiny)
    base = (
        1.0
        + 0.23 * np.sin(x_phase)[:, None, None]
        - 0.17 * np.cos(y_phase)[None, :, None]
        + 0.11 * z_fraction[None, None, :]
    )
    weight = np.empty((*base.shape, 3), float)
    weight[..., 0] = base
    weight[..., 1] = 0.83 * base + 0.07 * np.cos(x_phase)[:, None, None]
    weight[..., 2] = 0.0
    return weight


class AbsorptionVolumeCurrentEvaluator(VolumeCurrentEvaluator):
    """Physical-density evaluator for frozen native-Yee absorption weights."""

    @staticmethod
    def _fitted_epsilon_imaginary(fdtd, frequency_Hz: float) -> np.ndarray:
        wavelength_bounds = np.asarray(
            [
                float(fdtd.getnamed(SOURCE_NAME, "wavelength start")),
                float(fdtd.getnamed(SOURCE_NAME, "wavelength stop")),
            ]
        )
        frequency_bounds = c / wavelength_bounds
        values = []
        for component in (1, 2, 3):
            refractive_index = np.asarray(
                fdtd.getfdtdindex(
                    lib.MATERIAL_NAME,
                    np.asarray([frequency_Hz]),
                    float(np.min(frequency_bounds)),
                    float(np.max(frequency_bounds)),
                    component,
                )
            ).reshape(-1)[0]
            values.append(float(np.imag(complex(refractive_index) ** 2)))
        result = np.asarray(values, float)
        if not np.all(np.isfinite(result)) or np.any(result < 0.0):
            raise RuntimeError("invalid fitted TaIrTe4 loss from v261")
        return result

    def _absorption_from_result(
        self,
        *,
        result: dict[str, Any],
        incident_intensity_W_m2: float,
        weight_builder: WeightBuilder,
    ) -> AbsorptionForward:
        grid = {
            axis: np.asarray(result["fom"][axis]).reshape(-1)
            for axis in "xyzf"
        }
        volumes = component_shifted_trapezoid_volumes(
            x_m=grid["x"],
            y_m=grid["y"],
            z_m=grid["z"],
            delta_x_m=result["fom_delta_x"],
            delta_y_m=result["fom_delta_y"],
            delta_z_m=result["fom_delta_z"],
        )
        weight = np.asarray(weight_builder(grid, volumes), float)
        if "epsilon_imaginary" not in result:
            raise RuntimeError("missing solver-fitted epsilon imaginary parts")
        epsilon_imaginary = np.asarray(result["epsilon_imaginary"], float)
        observation = weighted_yee_absorption_and_wirtinger(
            electric_field_V_m=as_e5(result["fom"]["E"]),
            frequency_Hz=float(grid["f"][0]),
            epsilon_imaginary=epsilon_imaginary,
            component_volume_m3=volumes,
            density_weight=weight,
            power_scale=1.0 / incident_intensity_W_m2,
        )
        return AbsorptionForward(
            value=observation.weighted_value,
            observation=observation,
            q_observation=observation.wirtinger_source,
            weight=weight,
            field_result=result,
            incident_intensity_W_m2=incident_intensity_W_m2,
            epsilon_imaginary=epsilon_imaginary,
        )

    def _forward_absorption(
        self,
        fdtd,
        rho: np.ndarray,
        label: str,
        *,
        need_design: bool,
        weight_builder: WeightBuilder,
    ) -> AbsorptionForward:
        self._set_forward_state(fdtd, rho)
        source_amplitude_V_m = float(fdtd.getnamed(SOURCE_NAME, "amplitude"))
        incident_intensity = (
            0.5 * epsilon_0 * c * source_amplitude_V_m**2
        )
        analysis_frequency = c / (
            float(np.asarray(self.model.target_wl).reshape(-1)[0]) * 1e-6
        )
        fitted_epsilon_imaginary = self._fitted_epsilon_imaginary(
            fdtd, analysis_frequency
        )
        getters: dict[str, Callable] = {
            "fom": lambda f: f.getresult(lib.FIELD_REGION, "E"),
            "fom_delta_x": lambda f: np.asarray(
                f.getresult(lib.FIELD_REGION, "delta_x"), float
            ).reshape(-1),
            "fom_delta_y": lambda f: np.asarray(
                f.getresult(lib.FIELD_REGION, "delta_y"), float
            ).reshape(-1),
            "fom_delta_z": lambda f: np.asarray(
                f.getresult(lib.FIELD_REGION, "delta_z"), float
            ).reshape(-1),
            "epsilon_imaginary": lambda f: fitted_epsilon_imaginary,
        }
        if need_design:
            getters.update(
                {
                    "design": lambda f: f.getresult(
                        self.sim.design_monitor_name, "E"
                    ),
                    "index": lambda f: f.getresult(
                        self.sim.design_index_monitor_name, "index"
                    ),
                    "delta_x": lambda f: np.asarray(
                        f.getdata(self.sim.design_monitor_name, "delta_x"), float
                    ).reshape(-1),
                    "delta_y": lambda f: np.asarray(
                        f.getdata(self.sim.design_monitor_name, "delta_y"), float
                    ).reshape(-1),
                    "delta_z": lambda f: np.asarray(
                        f.getdata(self.sim.design_monitor_name, "delta_z"), float
                    ).reshape(-1),
                }
            )
        result = lib.run_project(fdtd, label, getters)
        return self._absorption_from_result(
            result=result,
            incident_intensity_W_m2=incident_intensity,
            weight_builder=weight_builder,
        )

    def postprocess_completed_forward(
        self,
        project_path,
        *,
        weight_builder: WeightBuilder = periodic_smooth_native_weight,
    ) -> AbsorptionForward:
        """Read one completed FSP without launching another EM solve."""
        self._ensure_prepared()
        with lib.open_control(project_path) as fdtd:
            source_amplitude_V_m = float(
                fdtd.getnamed(SOURCE_NAME, "amplitude")
            )
            incident_intensity = (
                0.5 * epsilon_0 * c * source_amplitude_V_m**2
            )
            field_result = fdtd.getresult(lib.FIELD_REGION, "E")
            frequency = float(
                np.asarray(field_result["f"]).reshape(-1)[0]
            )
            result = {
                "fom": field_result,
                "fom_delta_x": np.asarray(
                    fdtd.getresult(lib.FIELD_REGION, "delta_x"), float
                ).reshape(-1),
                "fom_delta_y": np.asarray(
                    fdtd.getresult(lib.FIELD_REGION, "delta_y"), float
                ).reshape(-1),
                "fom_delta_z": np.asarray(
                    fdtd.getresult(lib.FIELD_REGION, "delta_z"), float
                ).reshape(-1),
                "epsilon_imaginary": self._fitted_epsilon_imaginary(
                    fdtd, frequency
                ),
            }
        return self._absorption_from_result(
            result=result,
            incident_intensity_W_m2=incident_intensity,
            weight_builder=weight_builder,
        )

    def forward_absorption(
        self,
        rho: np.ndarray,
        *,
        label: str = "absorption_forward_only",
        density_mode: str = "exact",
        weight_builder: WeightBuilder = periodic_smooth_native_weight,
    ) -> float:
        """Forward weighted absorption for an independent density FD."""
        self._ensure_prepared()
        rho_geom = self._validate_rho(rho, require_probe_margin=False)
        if density_mode == "probe_safe":
            delta = self.delta
        elif density_mode == "exact":
            delta = 0.0
        else:
            raise ValueError("density mode must be probe_safe or exact")
        rho_solver = delta + (1.0 - 2.0 * delta) * rho_geom
        with lib.open_control(self.control_base) as fdtd:
            forward = self._forward_absorption(
                fdtd,
                rho_solver,
                label,
                need_design=False,
                weight_builder=weight_builder,
            )
        return float(forward.value)

    def value_and_gradient_absorption(
        self,
        rho: np.ndarray,
        *,
        label: str | None = None,
        density_mode: str = "probe_safe",
        weight_builder: WeightBuilder = periodic_smooth_native_weight,
    ) -> Evaluation:
        """Return weighted absorption and its physical-density gradient."""
        self._ensure_prepared()
        rho_geom = self._validate_rho(rho, require_probe_margin=False)
        if density_mode == "probe_safe":
            delta = self.delta
        elif density_mode == "exact":
            delta = 0.0
            self._validate_rho(rho_geom, require_probe_margin=True)
        else:
            raise ValueError("density mode must be probe_safe or exact")
        chain = 1.0 - 2.0 * delta
        rho_solver = delta + chain * rho_geom
        self._evaluation_index += 1
        prefix = label or f"absorption_eval_{self._evaluation_index:04d}"

        with lib.open_control(self.control_base) as fdtd:
            forward = self._forward_absorption(
                fdtd,
                rho_solver,
                f"{prefix}_forward",
                need_design=True,
                weight_builder=weight_builder,
            )
            q_source = fieldregion_periodic_source_right_inverse(
                forward.q_observation
            )
            field = as_e5(forward.field_result["fom"]["E"])
            observed_pairing = np.vdot(forward.q_observation, field)
            source_pairing = np.vdot(q_source, field)
            pairing = abs(source_pairing - observed_pairing) / max(
                abs(observed_pairing), 1e-300
            )
            if pairing > 1e-13:
                raise RuntimeError(
                    f"periodic source pairing error {pairing:.3e}"
                )
            grid = {
                axis: np.asarray(forward.field_result["fom"][axis]).reshape(-1)
                for axis in "xyzf"
            }
            active_components = [
                component
                for component in range(3)
                if np.linalg.norm(q_source[..., component]) > 0.0
            ]
            adjoints = {}
            source_meta = {}
            for component in active_components:
                name = "xyz"[component]
                ea, scale, amplitude, roundtrip = self._adjoint(
                    fdtd,
                    grid,
                    q_source,
                    component,
                    f"{prefix}_adjoint_{name}",
                )
                adjoints[component] = ea
                source_meta[name] = {
                    "component": component,
                    "profile_scale": scale,
                    "base_amplitude": amplitude,
                    "roundtrip_max_abs_error": roundtrip,
                }

            design_forward = as_e5(forward.field_result["design"])
            design_volumes = component_periodic_yee_volumes(
                forward.field_result["delta_x"],
                forward.field_result["delta_y"],
                forward.field_result["delta_z"],
            )
            fine_sensitivity = np.zeros_like(design_forward)
            for component in active_components:
                name = "xyz"[component]
                metadata = source_meta[name]
                adjoint = (
                    adjoints[component] * metadata["profile_scale"]
                )
                for epsilon_component in range(3):
                    fine_sensitivity[..., epsilon_component] += (
                        2.0
                        * epsilon_0
                        / metadata["base_amplitude"]
                        * design_volumes[epsilon_component][..., None]
                        * design_forward[..., epsilon_component]
                        * adjoint[..., epsilon_component]
                    )
            fdtd.switchtolayout()
            gradient_unique, leakage = self._measured_rho_transpose(
                fdtd,
                rho_solver,
                fine_sensitivity,
                forward.field_result["index"],
            )

        gradient_full = np.zeros(self.physical_shape, float)
        gradient_full[:-1, :-1, :] = gradient_unique * chain
        metadata = {
            "source_type": "FieldRegion volumetric current only; zero dipoles",
            "objective": (
                "native-Yee weighted TaIrTe4 absorption per incident intensity"
            ),
            "objective_unit": "m2 for dimensionless native density weight",
            "incident_polarization": self.incident_polarization,
            "incident_intensity_W_m2": forward.incident_intensity_W_m2,
            "epsilon_imaginary": forward.epsilon_imaginary.tolist(),
            "power_component_per_intensity_m2": (
                forward.observation.power_component_W.tolist()
            ),
            "power_total_per_intensity_m2": (
                forward.observation.power_total_W
            ),
            "active_adjoint_components": [
                "xyz"[component] for component in active_components
            ],
            "density_mode": density_mode,
            "solver_safe_affine": {
                "rho_step": self.rho_step,
                "safety_slack": self.safety_slack,
                "delta": delta,
                "chain_factor": chain,
                "formula": "rho_solver = delta + (1-2*delta)*rho_geom",
            },
            "periodic_source_pairing_relative_error": float(pairing),
            "source": source_meta,
            "rho_epsilon_transpose": {
                "method": (
                    "27-color measured centered layout-only Jacobian transpose"
                ),
                "step": self.rho_step,
                "electromagnetic_solves": 0,
                "max_owner_leakage_fraction": leakage,
            },
            "explicit_loss_density_derivative": (
                "zero: TaIrTe4 loss is fixed and design rho is above the flake"
            ),
        }
        return Evaluation(
            float(forward.value), gradient_full, metadata
        )
