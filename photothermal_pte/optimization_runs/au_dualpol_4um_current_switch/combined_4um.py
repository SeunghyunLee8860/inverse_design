"""Checkpoint-free combined Maxwell--thermal--electrical evaluator at 4 um."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    EPS0_F_PER_M,
    LAYOUT,
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    au_material_fraction,
    d_au_material_fraction_drho,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
    evaluate_fixed_source,
    map_native_q_to_thermal,
    pullback_thermal_source_weights,
)
from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.fdtdx_two_solve_adjoint import (
    adjoint_current_from_wirtinger,
    harmonic_material_gradient,
    quadratic_wirtinger_derivative,
)


def replace_named(objects, name: str, replacement):
    result = objects.copy()
    values = list(result.object_list)
    values[result.index(name)] = replacement
    return result.aset("object_list", values)


def electric_yee_dual_volumes(grid, grid_slice: tuple[slice, slice, slice]):
    widths = [np.asarray(grid.cell_widths(axis), dtype=np.float64) for axis in range(3)]
    edge_dual = [
        0.5 * (np.concatenate((value[:1], value[:-1])) + value)
        for value in widths
    ]
    bounds = tuple((int(part.start), int(part.stop)) for part in grid_slice)
    volumes = []
    for component in range(3):
        selected = []
        for axis, (lower, upper) in enumerate(bounds):
            metric = widths[axis] if axis == component else edge_dual[axis]
            selected.append(metric[lower:upper])
        volumes.append(
            selected[0][:, None, None]
            * selected[1][None, :, None]
            * selected[2][None, None, :]
        )
    return np.stack(volumes)


@dataclass
class CompiledOpticalRunner:
    polarization: str
    model: dict[str, object]
    solve_forward: object
    solve_adjoint: object
    volumes: dict[str, np.ndarray]
    physical_prefactor: float
    ta_imag: np.ndarray
    compile_forward_s: float
    compile_adjoint_s: float
    total_periods: int
    window_periods: int

    @classmethod
    def create(
        cls,
        polarization: str,
        example_rho: np.ndarray,
        *,
        total_periods: int = 16,
        window_periods: int = 4,
    ) -> "CompiledOpticalRunner":
        model = build_model(
            polarization,
            include_adjoint_source=True,
            total_periods=total_periods,
            window_periods=window_periods,
        )
        jax = model["jax"]
        jnp = model["jnp"]
        fdtdx = model["fdtdx"]
        au_slice = model["slices"]["au_design"]
        au_c3 = float(model["coefficients"]["au"][2])

        def arrays_for_density(density):
            strength = au_material_fraction(density)
            c3 = model["fixed_c3"]
            for component in range(3):
                c3 = c3.at[(0, component, *au_slice)].set(
                    au_c3 * strength[:, :, None]
                )
            return (
                model["base"]
                .reset()
                .aset("dispersive_c1", model["fixed_c1"])
                .aset("dispersive_c2", model["fixed_c2"])
                .aset("dispersive_c3", c3)
            )

        forward_objects = replace_named(
            model["placed"],
            "distributed_adjoint_source",
            model["placed"]["distributed_adjoint_source"].aset(
                "static_amplitude_factor", 0.0
            ),
        )

        def forward_function(density):
            return fdtdx.run_fdtd(
                arrays_for_density(density),
                forward_objects,
                model["config"],
                model["key"],
                show_progress=False,
            )[1]

        adjoint_base_objects = replace_named(
            model["placed"],
            "gaussian_source",
            model["placed"]["gaussian_source"].aset("static_amplitude_factor", 0.0),
        )

        def adjoint_function(density, complex_profile):
            arrays = arrays_for_density(density)
            source = (
                model["placed"]["distributed_adjoint_source"]
                .aset("complex_profile", complex_profile)
                .aset("static_amplitude_factor", 1.0)
            )
            source = source.apply(
                model["key"],
                arrays.inv_permittivities,
                arrays.inv_permeabilities,
                arrays.dispersive_c1,
                arrays.dispersive_c2,
                arrays.dispersive_c3,
                arrays.electric_conductivity,
                arrays.dispersive_c4,
            )
            objects = replace_named(
                adjoint_base_objects, "distributed_adjoint_source", source
            )
            return fdtdx.run_fdtd(
                arrays,
                objects,
                model["config"],
                model["key"],
                show_progress=False,
            )[1]

        rho = jnp.asarray(example_rho, dtype=jnp.float32)
        start = time.perf_counter()
        solve_forward = jax.jit(forward_function).lower(rho).compile()
        compile_forward_s = time.perf_counter() - start
        adjoint_shape = tuple(
            int(value)
            for value in model["placed"]["distributed_adjoint_source"].grid_shape
        )
        profile = jnp.zeros((3, *adjoint_shape), dtype=jnp.complex64)
        start = time.perf_counter()
        solve_adjoint = jax.jit(adjoint_function).lower(rho, profile).compile()
        compile_adjoint_s = time.perf_counter() - start
        eta0 = float(fdtdx.constants.eta0)
        physical_prefactor = (
            0.5 * float(model["omega_rad_s"]) * EPS0_F_PER_M * eta0**2
        )
        ta_imag = np.asarray(
            [
                model["epsilon"]["tairte4"]["b"].imag,
                model["epsilon"]["tairte4"]["a"].imag,
                model["epsilon"]["tairte4"]["c"].imag,
            ],
            dtype=np.float64,
        )[:, None, None, None]
        volumes = {
            "au": electric_yee_dual_volumes(model["grid"], model["slices"]["au_design"]),
            "tairte4": electric_yee_dual_volumes(
                model["grid"], model["slices"]["fixed_tairte4"]
            ),
        }
        return cls(
            polarization=polarization,
            model=model,
            solve_forward=solve_forward,
            solve_adjoint=solve_adjoint,
            volumes=volumes,
            physical_prefactor=physical_prefactor,
            ta_imag=ta_imag,
            compile_forward_s=compile_forward_s,
            compile_adjoint_s=compile_adjoint_s,
            total_periods=total_periods,
            window_periods=window_periods,
        )

    def run_forward(self, rho: np.ndarray):
        jnp = self.model["jnp"]
        start = time.perf_counter()
        output = self.solve_forward(jnp.asarray(rho, dtype=jnp.float32))
        fields = output.detector_states["au_late"]["phasor"][0, 0]
        self.model["jax"].block_until_ready(fields)
        return output, time.perf_counter() - start

    def run_adjoint(self, rho: np.ndarray, profile: np.ndarray):
        jnp = self.model["jnp"]
        start = time.perf_counter()
        output = self.solve_adjoint(
            jnp.asarray(rho, dtype=jnp.float32),
            jnp.asarray(profile, dtype=jnp.complex64),
        )
        fields = output.detector_states["au_late"]["phasor"][0, 0]
        self.model["jax"].block_until_ready(fields)
        return output, time.perf_counter() - start

    def fields_and_q(
        self, output, rho: np.ndarray, source_power_scale: float
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        e_au = np.asarray(output.detector_states["au_late"]["phasor"][0, 0])
        e_ta = np.asarray(output.detector_states["tairte4_late"]["phasor"][0, 0])
        strength = au_material_fraction(np.asarray(rho, dtype=np.float64))
        q = {
            "au": source_power_scale
            * self.physical_prefactor
            * float(self.model["epsilon"]["au"].imag)
            * strength[None, :, :, None]
            * np.abs(e_au) ** 2,
            "tairte4": source_power_scale
            * self.physical_prefactor
            * self.ta_imag
            * np.abs(e_ta) ** 2,
        }
        return {"au": e_au, "tairte4": e_ta}, q


def discrete_au_susceptibility(runner: CompiledOpticalRunner):
    """Return the carrier-frequency susceptibility of the stored float32 ADE.

    Density changes the Au oscillator's ``c3`` coefficient, not an abstract
    continuous-frequency permittivity.  The two are almost identical for this
    contract, but the exact discrete derivative is used so the certificate
    tests the same operator that advances the fields.
    """

    jnp = runner.model["jnp"]
    c1, c2, c3 = (
        jnp.asarray(value, dtype=jnp.float32)
        for value in runner.model["coefficients"]["au"]
    )
    theta = jnp.asarray(
        float(runner.model["omega_rad_s"])
        * float(runner.model["config"].time_step_duration),
        dtype=jnp.float32,
    )
    z_minus = jnp.exp(-1j * theta)
    z_plus = jnp.exp(1j * theta)
    return c3 / (z_minus - c1 - c2 * z_plus)


def evaluate_forward_multiphysics(
    runner: CompiledOpticalRunner,
    rho: np.ndarray,
    source_power_scale: float,
    cuda_device: int,
    *,
    need_gradient: bool,
) -> dict[str, object]:
    output, forward_s = runner.run_forward(rho)
    fields, q = runner.fields_and_q(output, rho, source_power_scale)
    state = build_thermal_state(rho)
    source_power, mapping, contexts = map_native_q_to_thermal(
        state,
        q_fields_W_m3=q,
        dual_volumes_m3=runner.volumes,
        material_slices={
            "au": runner.model["slices"]["au_design"],
            "tairte4": runner.model["slices"]["fixed_tairte4"],
        },
        realized_grid=runner.model["grid"],
    )
    evaluated = evaluate_fixed_source(
        rho, source_power, cuda_device, need_gradient=need_gradient
    )
    result: dict[str, object] = {
        **evaluated,
        "optical_output": output,
        "fields": fields,
        "q_fields_W_m3": q,
        "source_power_W": source_power,
        "mapping": mapping,
        "mapping_context": contexts,
        "forward_s": forward_s,
    }
    if need_gradient:
        weights = pullback_thermal_source_weights(
            np.asarray(evaluated["thermal_adjoint"]), contexts
        )
        native_contraction = 0.0
        for material in ("au", "tairte4"):
            native_power = q[material] * runner.volumes[material]
            native_contraction += float(np.vdot(weights[material], native_power))
        explicit_contraction = float(
            np.vdot(evaluated["thermal_adjoint"], source_power)
        )
        result.update(
            native_weights_A_W=weights,
            weighted_contraction_relative_error=abs(
                native_contraction - explicit_contraction
            )
            / max(abs(explicit_contraction), np.finfo(float).tiny),
        )
    return result


def combined_gradient(
    runner: CompiledOpticalRunner,
    rho: np.ndarray,
    source_power_scale: float,
    cuda_device: int,
) -> dict[str, object]:
    evaluated = evaluate_forward_multiphysics(
        runner,
        rho,
        source_power_scale,
        cuda_device,
        need_gradient=True,
    )
    fields = evaluated["fields"]
    weights = evaluated["native_weights_A_W"]
    strength = au_material_fraction(np.asarray(rho, dtype=np.float64))
    adjoint_shape = tuple(
        int(value)
        for value in runner.model["placed"]["distributed_adjoint_source"].grid_shape
    )
    e_stack = np.zeros((3, *adjoint_shape), dtype=np.complex64)
    coefficient = np.zeros((3, *adjoint_shape), dtype=np.float32)
    ta_z = slice(LAYOUT.sio2_cells, LAYOUT.sio2_cells + LAYOUT.tairte4_cells)
    e_stack[:, :, :, ta_z] = fields["tairte4"]
    ta_weight = weights["tairte4"]
    coefficient[:, :, :, ta_z] = runner.physical_prefactor * runner.ta_imag * ta_weight
    offset = (LAYOUT.flake_xy_cells - LAYOUT.au_xy_cells) // 2
    au_local = (
        slice(offset, offset + LAYOUT.au_xy_cells),
        slice(offset, offset + LAYOUT.au_xy_cells),
        slice(
            LAYOUT.sio2_cells + LAYOUT.tairte4_cells,
            LAYOUT.sio2_cells + LAYOUT.tairte4_cells + LAYOUT.au_cells,
        ),
    )
    e_stack[(slice(None), *au_local)] = fields["au"]
    au_weight = weights["au"]
    coefficient[(slice(None), *au_local)] = (
        runner.physical_prefactor
        * float(runner.model["epsilon"]["au"].imag)
        * au_weight
        * strength[None, :, :, None]
    )
    wirtinger = quadratic_wirtinger_derivative(
        runner.model["jnp"].asarray(e_stack),
        runner.model["jnp"].asarray(coefficient),
    )
    profile = np.asarray(
        adjoint_current_from_wirtinger(
            wirtinger, runner.model["config"].courant_number
        )
    )
    # FDTDX applies an impressed electric current in the E update one full
    # discrete time step after the field residual sampled by the phasor
    # detector.  The reciprocal harmonic right-hand side therefore carries
    # exp(+i*omega*dt).  This is the exact leapfrog time-staggering phase, not
    # an empirical gradient normalization.
    adjoint_source_phase = np.exp(
        1j
        * float(runner.model["omega_rad_s"])
        * float(runner.model["config"].time_step_duration)
    )
    profile = profile * adjoint_source_phase
    adjoint_output, adjoint_s = runner.run_adjoint(rho, profile)
    e_adj_au = adjoint_output.detector_states["au_late"]["phasor"][0, 0]
    d_strength = runner.model["jnp"].broadcast_to(
        runner.model["jnp"].asarray(d_au_material_fraction_drho(rho))[:, :, None],
        fields["au"].shape[1:],
    )
    d_epsilon = runner.model["jnp"].broadcast_to(
        d_strength[None] * discrete_au_susceptibility(runner),
        fields["au"].shape,
    )
    field_voxel = harmonic_material_gradient(
        runner.model["jnp"].asarray(fields["au"]),
        e_adj_au,
        d_epsilon,
        float(runner.model["omega_rad_s"]),
        float(runner.model["config"].time_step_duration),
    )
    field_voxel = field_voxel * runner.model["jnp"].asarray(runner.volumes["au"])
    field_unscaled = runner.model["jnp"].sum(field_voxel, axis=(0, 3))
    direct_loss_unscaled = (
        runner.physical_prefactor
        * float(runner.model["epsilon"]["au"].imag)
        * runner.model["jnp"].sum(
            runner.model["jnp"].asarray(weights["au"])
            * runner.model["jnp"].asarray(runner.volumes["au"])
            * d_strength[None]
            * runner.model["jnp"].abs(runner.model["jnp"].asarray(fields["au"])) ** 2,
            axis=(0, 3),
        )
    )
    gradient_field = np.asarray(field_unscaled, dtype=np.float64) * source_power_scale
    gradient_loss = (
        np.asarray(direct_loss_unscaled, dtype=np.float64) * source_power_scale
    )
    gradient_optical = gradient_field + gradient_loss
    gradient_direct = np.asarray(evaluated["gradient_direct_A"], dtype=np.float64)
    gradient_total = gradient_optical + gradient_direct
    evaluated.update(
        adjoint_s=adjoint_s,
        adjoint_output=adjoint_output,
        adjoint_profile=profile,
        gradient_optical_field_A=gradient_field,
        gradient_optical_direct_loss_A=gradient_loss,
        gradient_optical_A=gradient_optical,
        gradient_total_A=gradient_total,
        discrete_au_susceptibility=complex(discrete_au_susceptibility(runner)),
        target_au_susceptibility=complex(runner.model["epsilon"]["au"]) - 1.0,
        adjoint_source_phase_factor=complex(adjoint_source_phase),
        adjoint_source_phase_basis="exact FDTDX one-E-update time staggering",
    )
    return evaluated
