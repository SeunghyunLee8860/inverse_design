#!/usr/bin/env python3
"""GPU P_Q shape-adjoint diagnostic for an exact-binary Au rectangle.

The two x-normal faces move symmetrically with the half-width parameter.  The
candidate derivative is the sum of

1. the Maxwell field-mediated boundary perturbation, and
2. the explicit moving-domain derivative of the Au absorption integral.

This explicitly tests whether the continuous moving-domain trace is compatible
with the discrete conformal-Yee P_Q objective and fails closed if it is not.
No finite-difference fit, empirical normalization, or gradient rescaling is
used. Forward central differences are read from independently completed GPU
FDTD artifacts and are never used to modify the adjoint result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
LEGACY = HERE.parent / "legacy_v261_optical_support"
LUMERICAL_API = Path("/opt/lumerical/v261/api/python")
for path in (HERE, REPOSITORY, LEGACY, LUMERICAL_API):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    absorption_objective_and_source,
    fieldregion_profile,
    import_named_fieldregion_profile,
    monitor_electric,
)
from build_nonuniform_complex_yee_jacobian import index_detail  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    FIELD_REGION,
    FREQUENCY_HZ,
    open_fdtd,
    reconstruct_fieldregion_only_cw,
    relative,
    run_adjoint,
)


WAVELENGTH_M = 10.0e-6
AU_EPSILON = complex(-4642.2300000000005, 1674.64)
AIR_EPSILON = 1.0 + 0.0j
AU_HALF_Y_M = 10.0e-6
AU_Z_MIN_M = 0.05e-6
AU_Z_MAX_M = 0.10e-6
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
FD_CASES = {
    0.10: ("sharp_width_7p9_edge25_forward", "sharp_width_8p1_edge25_forward"),
    0.05: ("sharp_width_7p95_edge25_forward", "sharp_width_8p05_edge25_forward"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_project(case_directory: Path) -> tuple[Path, dict[str, object]]:
    result_path = case_directory / "case_result.json"
    result = json.loads(result_path.read_text())
    project = case_directory / "complex_material_control.fsp"
    stored = {
        str(Path(item["path"]).resolve()): item
        for item in result.get("raw_artifacts", [])
    }
    item = stored.get(str(project.resolve()))
    if item is None:
        raise RuntimeError(f"project missing from raw provenance: {project}")
    actual = sha256(project)
    if actual != item["sha256"]:
        raise RuntimeError(
            f"project SHA mismatch for {project}: {actual} != {item['sha256']}"
        )
    if not bool(result.get("passed", False)):
        raise RuntimeError(f"forward optical gate did not pass: {case_directory}")
    return project, result


def add_adjoint_fieldregion(fdtd: object, grid: dict[str, np.ndarray]) -> None:
    if int(fdtd.getnamednumber(FIELD_REGION)) == 0:
        region = fdtd.addfieldregion()
        region["name"] = FIELD_REGION
        region["monitor type"] = "3D"
    elif int(fdtd.getnamednumber(FIELD_REGION)) != 1:
        raise RuntimeError(f"non-unique {FIELD_REGION}")
    for axis in "xyz":
        fdtd.setnamed(FIELD_REGION, f"{axis} min", float(grid[axis][0]))
        fdtd.setnamed(FIELD_REGION, f"{axis} max", float(grid[axis][-1]))
    fdtd.setnamed(FIELD_REGION, "override global monitor settings", True)
    fdtd.setnamed(FIELD_REGION, "use source limits", False)
    fdtd.setnamed(FIELD_REGION, "use wavelength spacing", True)
    fdtd.setnamed(FIELD_REGION, "wavelength center", WAVELENGTH_M)
    fdtd.setnamed(FIELD_REGION, "wavelength span", 0.0)
    fdtd.setnamed(FIELD_REGION, "frequency points", 1)
    try:
        fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
    except Exception:
        pass
    fdtd.setnamed(FIELD_REGION, "source mode", True)


class VectorizedNoInterpFields:
    """Vectorized evaluator on the three component-specific Yee grids."""

    def __init__(
        self,
        electric: np.ndarray,
        epsilon: np.ndarray,
        grid: dict[str, np.ndarray],
    ):
        displacement = EPS0 * epsilon * electric
        base = [np.asarray(grid[axis], float) for axis in "xyz"]
        delta = [np.asarray(grid[f"delta_{axis}"], float) for axis in "xyz"]
        self.electric_interpolators = []
        self.displacement_interpolators = []
        for component in range(3):
            axes = [np.array(axis, copy=True) for axis in base]
            axes[component] += delta[component]
            self.electric_interpolators.append(
                RegularGridInterpolator(
                    tuple(axes),
                    np.asarray(electric[..., 0, component]),
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
            )
            self.displacement_interpolators.append(
                RegularGridInterpolator(
                    tuple(axes),
                    np.asarray(displacement[..., 0, component]),
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
            )

    @staticmethod
    def evaluate(interpolators, x, y, z):
        x_array, y_array, z_array = np.broadcast_arrays(
            np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        )
        shape = x_array.shape
        points = np.column_stack(
            (x_array.reshape(-1), y_array.reshape(-1), z_array.reshape(-1))
        )
        value = np.stack(
            [interpolator(points) for interpolator in interpolators], axis=-1
        )
        return value.reshape(*shape, 3) if shape else value.reshape(3)

    def getfield(self, x, y, z, _wavelength):
        return self.evaluate(self.electric_interpolators, x, y, z)

    def getDfield(self, x, y, z, _wavelength):
        return self.evaluate(self.displacement_interpolators, x, y, z)


def build_nointerp_fields(
    electric: np.ndarray,
    epsilon: np.ndarray,
    grid: dict[str, np.ndarray],
):
    return VectorizedNoInterpFields(electric, epsilon, grid)


def project(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    unit = normal / np.linalg.norm(normal)
    return np.dot(vector, unit) * unit


def surface_integrals(
    forward_fields,
    adjoint_fields,
    *,
    half_width_m: float,
    dy_m: float,
    dz_m: float,
) -> dict[str, float | int]:
    """Midpoint-quadrature integrals over both moving vertical Au faces."""

    ny = int(round((2.0 * AU_HALF_Y_M) / dy_m))
    nz = int(round((AU_Z_MAX_M - AU_Z_MIN_M) / dz_m))
    if ny < 2 or nz < 2:
        raise ValueError("surface quadrature is too coarse")
    dy = 2.0 * AU_HALF_Y_M / ny
    dz = (AU_Z_MAX_M - AU_Z_MIN_M) / nz
    ys = -AU_HALF_Y_M + (np.arange(ny) + 0.5) * dy
    zs = AU_Z_MIN_M + (np.arange(nz) + 0.5) * dz
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    sample_shape = yy.shape

    def evaluate(field_function, x: float) -> np.ndarray:
        xx = np.full(sample_shape, x, float)
        ww = np.full(sample_shape, WAVELENGTH_M, float)
        value = np.asarray(field_function(xx, yy, zz, ww), complex)
        if value.shape == (3, *sample_shape):
            value = np.moveaxis(value, 0, -1)
        if value.shape != (*sample_shape, 3):
            raise RuntimeError(
                f"vectorized boundary field shape {value.shape} != "
                f"{(*sample_shape, 3)}"
            )
        return value

    omega = 2.0 * np.pi * FREQUENCY_HZ
    indirect = 0.0
    direct = 0.0
    finite = True
    for x, normal_x in ((-half_width_m, -1.0), (half_width_m, 1.0)):
        normal = np.asarray([normal_x, 0.0, 0.0])
        ef = evaluate(forward_fields.getfield, x)
        df = evaluate(forward_fields.getDfield, x)
        ea = evaluate(adjoint_fields.getfield, x)
        da = evaluate(adjoint_fields.getDfield, x)
        finite = finite and bool(
            all(np.all(np.isfinite(value)) for value in (ef, df, ea, da))
        )
        ef_parallel = ef - ef[..., 0, None] * normal
        ea_parallel = ea - ea[..., 0, None] * normal
        df_perp = df[..., 0, None] * normal
        da_perp = da[..., 0, None] * normal
        kernel = (
            2.0
            * EPS0
            * (AU_EPSILON - AIR_EPSILON)
            * np.sum(ef_parallel * ea_parallel, axis=-1)
            + (1.0 / AIR_EPSILON - 1.0 / AU_EPSILON)
            / EPS0
            * np.sum(df_perp * da_perp, axis=-1)
        )
        indirect += float(np.sum(np.real(kernel))) * dy * dz

        # Inside-Au trace reconstructed from the electromagnetic continuity
        # variables: tangential E and normal D.
        e_inside_squared = np.sum(np.abs(ef_parallel) ** 2, axis=-1) + np.sum(
            np.abs(df_perp / (EPS0 * AU_EPSILON)) ** 2, axis=-1
        )
        q_inside = (
            0.5
            * EPS0
            * omega
            * float(np.imag(AU_EPSILON))
            * e_inside_squared
        )
        direct += float(np.sum(q_inside)) * dy * dz
    return {
        "dy_m": dy,
        "dz_m": dz,
        "ny": ny,
        "nz": nz,
        "surface_samples": 2 * ny * nz,
        "indirect_W_per_m": indirect,
        "direct_W_per_m": direct,
        "total_W_per_m": indirect + direct,
        "all_finite": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--baseline-case", default="sharp_width_8p0_edge25_forward"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 6")
    parser.add_argument("--resume-completed-adjoint", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume_completed_adjoint:
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "au_sharp_interface_pq_adjoint_result.json"
    result: dict[str, object] = {
        "status": "FAILED_AU_SHARP_INTERFACE_PQ_ADJOINT",
        "passed": False,
        "Maxwell_forward_solves": 0,
        "Maxwell_adjoint_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "CPU_FDTD_fallback": False,
    }
    fdtd = None
    try:
        baseline_project, baseline_result = checked_project(
            args.raw_root / args.baseline_case
        )
        finite_differences: dict[str, object] = {}
        for h_um, (minus_name, plus_name) in FD_CASES.items():
            _, minus = checked_project(args.raw_root / minus_name)
            _, plus = checked_project(args.raw_root / plus_name)
            derivative = (
                float(plus["P_Q_W"]) - float(minus["P_Q_W"])
            ) / (2.0 * h_um)
            finite_differences[f"h_{h_um:g}_um"] = {
                "h_um": h_um,
                "minus_case": minus_name,
                "plus_case": plus_name,
                "derivative_W_per_um": derivative,
            }

        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        fdtd.load(str(baseline_project))
        fdtd.runanalysis(PABS_GROUP)
        forward, grid = monitor_electric(fdtd, PABS_FIELD)
        detail = index_detail(fdtd)
        epsilon = np.stack(
            [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
        )[..., None, :]
        if epsilon.shape != forward.shape:
            raise RuntimeError(
                f"forward E/index shape mismatch {forward.shape} != {epsilon.shape}"
            )
        objective, native_source, component_power = absorption_objective_and_source(
            forward, epsilon, grid
        )
        objective_error = relative(objective, float(baseline_result["P_Q_W"]))
        if objective_error >= 5.0e-3:
            raise RuntimeError(
                f"baseline P_Q reconstruction error {objective_error:.3e}"
            )

        profile, profile_scale = fieldregion_profile(native_source)
        original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
        template = output / "au_sharp_interface_pq_adjoint_template.fsp"
        adjoint_project = output / "au_sharp_interface_pq_adjoint_gpu.fsp"
        if args.resume_completed_adjoint:
            if not template.is_file() or not adjoint_project.is_file():
                raise FileNotFoundError(
                    "resume requires the completed template and adjoint FSP"
                )
            fdtd.load(str(adjoint_project))
            base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
            imported = np.asarray(
                fdtd.getresult(FIELD_REGION, "source profile")["E"], complex
            )
            roundtrip = float(np.max(np.abs(imported - profile)))
            del imported
            fdtd.cwnorm(1)
            electric_first, adjoint_grid = monitor_electric(fdtd, PABS_FIELD)
            fdtd.cwnorm(2)
            electric_average, average_grid = monitor_electric(fdtd, PABS_FIELD)
            average_mismatch = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(adjoint_grid[key])
                            - np.asarray(average_grid[key])
                        )
                    )
                )
                for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
            )
            electric, normalization = reconstruct_fieldregion_only_cw(
                electric_first, electric_average
            )
            normalization["grid_mismatch_m"] = average_mismatch
            log_audit = audit.log_audit(output)
            adjoint = {
                "electric": electric,
                "grid": adjoint_grid,
                "resources": {"reused_completed_GPU_adjoint": True},
                "resource_used": "REUSED_COMPLETED_GPU_ADJOINT",
                "solver_mode": "GPU",
                "named_source_normalization": normalization,
                "log_audit": log_audit,
                "wall_s": 0.0,
                "project": {
                    "path": str(adjoint_project),
                    "size_bytes": adjoint_project.stat().st_size,
                    "sha256": sha256(adjoint_project),
                },
                "reused_without_new_Maxwell_solve": True,
            }
            result["Maxwell_adjoint_solves"] = 1
            result["Maxwell_adjoint_solves_this_invocation"] = 0
        else:
            fdtd.switchtolayout()
            fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
            fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
            add_adjoint_fieldregion(fdtd, grid)
            roundtrip = import_named_fieldregion_profile(
                fdtd, FIELD_REGION, grid, profile
            )
            base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
            fdtd.save(str(template))
            adjoint = run_adjoint(
                fdtd,
                audit,
                runtime,
                template=template,
                project=adjoint_project,
            )
            result["Maxwell_adjoint_solves"] = 1
            result["Maxwell_adjoint_solves_this_invocation"] = 1
        source_meta = {
            "method": (
                "official common-grid GPU FieldRegion P_Q vector source; "
                "forward Gaussian retained at zero amplitude as mesh anchor"
            ),
            "profile_scale": profile_scale,
            "fieldregion_base_amplitude": base_amplitude,
            "source_profile_roundtrip_max_abs_error": roundtrip,
            "forward_Gaussian_original_amplitude": original_amplitude,
            "forward_Gaussian_adjoint_amplitude": 0.0,
            "template": {
                "path": str(template),
                "size_bytes": template.stat().st_size,
                "sha256": sha256(template),
            },
        }
        del profile
        del native_source
        mismatch = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(grid[key])
                        - np.asarray(adjoint["grid"][key])
                    )
                )
            )
            for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
        )
        if mismatch >= 2.0e-18:
            raise RuntimeError(f"forward/adjoint grid mismatch {mismatch:.3e} m")

        forward_fields = build_nointerp_fields(forward, epsilon, grid)
        scaled_adjoint = np.asarray(adjoint["electric"]) * (
            profile_scale / base_amplitude
        )
        adjoint_fields = build_nointerp_fields(scaled_adjoint, epsilon, grid)
        quadratures = []
        for dy_nm, dz_nm in ((50.0, 5.0), (25.0, 2.5), (12.5, 1.25)):
            quadratures.append(
                surface_integrals(
                    forward_fields,
                    adjoint_fields,
                    half_width_m=8.0e-6,
                    dy_m=dy_nm * 1e-9,
                    dz_m=dz_nm * 1e-9,
                )
            )
        selected = quadratures[-1]
        adjoint_w_per_um = float(selected["total_W_per_m"]) * 1.0e-6
        indirect_w_per_um = float(selected["indirect_W_per_m"]) * 1.0e-6
        direct_w_per_um = float(selected["direct_W_per_m"]) * 1.0e-6
        qconv = relative(
            float(quadratures[-1]["total_W_per_m"]),
            float(quadratures[-2]["total_W_per_m"]),
        )
        comparisons = {}
        for name, fd_row in finite_differences.items():
            fd_value = float(fd_row["derivative_W_per_um"])
            comparisons[name] = {
                "FD_W_per_um": fd_value,
                "AD_W_per_um": adjoint_w_per_um,
                "relative_error": relative(adjoint_w_per_um, fd_value),
            }
        strong_error = float(comparisons["h_0.05_um"]["relative_error"])
        passed = bool(
            strong_error < 0.01
            and qconv < 5.0e-3
            and roundtrip == 0.0
            and mismatch < 2.0e-18
            and all(bool(row["all_finite"]) for row in quadratures)
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
        )
        result.update(
            {
                "status": (
                    "VALIDATED_AU_SHARP_INTERFACE_PQ_ADJOINT"
                    if passed
                    else (
                        "FAILED_AU_SHARP_INTERFACE_PQ_CONTINUOUS_TRACE_"
                        "INCOMPATIBLE_WITH_DISCRETE_YEE_OBJECTIVE"
                    )
                ),
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "isolated exact-binary Au optical P_Q width derivative; "
                    "no thermal, electrical, PTE, or optimization result"
                ),
                "baseline": {
                    "case": args.baseline_case,
                    "project": {
                        "path": str(baseline_project),
                        "size_bytes": baseline_project.stat().st_size,
                        "sha256": sha256(baseline_project),
                    },
                    "P_Q_W": baseline_result["P_Q_W"],
                    "P_six_W": baseline_result["P_six_W"],
                    "closure_relative": baseline_result[
                        "six_face_closure_relative"
                    ],
                    "P_Q_reconstructed_W": objective,
                    "P_Q_reconstruction_relative_error": objective_error,
                    "component_power_W": component_power,
                },
                "shape_parameter": {
                    "name": "Au_half_x",
                    "value_um": 8.0,
                    "moved_faces": ["x_min", "x_max"],
                    "fixed_faces": ["y_min", "y_max", "z_min", "z_max"],
                },
                "derivative_decomposition": {
                    "field_mediated_boundary_W_per_um": indirect_w_per_um,
                    "explicit_moving_absorption_domain_W_per_um": direct_w_per_um,
                    "total_AD_W_per_um": adjoint_w_per_um,
                    "direct_term_definition": (
                        "inside-Au absorption trace reconstructed from continuous "
                        "tangential E and normal D"
                    ),
                },
                "surface_quadrature": quadratures,
                "surface_quadrature_final_refinement_relative_change": qconv,
                "finite_difference": finite_differences,
                "AD_FD_comparison": comparisons,
                "source": source_meta,
                "forward_adjoint_maximum_coordinate_mismatch_m": mismatch,
                "adjoint": {
                    key: value
                    for key, value in adjoint.items()
                    if key not in {"electric", "grid"}
                },
                "adjoint_grid": {
                    "shape_xyz": [
                        int(np.asarray(grid[axis]).size) for axis in "xyz"
                    ],
                    "bounds_m": {
                        axis: [
                            float(np.asarray(grid[axis])[0]),
                            float(np.asarray(grid[axis])[-1]),
                        ]
                        for axis in "xyz"
                    },
                },
                "gates": {
                    "strong_h0p05_relative_error_lt_1pct": strong_error < 0.01,
                    "surface_quadrature_change_lt_0p5pct": qconv < 5.0e-3,
                    "source_roundtrip_exact": roundtrip == 0.0,
                    "coordinate_mismatch_lt_2e_18_m": mismatch < 2.0e-18,
                    "adjoint_auto_shutoff_lt_1e_5": float(
                        adjoint["log_audit"]["final_auto_shutoff"]
                    )
                    < 1.0e-5,
                },
                "gray_Au_air_material_used": False,
                "Q_clipping_smoothing_gain_or_rescaling": False,
                "finite_difference_used_to_fit_or_rescale_AD": False,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if bool(result.get("passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
