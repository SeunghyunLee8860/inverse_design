#!/usr/bin/env python3
"""Decompose the selected-grid optical adjoint without new Maxwell solves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    component_volumes,
    monitor_electric,
)
from run_production_combined_adfd_smoke import (  # noqa: E402
    FREQUENCY_HZ,
    load_operator,
    open_fdtd,
    relative,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-directory", type=Path, required=True)
    parser.add_argument("--jacobian-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrected-result", type=Path)
    args = parser.parse_args()
    root = args.combined_directory.expanduser().resolve()
    result_path = root / "production_combined_adfd_smoke_result.json"
    result = json.loads(result_path.read_text())
    raw_path = Path(result["raw_artifact"]["path"])
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("combined raw SHA mismatch")
    raw = np.load(raw_path)
    direction = np.asarray(raw["direction"], float)
    operator, rho, operator_meta = load_operator(
        args.jacobian_directory,
        "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
    )
    if not np.array_equal(rho, np.asarray(raw["rho"], float)):
        raise RuntimeError("operator and combined baseline density differ")

    base_project = Path(result["base_forward"]["project"]["path"])
    corrected = None
    if args.corrected_result is not None:
        corrected = json.loads(args.corrected_result.resolve().read_text())
        adjoint_project = Path(corrected["adjoint"]["project"]["path"])
        adjoint_expected_sha = corrected["adjoint"]["project"]["sha256"]
        template_path = Path(corrected["source_method"]["template"]["path"])
        profile_scale = float(corrected["source_method"]["profile_scale"])
    else:
        adjoint_project = Path(result["adjoint"]["project"]["path"])
        adjoint_expected_sha = result["adjoint"]["project"]["sha256"]
        template_path = Path(result["adjoint_source"]["template"]["path"])
        profile_scale = float(result["adjoint_source"]["profile_scale"])
    for path, expected in (
        (base_project, result["base_forward"]["project"]["sha256"]),
        (adjoint_project, adjoint_expected_sha),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"FSP SHA mismatch: {path}")

    fdtd, _, _ = open_fdtd("GPU 0")
    try:
        fdtd.load(str(base_project))
        forward, forward_grid = monitor_electric(fdtd, PABS_FIELD)
        fdtd.load(str(adjoint_project))
        fdtd.cwnorm(1)
        adjoint_first, adjoint_grid = monitor_electric(fdtd, PABS_FIELD)
        fdtd.cwnorm(2)
        adjoint_average, adjoint_average_grid = monitor_electric(
            fdtd, PABS_FIELD
        )
    finally:
        fdtd.close()
    mismatch = max(
        float(
            np.max(
                np.abs(
                    np.asarray(forward_grid[key])
                    - np.asarray(adjoint_grid[key])
                )
            )
        )
        for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
    )
    mismatch = max(
        mismatch,
        max(
            float(
                np.max(
                    np.abs(
                        np.asarray(forward_grid[key])
                        - np.asarray(adjoint_average_grid[key])
                    )
                )
            )
            for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
        ),
    )
    first_over_average = np.vdot(adjoint_average, adjoint_first) / np.vdot(
        adjoint_average, adjoint_average
    )
    normalization_spatial_residual = float(
        np.linalg.norm(
            adjoint_first - first_over_average * adjoint_average
        )
        / max(float(np.linalg.norm(adjoint_first)), np.finfo(float).tiny)
    )
    fieldregion_over_first_source_spectrum = 2.0 * first_over_average - 1.0
    adjoint = adjoint_first / fieldregion_over_first_source_spectrum
    volumes = component_volumes(forward_grid)
    base_amplitude = float(
        (
            corrected.get("fieldregion_base_amplitude", 0.0)
            if corrected is not None
            else result["adjoint_source"]["solver_source_layout_adjustment"].get(
                "fieldregion_base_amplitude", 0.0
            )
        )
    )
    # Older result schema stores the amplitude only in the template-preparation
    # result.  The source object itself is authoritative and read-only here.
    if base_amplitude == 0.0:
        fdtd, _, _ = open_fdtd("GPU 0")
        try:
            fdtd.load(str(template_path))
            base_amplitude = float(
                fdtd.getnamed("run002_component_yee_adjoint_region", "base amplitude")
            )
        finally:
            fdtd.close()

    jvp = operator.jvp(direction)
    indirect_cotangent = {}
    direct_cotangent = {}
    component_rows = {}
    for index, component in enumerate("xyz"):
        field = forward[..., 0, index]
        adjoint_field = adjoint[..., 0, index]
        pulled = np.asarray(
            raw[f"native_Q{component}_density_sensitivity_A_m3_W"], float
        )
        indirect_cotangent[component] = (
            (2.0 * EPS0 / base_amplitude)
            * volumes[index]
            * field
            * (adjoint_field * profile_scale)
        )
        direct_cotangent[component] = (
            -1j
            * 0.5
            * EPS0
            * (2.0 * np.pi * FREQUENCY_HZ)
            * pulled
            * np.abs(field) ** 2
        )
        component_rows[component] = {
            "indirect_directional_A": float(
                np.real(np.sum(indirect_cotangent[component] * jvp[component]))
            ),
            "direct_directional_A": float(
                np.real(np.sum(direct_cotangent[component] * jvp[component]))
            ),
            "J_direction_L2": float(np.linalg.norm(jvp[component])),
            "field_L2": float(np.linalg.norm(field)),
            "adjoint_field_L2": float(np.linalg.norm(adjoint_field)),
        }
    contraction_variants = {}
    for name, forward_transform, adjoint_transform in (
        ("forward_times_adjoint", lambda value: value, lambda value: value),
        ("conj_forward_times_adjoint", np.conj, lambda value: value),
        ("forward_times_conj_adjoint", lambda value: value, np.conj),
        ("conj_forward_times_conj_adjoint", np.conj, np.conj),
    ):
        variant_cotangent = {}
        for index, component in enumerate("xyz"):
            variant_cotangent[component] = (
                (2.0 * EPS0 / base_amplitude)
                * volumes[index]
                * forward_transform(forward[..., 0, index])
                * adjoint_transform(adjoint[..., 0, index] * profile_scale)
            )
        variant_gradient = operator.vjp(variant_cotangent)
        variant_directional = float(np.sum(variant_gradient * direction))
        contraction_variants[name] = {
            "indirect_directional_A": variant_directional,
            "indirect_gradient_L2_A": float(np.linalg.norm(variant_gradient)),
        }
    indirect = operator.vjp(indirect_cotangent)
    direct = operator.vjp(direct_cotangent)
    indirect_directional = float(np.sum(indirect * direction))
    direct_directional = float(np.sum(direct * direction))
    explicit_indirect = sum(
        row["indirect_directional_A"] for row in component_rows.values()
    )
    explicit_direct = sum(
        row["direct_directional_A"] for row in component_rows.values()
    )
    fd_optical = json.loads(
        (root / "selected_combined_failure_decomposition.json").read_text()
    )["directional_derivatives_A"]["FD_optical_fixed_thermal_material"]
    output = {
        "status": "DIAGNOSED_SELECTED_OPTICAL_GRADIENT_TERMS",
        "Maxwell_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "coordinate_mismatch_m": mismatch,
        "profile_scale": profile_scale,
        "fieldregion_base_amplitude": base_amplitude,
        "operator": operator_meta,
        "directional_derivatives_A": {
            "indirect_field_mediated": indirect_directional,
            "direct_material_loss": direct_directional,
            "total_optical_AD": indirect_directional + direct_directional,
            "total_optical_FD": fd_optical,
        },
        "checks": {
            "indirect_VJP_explicit_relative_error": relative(
                indirect_directional, explicit_indirect
            ),
            "direct_VJP_explicit_relative_error": relative(
                direct_directional, explicit_direct
            ),
            "total_optical_relative_error": relative(
                indirect_directional + direct_directional, fd_optical
            ),
        },
        "component_directional_terms": component_rows,
        "complex_contraction_variants": contraction_variants,
        "named_source_normalization": {
            "method": (
                "FieldRegion-only CW adjoint reconstructed from official "
                "cwnorm(1) and cwnorm(2) states while the zero-amplitude "
                "forward Gaussian remains enabled solely as a mesh anchor"
            ),
            "first_over_average_real": float(np.real(first_over_average)),
            "first_over_average_imag": float(np.imag(first_over_average)),
            "fieldregion_over_first_source_spectrum_real": float(
                np.real(fieldregion_over_first_source_spectrum)
            ),
            "fieldregion_over_first_source_spectrum_imag": float(
                np.imag(fieldregion_over_first_source_spectrum)
            ),
            "fieldregion_only_field_multiplier_real": float(
                np.real(1.0 / fieldregion_over_first_source_spectrum)
            ),
            "fieldregion_only_field_multiplier_imag": float(
                np.imag(1.0 / fieldregion_over_first_source_spectrum)
            ),
            "two_normalization_state_spatial_residual": (
                normalization_spatial_residual
            ),
            "uses_finite_difference_fit": False,
            "empirical_gradient_rescaling": False,
        },
        "gradient_norms_A": {
            "indirect": float(np.linalg.norm(indirect)),
            "direct": float(np.linalg.norm(direct)),
            "total": float(np.linalg.norm(indirect + direct)),
        },
        "artifacts": {
            "combined_result": {"path": str(result_path), "sha256": sha256(result_path)},
            "combined_raw": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "base_FSP": {"path": str(base_project), "sha256": sha256(base_project)},
            "adjoint_FSP": {"path": str(adjoint_project), "sha256": sha256(adjoint_project)},
            "corrected_result": (
                {"path": str(args.corrected_result.resolve()), "sha256": sha256(args.corrected_result.resolve())}
                if args.corrected_result is not None
                else None
            ),
        },
    }
    gradient_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        gradient_path,
        rho=rho,
        direction=direction,
        gradient_indirect_A=indirect,
        gradient_direct_A=direct,
        gradient_total_optical_A=indirect + direct,
    )
    output["artifacts"]["corrected_gradient_NPZ"] = {
        "path": str(gradient_path),
        "size_bytes": gradient_path.stat().st_size,
        "sha256": sha256(gradient_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
