#!/usr/bin/env python3
"""Offline decomposition of the selected scalar-P_Q centered difference."""

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
from build_nonuniform_complex_yee_jacobian import index_detail  # noqa: E402
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


def objective(
    electric: np.ndarray,
    epsilon: np.ndarray,
    volumes: dict[int, np.ndarray],
) -> float:
    omega = 2.0 * np.pi * FREQUENCY_HZ
    return float(
        sum(
            np.sum(
                0.5
                * EPS0
                * omega
                * np.imag(epsilon[..., 0, index])
                * np.abs(electric[..., 0, index]) ** 2
                * volumes[index]
            )
            for index in range(3)
        )
    )


def read_state(fdtd: object, path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    fdtd.load(str(path))
    electric, grid = monitor_electric(fdtd, PABS_FIELD)
    detail = index_detail(fdtd)
    epsilon = np.stack(
        [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
    )[..., None, :]
    if electric.shape != epsilon.shape:
        raise RuntimeError(f"E/index mismatch in {path}: {electric.shape} != {epsilon.shape}")
    return electric, epsilon, grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-directory", type=Path, required=True)
    parser.add_argument("--jacobian-directory", type=Path, required=True)
    parser.add_argument("--scalar-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.combined_directory.expanduser().resolve()
    combined_path = root / "production_combined_adfd_smoke_result.json"
    combined = json.loads(combined_path.read_text())
    scalar_path = args.scalar_result.expanduser().resolve()
    scalar = json.loads(scalar_path.read_text())
    raw_path = Path(combined["raw_artifact"]["path"])
    if sha256(raw_path) != combined["raw_artifact"]["sha256"]:
        raise RuntimeError("combined raw SHA mismatch")
    raw = np.load(raw_path)
    direction = np.asarray(raw["direction"], float)
    operator, rho, operator_meta = load_operator(
        args.jacobian_directory,
        "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
    )
    if not np.array_equal(rho, np.asarray(raw["rho"], float)):
        raise RuntimeError("operator/combined density mismatch")

    base_path = Path(combined["base_forward"]["project"]["path"])
    plus_path = Path(combined["FD_pair"]["plus"]["forward"]["project"]["path"])
    minus_path = Path(combined["FD_pair"]["minus"]["forward"]["project"]["path"])
    for path, expected in (
        (base_path, combined["base_forward"]["project"]["sha256"]),
        (plus_path, combined["FD_pair"]["plus"]["forward"]["project"]["sha256"]),
        (minus_path, combined["FD_pair"]["minus"]["forward"]["project"]["sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"FSP SHA mismatch: {path}")

    fdtd, _, _ = open_fdtd("GPU 0")
    try:
        e0, eps0, grid0 = read_state(fdtd, base_path)
        ep, epsp, gridp = read_state(fdtd, plus_path)
        em, epsm, gridm = read_state(fdtd, minus_path)
        adjoint_path = Path(scalar["adjoint"]["project"]["path"])
        if sha256(adjoint_path) != scalar["adjoint"]["project"]["sha256"]:
            raise RuntimeError("scalar adjoint FSP SHA mismatch")
        fdtd.load(str(adjoint_path))
        fdtd.cwnorm(1)
        ea_first, grida = monitor_electric(fdtd, PABS_FIELD)
        fdtd.cwnorm(2)
        ea_average, grid_average = monitor_electric(fdtd, PABS_FIELD)
    finally:
        fdtd.close()
    mismatch = max(
        float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))))
        for left, right in (
            (grid0, gridp),
            (grid0, gridm),
            (grid0, grida),
            (grid0, grid_average),
        )
        for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
    )
    volumes = component_volumes(grid0)
    step = float(combined["step"])
    values = {
        "P_00": objective(e0, eps0, volumes),
        "P_pp": objective(ep, epsp, volumes),
        "P_mm": objective(em, epsm, volumes),
        "P_0p": objective(ep, eps0, volumes),
        "P_0m": objective(em, eps0, volumes),
        "P_p0": objective(e0, epsp, volumes),
        "P_m0": objective(e0, epsm, volumes),
        "P_pm": objective(em, epsp, volumes),
        "P_mp": objective(ep, epsm, volumes),
    }
    fd_total = (values["P_pp"] - values["P_mm"]) / (2.0 * step)
    fd_material_base_field = (values["P_p0"] - values["P_m0"]) / (2.0 * step)
    fd_field_base_epsilon = (values["P_0p"] - values["P_0m"]) / (2.0 * step)
    fd_material_symmetric = (
        (values["P_pp"] - values["P_mp"])
        + (values["P_pm"] - values["P_mm"])
    ) / (4.0 * step)
    fd_field_symmetric = (
        (values["P_pp"] - values["P_pm"])
        + (values["P_mp"] - values["P_mm"])
    ) / (4.0 * step)

    jvp = operator.jvp(direction)
    actual = {
        component: (
            epsp[..., 0, index] - epsm[..., 0, index]
        ) / (2.0 * step)
        for index, component in enumerate("xyz")
    }
    j_rows = {}
    for component in "xyz":
        numerator = float(np.linalg.norm(actual[component] - jvp[component]))
        denominator = max(
            float(np.linalg.norm(actual[component])),
            float(np.linalg.norm(jvp[component])),
            np.finfo(float).tiny,
        )
        j_rows[component] = {
            "centered_FD_vs_JVP_relative_L2": numerator / denominator,
            "centered_FD_L2": float(np.linalg.norm(actual[component])),
            "JVP_L2": float(np.linalg.norm(jvp[component])),
            "maximum_abs_difference": float(
                np.max(np.abs(actual[component] - jvp[component]))
            ),
        }

    ad = scalar["directional_derivatives_W"]
    profile_scale = float(scalar["source_method"]["profile_scale"])
    base_amplitude = float(scalar["source_method"]["fieldregion_base_amplitude"])
    first_over_average = np.vdot(ea_average, ea_first) / np.vdot(
        ea_average, ea_average
    )
    normalization_fit_residual = float(
        np.linalg.norm(ea_first - first_over_average * ea_average)
        / max(float(np.linalg.norm(ea_first)), np.finfo(float).tiny)
    )
    # cwnorm(1) uses the first active (zero-amplitude mesh-anchor Gaussian)
    # source spectrum s_g, while cwnorm(2) uses (s_g+s_fr)/2.  Therefore
    # E_first/E_average=(s_g+s_fr)/(2*s_g), which determines the FieldRegion-
    # only spectrum ratio without reference to finite differences.
    fieldregion_over_first_source_spectrum = 2.0 * first_over_average - 1.0
    ea = ea_first / fieldregion_over_first_source_spectrum
    contraction_variants = {}
    for name, forward_transform, adjoint_transform in (
        ("forward_times_adjoint", lambda value: value, lambda value: value),
        ("conj_forward_times_adjoint", np.conj, lambda value: value),
        ("forward_times_conj_adjoint", lambda value: value, np.conj),
        ("conj_forward_times_conj_adjoint", np.conj, np.conj),
    ):
        complex_directional = sum(
            np.sum(
                (2.0 * EPS0 / base_amplitude)
                * volumes[index]
                * forward_transform(e0[..., 0, index])
                * adjoint_transform(ea[..., 0, index] * profile_scale)
                * jvp[component]
            )
            for index, component in enumerate("xyz")
        )
        contraction_variants[name] = {
            "complex_directional_real_W": float(np.real(complex_directional)),
            "complex_directional_imag_W": float(np.imag(complex_directional)),
            "real_part_relative_error_vs_FD_field": relative(
                float(np.real(complex_directional)), fd_field_base_epsilon
            ),
            "negative_imag_part_relative_error_vs_FD_field": relative(
                float(-np.imag(complex_directional)), fd_field_base_epsilon
            ),
        }
    corrected_indirect = contraction_variants["forward_times_adjoint"][
        "complex_directional_real_W"
    ]
    corrected_total = float(ad["direct_material_loss"]) + corrected_indirect

    output = {
        "status": "DIAGNOSED_SELECTED_SCALAR_PQ_FD_DECOMPOSITION",
        "Maxwell_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "coordinate_mismatch_m": mismatch,
        "step": step,
        "mixed_objective_values_W": values,
        "directional_derivatives_W": {
            "FD_total": fd_total,
            "FD_material_at_base_field": fd_material_base_field,
            "FD_field_at_base_epsilon": fd_field_base_epsilon,
            "FD_base_split_sum": fd_material_base_field + fd_field_base_epsilon,
            "FD_base_split_cross_residual": (
                fd_total - fd_material_base_field - fd_field_base_epsilon
            ),
            "FD_material_symmetric_exact_split": fd_material_symmetric,
            "FD_field_symmetric_exact_split": fd_field_symmetric,
            "FD_symmetric_split_sum": fd_material_symmetric + fd_field_symmetric,
            "AD_direct_material_loss": ad["direct_material_loss"],
            "AD_indirect_field_mediated": ad["indirect_field_mediated"],
            "AD_total": ad["total_AD"],
        },
        "relative_errors": {
            "AD_direct_vs_FD_material_base": relative(
                float(ad["direct_material_loss"]), fd_material_base_field
            ),
            "AD_indirect_vs_FD_field_base": relative(
                float(ad["indirect_field_mediated"]), fd_field_base_epsilon
            ),
            "AD_total_vs_FD_total": relative(float(ad["total_AD"]), fd_total),
            "FD_symmetric_split_closure": relative(
                fd_material_symmetric + fd_field_symmetric, fd_total
            ),
            "named_source_corrected_indirect_vs_FD_field_base": relative(
                corrected_indirect, fd_field_base_epsilon
            ),
            "named_source_corrected_total_vs_FD_total": relative(
                corrected_total, fd_total
            ),
        },
        "material_JVP_checks": j_rows,
        "complex_contraction_variants": contraction_variants,
        "named_source_normalization": {
            "method": (
                "FieldRegion-only CW adjoint reconstructed from the same raw "
                "monitor data under official cwnorm(1) and cwnorm(2) states"
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
            "two_normalization_state_spatial_residual": normalization_fit_residual,
            "uses_finite_difference_fit": False,
            "empirical_gradient_rescaling": False,
            "corrected_indirect_directional_W": corrected_indirect,
            "corrected_total_directional_W": corrected_total,
        },
        "operator": operator_meta,
        "artifacts": {
            "combined_result": {
                "path": str(combined_path),
                "sha256": sha256(combined_path),
            },
            "scalar_result": {"path": str(scalar_path), "sha256": sha256(scalar_path)},
            "base_FSP": {"path": str(base_path), "sha256": sha256(base_path)},
            "plus_FSP": {"path": str(plus_path), "sha256": sha256(plus_path)},
            "minus_FSP": {"path": str(minus_path), "sha256": sha256(minus_path)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
