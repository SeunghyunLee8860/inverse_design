#!/usr/bin/env python3
"""Publish the Au temperature-carrier and smooth-shape diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = Path("/data/seunghyun/tairte4/raw_artifacts/au_topology_validation")
STATUS = "BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_NO_STABLE_GPU_DIFFERENTIABLE_AU_PATH"

PATHS = {
    "moderate_synthetic": RAW
    / "temperature_density_synthetic_conformal1_base1p5_target_n2_k0p5_gpu0"
    / "case_result.json",
    "real_heat_pva": RAW
    / "temperature_density_real_heat_mat_base1p5_target_n2_k0p5_gpu0"
    / "case_result.json",
    "real_heat_conformal": RAW
    / "temperature_density_real_heat_mat_conformal1b_base1p5_target_n2_k0p5_gpu0"
    / "case_result.json",
    "au_span1": RAW
    / "temperature_density_au_50nm_rho1_conformal1_gpu0"
    / "case_result.json",
    "au_span1000_linear": RAW
    / "temperature_density_au_50nm_rho1_span1000_conformal1_gpu0"
    / "case_result.json",
    "au_span1000_table": RAW
    / "temperature_density_au_50nm_rho1_span1000_table_conformal1_gpu0"
    / "case_result.json",
    "au_span1000_dt0p5_short": RAW
    / "temperature_density_au_50nm_rho1_span1000_conformal1_dt0p5_short_gpu0"
    / "case_result.json",
    "au_reverse": RAW
    / "temperature_density_reverse_au_50nm_rho1_conformal1_gpu0"
    / "case_result.json",
    "smooth_one_sided": RAW
    / "pva5_smooth3d_ellipsoid_one_sided_traces_v2"
    / "au_smooth3d_one_sided_trace_result.json",
    "smooth_discrete_epsilon": RAW
    / "pva5_smooth3d_discrete_epsilon_shape_adjoint_gpu0_v4"
    / "au_smooth3d_discrete_epsilon_shape_result.json",
    "smooth_boundary": RAW
    / "pva5_smooth3d_ellipsoid_boundary_adjoint_gpu0"
    / "au_smooth3d_ellipsoid_boundary_adjoint_result.json",
    "heat_mat": Path(
        "/data/seunghyun/tairte4/raw_artifacts/au_topology_validation/"
        "real_heat_thermal_dataset.mat"
    ),
}


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def divergence(result: dict) -> bool:
    return "fields were diverging" in str(result.get("error", ""))


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = {name: read(path) for name, path in PATHS.items() if path.suffix == ".json"}
    moderate = data["moderate_synthetic"]
    one_sided = data["smooth_one_sided"]
    discrete = data["smooth_discrete_epsilon"]
    smooth = data["smooth_boundary"]

    epsilon_readback = moderate["epsilon_component_readback"]
    requested_epsilon = moderate["material"]["requested_epsilon"]
    moderate_errors = {
        component: abs(
            complex(*row["epsilon_interior_median"])
            - complex(*requested_epsilon)
        )
        / abs(complex(*requested_epsilon))
        for component, row in epsilon_readback.items()
    }
    exact_au_cases = {
        name: {
            "status": data[name]["status"],
            "diverged": divergence(data[name]),
            "carrier_span_K": data[name]
            .get("material", {})
            .get("temperature_attribute_numerical_span_K"),
            "temperature_model": data[name]
            .get("material", {})
            .get("temperature_interpolation_model"),
            "interpolation_direction": data[name]
            .get("material", {})
            .get("interpolation_direction"),
            "dt_stability_factor": data[name]
            .get("built_source_contract", {})
            .get("mesh_contract", {})
            .get("dt_stability_factor"),
            "CPU_FDTD_fallback": False,
        }
        for name in (
            "au_span1",
            "au_span1000_linear",
            "au_span1000_table",
            "au_span1000_dt0p5_short",
            "au_reverse",
        )
    }
    trace_signs = [bool(row["FD_sign_agrees"]) for row in one_sided["traces"]]
    trace_errors = [float(row["FD_relative_error"]) for row in one_sided["traces"]]
    strong_discrete = discrete["comparisons"]["h_0.05_um"]

    summary = {
        "status": STATUS,
        "production_Au_optimization_permitted": False,
        "solver_version": moderate["solver_version"],
        "Au_optical_endpoint_10um": {
            "n": 12.1,
            "k": 69.2,
            "epsilon": [-4642.2300000000005, 1674.64],
            "source": "Ordal endpoint already frozen by the repository material audit",
        },
        "temperature_attribute_contract": {
            "attribute_is_physical_temperature": False,
            "may_be_exported_to_thermal_solver": False,
            "PVA_result": "attribute ignored in the material solution",
            "conformal_variant_1_moderate_endpoint": {
                "requested_epsilon": requested_epsilon,
                "component_relative_errors": moderate_errors,
                "P_Q_W": moderate["P_Q_W"],
                "P_six_W": moderate["P_six_W"],
                "six_face_closure_relative": moderate[
                    "six_face_closure_relative"
                ],
                "auto_shutoff": moderate["log_audit"]["final_auto_shutoff"],
                "passed": moderate["passed"],
            },
            "exact_Au_50nm_cases": exact_au_cases,
            "conclusion": (
                "the v261 conformal GPU path honors a moderate complex-index "
                "temperature carrier, but every exact-Au endpoint variant "
                "diverges, including 1000 K numerical spans, a nonlinear table, "
                "and a reduced 0.5 CFL stability factor"
            ),
        },
        "smooth3D_boundary_trace": {
            "FD_J_proxy_per_um": one_sided["FD_target_h0p05_J_proxy_per_um"],
            "tested_offsets_nm": [
                1.0e9 * float(row["field_trace_normal_offset_m"])
                for row in one_sided["traces"]
            ],
            "any_trace_sign_agrees": any(trace_signs),
            "minimum_trace_relative_error": min(trace_errors),
            "geometric_trace_relative_error": next(
                float(row["FD_relative_error"])
                for row in one_sided["traces"]
                if float(row["field_trace_normal_offset_m"]) == 0.0
            ),
            "interpretation": (
                "neither Au-inside, geometric, air-outside, nor symmetric "
                "one-sided traces recover the independent central-FD sign"
            ),
        },
        "smooth3D_solver_discrete_epsilon": {
            "strong_h0p05": strong_discrete,
            "derivative_step_relative_change": discrete[
                "discrete_derivative_step_relative_change"
            ],
            "passed": discrete["passed"],
            "interpretation": (
                "the conformal geometry-to-diagonal-epsilon Jacobian is not "
                "step converged and its adjoint contraction has the wrong sign"
            ),
        },
        "preserved_previous_controls": {
            "exact_scalar_Au_forward": (
                "stable GPU reference; closure 0.108011% in the existing "
                "20x20x0.05 um binary control"
            ),
            "fixed_geometry_material_AD_FD": "passed previously at 0.003896%",
            "original_smooth3D_boundary_status": smooth["status"],
        },
        "excluded_causes": [
            "forward/adjoint coordinate mismatch",
            "FieldRegion source roundtrip or cwnorm mixing",
            "only sharp polygon corners or top/bottom rims",
            "only exact-boundary versus one-sided field interpolation",
            "only a too-large one-kelvin material sensitivity coefficient",
        ],
        "remaining_blocker": (
            "no tested v261 GPU representation is simultaneously stable at "
            "the exact Au endpoint and differentiable with a validated gradient"
        ),
        "next_research_route_not_yet_validated": (
            "a causal dispersive Drude/ADE density parameterization whose "
            "spatial oscillator-strength Jacobian is exposed by the solver, or "
            "a different Maxwell backend with a certified metal topology adjoint"
        ),
        "no_thermal_electrical_PTE_or_optimization_executed": True,
    }

    rows = []
    for name in ("moderate_synthetic", "real_heat_pva", "real_heat_conformal"):
        row = data[name]
        rows.append(
            {
                "case": name,
                "status": row.get("status"),
                "passed": row.get("passed"),
                "diverged": divergence(row),
                "P_Q_W": row.get("P_Q_W"),
                "P_six_W": row.get("P_six_W"),
                "closure_percent": (
                    None
                    if row.get("six_face_closure_relative") is None
                    else 100.0 * float(row["six_face_closure_relative"])
                ),
                "note": "temperature carrier is numerical, never thermal T",
            }
        )
    for name, info in exact_au_cases.items():
        rows.append(
            {
                "case": name,
                "status": info["status"],
                "passed": False,
                "diverged": info["diverged"],
                "P_Q_W": None,
                "P_six_W": None,
                "closure_percent": None,
                "note": (
                    f"span={info['carrier_span_K']} K; "
                    f"model={info['temperature_model']}; "
                    f"direction={info['interpolation_direction']}; "
                    f"dt_factor={info['dt_stability_factor']}"
                ),
            }
        )

    manifest_paths = {
        "moderate_temperature_carrier_result": PATHS["moderate_synthetic"],
        "moderate_temperature_carrier_fsp": PATHS["moderate_synthetic"].parent
        / "complex_material_control.fsp",
        "moderate_temperature_carrier_q": PATHS["moderate_synthetic"].parent
        / "complex_material_control_q.npz",
        "real_heat_temperature_dataset": PATHS["heat_mat"],
        "real_heat_conformal_result": PATHS["real_heat_conformal"],
        "exact_Au_span1000_linear_result": PATHS["au_span1000_linear"],
        "exact_Au_span1000_linear_fsp": PATHS["au_span1000_linear"].parent
        / "complex_material_control.fsp",
        "exact_Au_span1000_linear_log": PATHS["au_span1000_linear"].parent
        / "complex_material_control_p0.log",
        "exact_Au_span1000_table_result": PATHS["au_span1000_table"],
        "exact_Au_span1000_table_fsp": PATHS["au_span1000_table"].parent
        / "complex_material_control.fsp",
        "exact_Au_span1000_table_log": PATHS["au_span1000_table"].parent
        / "complex_material_control_p0.log",
        "exact_Au_span1000_dt0p5_result": PATHS["au_span1000_dt0p5_short"],
        "exact_Au_span1000_dt0p5_fsp": PATHS["au_span1000_dt0p5_short"].parent
        / "complex_material_control.fsp",
        "exact_Au_span1000_dt0p5_log": PATHS["au_span1000_dt0p5_short"].parent
        / "complex_material_control_p0.log",
        "smooth3D_one_sided_trace_result": PATHS["smooth_one_sided"],
        "smooth3D_discrete_epsilon_result": PATHS["smooth_discrete_epsilon"],
    }
    manifest = {
        "status": STATUS,
        "raw_files_committed": False,
        "raw_files": [artifact(role, path) for role, path in manifest_paths.items()],
        "generation_commands": [
            "python 31_run_au_temperature_density_endpoint_control.py --rho 1 --carrier-span-k 1000 ...",
            "python 31_run_au_temperature_density_endpoint_control.py --rho 1 --carrier-span-k 1000 --temperature-table ...",
            "python 32_analyze_au_smooth3d_one_sided_traces.py --output-dir <raw>",
            "python 33_validate_au_smooth3d_discrete_epsilon_shape_adjoint.py --output-dir <raw>",
            "python 34_summarize_au_temperature_carrier_and_discrete_shape.py",
        ],
        "CPU_FDTD_fallback": False,
        "thermal_electrical_PTE_or_optimization_executed": False,
    }

    offsets = np.asarray(
        [1e9 * float(row["field_trace_normal_offset_m"]) for row in one_sided["traces"]]
    )
    trace_ad = np.asarray(
        [1e30 * float(row["AD_J_proxy_per_um"]) for row in one_sided["traces"]]
    )
    fd_scaled = 1e30 * float(one_sided["FD_target_h0p05_J_proxy_per_um"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    requested = complex(*requested_epsilon)
    labels = list("xyz")
    x = np.arange(3)
    width = 0.25
    axes[0, 0].bar(x - width, [requested.real] * 3, width, label="requested Re eps")
    axes[0, 0].bar(
        x,
        [epsilon_readback[c]["epsilon_interior_median"][0] for c in labels],
        width,
        label="readback Re eps",
    )
    axes[0, 0].bar(
        x + width,
        [epsilon_readback[c]["epsilon_interior_median"][1] for c in labels],
        width,
        label="readback Im eps",
    )
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_title("Moderate complex-index carrier: exact readback")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(offsets, trace_ad, "o-", label="boundary AD trace")
    axes[0, 1].axhline(fd_scaled, color="black", ls="--", label="central FD")
    axes[0, 1].set_xlabel("field trace offset (nm); negative=inside Au")
    axes[0, 1].set_ylabel("derivative (1e-30 J-proxy/um)")
    axes[0, 1].set_title("Smooth 3-D one-sided traces: all wrong sign")
    axes[0, 1].legend(fontsize=8)

    h = [0.1, 0.05]
    fd = [
        1e30 * discrete["comparisons"][f"h_{v:g}_um"]["FD_J_proxy_per_um"]
        for v in h
    ]
    ad = [
        1e30
        * discrete["comparisons"][f"h_{v:g}_um"][
            "discrete_epsilon_AD_J_proxy_per_um"
        ]
        for v in h
    ]
    axes[1, 0].plot(h, fd, "o-", label="objective FD")
    axes[1, 0].plot(h, ad, "s-", label="discrete-epsilon AD")
    axes[1, 0].set_xlabel("central geometry step h (um)")
    axes[1, 0].set_ylabel("derivative (1e-30 J-proxy/um)")
    axes[1, 0].set_title("Conformal d-epsilon/da is not step converged")
    axes[1, 0].legend(fontsize=8)

    outcome_names = ["moderate\ncarrier", "exact Au\nlinear", "exact Au\ntable", "smooth\nboundary", "discrete\nd-eps"]
    outcome_values = [1, 0, 0, 0, 0]
    axes[1, 1].bar(outcome_names, outcome_values, color=["#2ca02c", "#d62728", "#d62728", "#d62728", "#d62728"])
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_ylabel("gate pass (1=yes)")
    axes[1, 1].set_title("No stable differentiable exact-Au route")
    fig.suptitle("v261 GPU Au topology-gradient diagnosis", fontsize=15)
    fig.tight_layout()

    (RESULTS / "au_temperature_carrier_and_smooth3d_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (RESULTS / "au_temperature_carrier_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "AU_TEMPERATURE_CARRIER_RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    fig.savefig(RESULTS / "au_temperature_carrier_and_smooth3d_diagnosis.png", dpi=180)
    plt.close(fig)

    report = f"""# Au temperature-carrier and smooth-3D gradient diagnosis

Status: `{STATUS}`

## Result

The v261 GPU temperature-grid coupling is real, but it does **not** provide a
stable exact-Au topology path.  With conformal variant 1, the moderate control
`n=2, k=0.5` reproduces `epsilon=3.75+2i` on every component grid.  It closes
`P_Q` against the six faces to `{100*moderate['six_face_closure_relative']:.6f}%`
and reaches auto-shutoff `{moderate['log_audit']['final_auto_shutoff']:.6e}`.
This numerical attribute is never a physical temperature.

The same mechanism diverges at the exact 10-um Ordal Au endpoint
`n+ik=12.1+69.2i`.  The failure remains for a 50-nm film, forward and reverse
base directions, linear and table models, and carrier spans of 1 K and 1000 K.
The latter reduces the recorded sensitivities to `dn/dT=0.0111` and
`dk/dT=0.0692`.  Reducing the FDTD stability factor from 0.99 to 0.5 still
diverges at approximately `2.49e-13 s`, so neither coefficient scaling nor a
smaller Courant time step cures the failure.  CPU FDTD fallback was prohibited.

## Boundary root cause

The installed LumOpt kernel is the standard tangential-E/normal-D continuous
shape derivative.  On the fully smooth 3-D ellipsoid, offsets from -100 to
+100 nm were evaluated without changing the surface or fitting to FD.  None
reproduces the independent central-FD sign; the best relative mismatch is
`{100*min(trace_errors):.3f}%`.

The solver-discrete conformal-Yee diagonal epsilon derivative also fails.  At
`h=0.05 um`, its contraction is
`{strong_discrete['discrete_epsilon_AD_J_proxy_per_um']:.12e}` J-proxy/um,
whereas FD is `{strong_discrete['FD_J_proxy_per_um']:.12e}` J-proxy/um.  The
sign is wrong and the discrete derivative changes by
`{100*discrete['discrete_derivative_step_relative_change']:.3f}%` between the
two steps.  This is not a coordinate-pairing error: forward/adjoint mismatch is
`{discrete['maximum_forward_adjoint_grid_mismatch_m']:.3e}` m.

## Decision

Exact scalar Au remains a valid forward GPU material, and the fixed-geometry
material adjoint remains valid.  What is blocked is a representation that is
both stable at exact Au and differentiable for topology optimization.  No Au
thermal, electrical, PTE, adjoint-chain, or optimization result is promoted.
No clipping, smoothing, empirical normalization, or gradient rescaling was
used.

A future route must expose a causal dispersive Drude/ADE oscillator-strength
Jacobian on the spatial grid, or use another Maxwell backend with a certified
metal topology adjoint.  A per-pixel full-Maxwell finite-difference Jacobian is
not an acceptable production method.

Official v261 implementation inspected:
`/opt/lumerical/v261/api/python/lumopt/utilities/gradients.py`.
Official temperature-grid documentation:
https://optics.ansys.com/hc/en-us/articles/360034901773-Temperature-dependent-refractive-index-models
"""
    (RESULTS / "AU_TEMPERATURE_CARRIER_AND_SMOOTH3D_DIAGNOSIS_REPORT.md").write_text(
        report
    )
    print(json.dumps({"status": STATUS, "results": str(RESULTS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
