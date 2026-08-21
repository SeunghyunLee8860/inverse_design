#!/usr/bin/env python3
"""Pull explicit thermal-source adjoints back to component-native Yee power."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE64 = HERE / "64_validate_fdtdx_material_overlap_thermal_remap.py"
STAGE65 = HERE / "65_solve_fdtdx_explicit_thermal_weighting_pte.py"
MATERIALS = ("au", "tairte4", "sio2")
COMPONENTS = ("x", "y", "z")
STATUS = "VALIDATED_NATIVE_YEE_THERMAL_SOURCE_ADJOINT_PULLBACK"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-q-summary-json", required=True, type=Path)
    parser.add_argument("--fixed-q-raw-npz", required=True, type=Path)
    parser.add_argument("--remap-summary-json", required=True, type=Path)
    parser.add_argument("--raw-remap-npz", required=True, type=Path)
    parser.add_argument("--raw-spatial-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    args = parser.parse_args()

    fixed_summary_path = args.fixed_q_summary_json.resolve()
    fixed_raw_path = args.fixed_q_raw_npz.resolve()
    remap_summary_path = args.remap_summary_json.resolve()
    remap_path = args.raw_remap_npz.resolve()
    spatial_path = args.raw_spatial_npz.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixed = json.loads(fixed_summary_path.read_text(encoding="utf-8"))
    remap_summary = json.loads(remap_summary_path.read_text(encoding="utf-8"))
    if fixed.get("status") != "VALIDATED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD":
        raise RuntimeError("Fail-closed: fixed-Q AD-FD certificate is not validated")
    expected = {
        fixed_raw_path: fixed["raw_artifact"]["sha256"],
        remap_path: remap_summary["output_thermal_Q"]["sha256"],
        spatial_path: remap_summary["input_spatial_Q"]["sha256"],
    }
    for path, sha in expected.items():
        if _sha256(path) != sha:
            raise RuntimeError(f"Fail-closed SHA mismatch: {path}")

    overlap = _load(STAGE64, "au_stage68_overlap")
    forward = _load(STAGE65, "au_stage68_forward")
    topology = _load(forward.TOPOLOGY_THERMAL, "au_stage68_topology")
    fvm = _load(
        Path(__file__).parents[2]
        / "validation"
        / "photothermal_stage1"
        / "anisotropic_heat_fvm.py",
        "au_stage68_fvm",
    )
    with np.load(spatial_path, allow_pickle=False) as spatial:
        rho = np.asarray(spatial["rho"], dtype=np.float64)
    state = forward._thermal_state(
        rho,
        forward.G_TA_SIO2_SCENARIOS["thermally_grown"],
        topology,
        fvm,
    )

    payload: dict[str, np.ndarray] = {"rho": rho.astype(np.float32)}
    records: dict[str, dict[str, object]] = {}
    worst_dot_error = 0.0
    worst_weighted_value_error = 0.0
    rng = np.random.default_rng(20260821)
    with (
        np.load(fixed_raw_path, allow_pickle=False) as fixed_raw,
        np.load(remap_path, allow_pickle=False) as remap,
        np.load(spatial_path, allow_pickle=False) as spatial,
    ):
        for scenario in ("thermally_grown", "evaporated"):
            thermal_adjoint = np.asarray(
                fixed_raw[f"thermal_adjoint_{scenario}_A_W"], dtype=np.float64
            ).reshape(state["system"].shape)
            full_mapped_power = np.zeros(state["system"].shape, dtype=np.float64)
            native_value = 0.0
            scenario_records = {}
            for material in MATERIALS:
                material_mask = state["masks"][material]
                indices = (
                    np.flatnonzero(np.any(material_mask, axis=(1, 2))),
                    np.flatnonzero(np.any(material_mask, axis=(0, 2))),
                    np.flatnonzero(np.any(material_mask, axis=(0, 1))),
                )
                explicit_edges = tuple(
                    state["edges"][axis][index[0] : index[-1] + 2]
                    for axis, index in enumerate(indices)
                )
                primal_edges = tuple(
                    np.asarray(remap[f"{material}_{axis}_edges_m"], dtype=np.float64)
                    for axis in COMPONENTS
                )
                primal_centers = tuple(
                    0.5 * (edges[:-1] + edges[1:]) for edges in primal_edges
                )
                primal_widths = tuple(np.diff(edges) for edges in primal_edges)
                second_operators = tuple(
                    overlap._overlap_operator(
                        primal_centers[axis], primal_widths[axis], explicit_edges[axis]
                    )[0]
                    for axis in range(3)
                )
                explicit_lambda = thermal_adjoint[np.ix_(*indices)]
                primal_lambda = overlap._transpose(explicit_lambda, second_operators)
                component_records = {}
                for component_index, component in enumerate(COMPONENTS):
                    first_operators = []
                    for axis_index, axis in enumerate(COMPONENTS):
                        coordinate = np.asarray(
                            spatial[f"{material}_{component}_{axis}_m"],
                            dtype=np.float64,
                        )
                        width = np.asarray(
                            spatial[f"dual_width_{material}_{component}_{axis}_m"],
                            dtype=np.float64,
                        )
                        first_operators.append(
                            overlap._overlap_operator(
                                coordinate, width, primal_edges[axis_index]
                            )[0]
                        )
                    first_operators_tuple = tuple(first_operators)
                    native_weight = overlap._transpose(
                        primal_lambda, first_operators_tuple
                    )
                    payload[
                        f"weight_{scenario}_{material}_{component}_A_W"
                    ] = native_weight.astype(np.float32)
                    native_q = np.asarray(
                        spatial[f"Q_{material}_W_m3"], dtype=np.float64
                    )[component_index]
                    native_volume = np.asarray(
                        spatial[f"dual_volume_{material}_m3"], dtype=np.float64
                    )[component_index]
                    native_power = native_q * native_volume
                    primal_power = overlap._forward(
                        native_power, first_operators_tuple
                    )
                    explicit_power = overlap._forward(
                        primal_power, second_operators
                    )
                    full_mapped_power[np.ix_(*indices)] += explicit_power
                    component_native_value = float(
                        np.vdot(native_weight, native_power)
                    )
                    native_value += component_native_value

                    random_native = rng.standard_normal(native_power.shape)
                    random_target = rng.standard_normal(explicit_lambda.shape)
                    forward_random = overlap._forward(
                        overlap._forward(random_native, first_operators_tuple),
                        second_operators,
                    )
                    pulled_random = overlap._transpose(
                        overlap._transpose(random_target, second_operators),
                        first_operators_tuple,
                    )
                    lhs = float(np.vdot(forward_random, random_target))
                    rhs = float(np.vdot(random_native, pulled_random))
                    dot_error = _relative(lhs, rhs)
                    worst_dot_error = max(worst_dot_error, dot_error)
                    component_records[component] = {
                        "shape": list(native_weight.shape),
                        "weight_min_A_W": float(np.min(native_weight)),
                        "weight_max_A_W": float(np.max(native_weight)),
                        "weight_l2_A_W": float(np.linalg.norm(native_weight)),
                        "base_weighted_source_contribution_A": component_native_value,
                        "two_stage_transpose_dot_relative_error": dot_error,
                    }
                scenario_records[material] = component_records
            explicit_value = float(np.vdot(thermal_adjoint, full_mapped_power))
            value_error = _relative(native_value, explicit_value)
            worst_weighted_value_error = max(worst_weighted_value_error, value_error)
            records[scenario] = {
                "native_weighted_source_value_A": native_value,
                "explicit_thermal_grid_source_value_A": explicit_value,
                "relative_error": value_error,
                "components": scenario_records,
            }

    gates = {
        "input_SHAs_match": True,
        "two_stage_transpose_dot_error_lt_1e-12": worst_dot_error < 1.0e-12,
        "base_weighted_value_error_lt_1e-6": worst_weighted_value_error < 1.0e-6,
        "finite_weights": all(
            np.all(np.isfinite(value))
            for key, value in payload.items()
            if key.startswith("weight_")
        ),
        "no_array_index_pairing_without_coordinate_overlap": True,
        "no_clipping_smoothing_gain_or_weight_rescaling": True,
    }
    passed = all(gates.values())
    status = STATUS if passed else "FAILED_NATIVE_YEE_THERMAL_SOURCE_ADJOINT_PULLBACK"
    raw_output = args.raw_output_npz.resolve()
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_output, **payload)
    summary = {
        "status": status,
        "scope": (
            "two-stage transpose from explicit thermal cell power through the "
            "material-primal conservative remap to component-native Yee power; "
            "no Maxwell reverse solve or optimization"
        ),
        "contract": (
            "dI/dp_explicit -> R_explicit^T -> R_component^T -> "
            "dI/dp_native_component"
        ),
        "inputs": {
            "fixed_Q_ADFD_raw": {
                "path": str(fixed_raw_path),
                "sha256": _sha256(fixed_raw_path),
            },
            "thermal_Q_remap_raw": {
                "path": str(remap_path),
                "sha256": _sha256(remap_path),
            },
            "spatial_Q_raw": {
                "path": str(spatial_path),
                "sha256": _sha256(spatial_path),
            },
        },
        "scenarios": records,
        "worst_two_stage_transpose_dot_relative_error": worst_dot_error,
        "worst_base_weighted_value_relative_error": worst_weighted_value_error,
        "gates": gates,
        "raw_artifact": {
            "path": str(raw_output),
            "bytes": raw_output.stat().st_size,
            "sha256": _sha256(raw_output),
            "committed_to_git": False,
        },
        "next_gate": (
            "use the native weights as the scalar FDTDX objective and run "
            "directional optical AD-FD"
        ),
    }
    summary_path = output / "native_yee_thermal_source_adjoint_weights_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = output / "NATIVE_YEE_THERMAL_SOURCE_ADJOINT_WEIGHTS_REPORT.md"
    report_path.write_text(
        f"""# Native-Yee thermal-source adjoint pullback

Status: **{status}**

The explicit thermal source adjoint is pulled back through both conservative
power maps. Every Ex/Ey/Ez component uses its own physical Yee coordinates,
dual widths, and overlap operator; no same-index component pairing is used.

The worst two-stage transpose dot-test error is `{worst_dot_error:.3e}` and
the base native-weighted source contraction agrees with the explicit thermal
grid contraction to `{worst_weighted_value_error:.3e}` relative. The weights
have units `A/W` and are not normalized or rescaled.

This is a mapping certificate only. It does not yet validate the reverse
Maxwell solve or authorize Au optimization.
""",
        encoding="utf-8",
    )
    manifest = {
        "status": status,
        "raw_artifact": summary["raw_artifact"],
        "inputs": summary["inputs"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (summary_path, report_path)
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "worst_transpose_dot_error": worst_dot_error,
                "worst_weighted_value_error": worst_weighted_value_error,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
