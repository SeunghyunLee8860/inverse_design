#!/usr/bin/env python3
"""Conservatively remap component-native Yee Q to material thermal cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


MATERIALS = ("au", "tairte4", "sio2")
COMPONENTS = ("x", "y", "z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def _primal_edges(coordinate: np.ndarray, width: np.ndarray) -> np.ndarray:
    lower = coordinate - 0.5 * width
    upper = coordinate + 0.5 * width
    mismatch = float(np.max(np.abs(upper[:-1] - lower[1:]))) if len(width) > 1 else 0.0
    if mismatch > 5.0e-13:
        raise RuntimeError(f"Non-contiguous primal cell edges: {mismatch:.6e} m")
    return np.concatenate((lower[:1], upper))


def _overlap_operator(
    coordinate: np.ndarray,
    width: np.ndarray,
    target_edges: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Map source dual-cell power to target cells by normalized overlap."""
    target_count = len(target_edges) - 1
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    retained = np.zeros(len(coordinate), dtype=np.float64)
    domain_low, domain_high = target_edges[0], target_edges[-1]
    for source_index, (center, full_width) in enumerate(zip(coordinate, width, strict=True)):
        full_low = center - 0.5 * full_width
        full_high = center + 0.5 * full_width
        low = max(full_low, domain_low)
        high = min(full_high, domain_high)
        denominator = high - low
        if denominator <= 0.0:
            raise RuntimeError(
                f"Source dual cell {source_index} has no material-domain overlap"
            )
        retained[source_index] = denominator / full_width
        first = max(int(np.searchsorted(target_edges, low, side="right")) - 1, 0)
        last = min(int(np.searchsorted(target_edges, high, side="left")) + 1, target_count)
        for target_index in range(first, last):
            overlap = max(
                0.0,
                min(high, target_edges[target_index + 1])
                - max(low, target_edges[target_index]),
            )
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(source_index)
                values.append(overlap / denominator)
    operator = sparse.coo_matrix(
        (values, (rows, columns)), shape=(target_count, len(coordinate))
    ).tocsr()
    column_sum = np.asarray(operator.sum(axis=0)).reshape(-1)
    if not np.allclose(column_sum, 1.0, rtol=0.0, atol=2.0e-13):
        raise RuntimeError(
            f"Overlap operator is not conservative: max error {np.max(np.abs(column_sum - 1.0))}"
        )
    return operator, retained


def _apply_axis(array: np.ndarray, operator: sparse.spmatrix, axis: int) -> np.ndarray:
    moved = np.moveaxis(array, axis, 0)
    transformed = operator @ moved.reshape(moved.shape[0], -1)
    reshaped = np.asarray(transformed).reshape((operator.shape[0],) + moved.shape[1:])
    return np.moveaxis(reshaped, 0, axis)


def _forward(
    array: np.ndarray,
    operators: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
) -> np.ndarray:
    result = array
    for axis, operator in enumerate(operators):
        result = _apply_axis(result, operator, axis)
    return result


def _transpose(
    array: np.ndarray,
    operators: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
) -> np.ndarray:
    result = array
    for axis in reversed(range(3)):
        result = _apply_axis(result, operators[axis].transpose().tocsr(), axis)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-summary-json", required=True, type=Path)
    parser.add_argument("--raw-spatial-npz", required=True, type=Path)
    parser.add_argument("--raw-remap-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    summary_path = args.spatial_summary_json.resolve()
    source_raw_path = args.raw_spatial_npz.resolve()
    remap_raw_path = args.raw_remap_npz.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_sha = _sha256(source_raw_path)
    if source_sha != source_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed: spatial-Q source SHA does not match its summary")

    rng = np.random.default_rng(260821)
    rows: list[dict[str, object]] = []
    output_payload: dict[str, np.ndarray] = {}
    target_q: dict[str, np.ndarray] = {}
    target_edges_by_material: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    maximum_power_error = 0.0
    maximum_dot_error = 0.0
    maximum_axis_column_error = 0.0
    maximum_boundary_redistributed_power_fraction = 0.0
    finite_nonnegative = True

    with np.load(source_raw_path, allow_pickle=False) as source:
        for material in MATERIALS:
            q_native = np.asarray(source[f"Q_{material}_W_m3"], dtype=np.float64)
            volume_native = np.asarray(
                source[f"dual_volume_{material}_m3"], dtype=np.float64
            )
            # The target thermal cells are the material's primal FDTD cells:
            # x from Ex centres, y from Ey centres, z from Ez centres.
            target_edges = []
            for axis in COMPONENTS:
                coordinate = np.asarray(source[f"{material}_{axis}_{axis}_m"])
                width = np.asarray(source[f"dual_width_{material}_{axis}_{axis}_m"])
                target_edges.append(_primal_edges(coordinate, width))
            target_edges_tuple = tuple(target_edges)
            target_edges_by_material[material] = target_edges_tuple
            dx, dy, dz = (np.diff(edges) for edges in target_edges_tuple)
            target_volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
            target_power_components = []

            for component_index, component in enumerate(COMPONENTS):
                operators = []
                retained_axes = []
                for axis in COMPONENTS:
                    coordinate = np.asarray(source[f"{material}_{component}_{axis}_m"])
                    width = np.asarray(
                        source[f"dual_width_{material}_{component}_{axis}_m"]
                    )
                    operator, retained = _overlap_operator(
                        coordinate, width, target_edges_tuple[COMPONENTS.index(axis)]
                    )
                    operators.append(operator)
                    retained_axes.append(retained)
                    column_error = float(
                        np.max(np.abs(np.asarray(operator.sum(axis=0)).reshape(-1) - 1.0))
                    )
                    maximum_axis_column_error = max(
                        maximum_axis_column_error, column_error
                    )
                operators_tuple = tuple(operators)
                native_power = q_native[component_index] * volume_native[component_index]
                remapped_power = _forward(native_power, operators_tuple)
                target_power_components.append(remapped_power)
                source_power = float(np.sum(native_power))
                target_power_value = float(np.sum(remapped_power))
                power_error = _relative(source_power, target_power_value)
                maximum_power_error = max(maximum_power_error, power_error)
                retained_volume_fraction = (
                    retained_axes[0][:, None, None]
                    * retained_axes[1][None, :, None]
                    * retained_axes[2][None, None, :]
                )
                redistributed_fraction = float(
                    np.sum(native_power * (1.0 - retained_volume_fraction))
                    / max(source_power, np.finfo(float).tiny)
                )
                maximum_boundary_redistributed_power_fraction = max(
                    maximum_boundary_redistributed_power_fraction,
                    redistributed_fraction,
                )

                test_target = rng.standard_normal(remapped_power.shape)
                lhs = float(np.vdot(_forward(native_power, operators_tuple), test_target))
                rhs = float(np.vdot(native_power, _transpose(test_target, operators_tuple)))
                dot_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)
                maximum_dot_error = max(maximum_dot_error, dot_error)
                rows.append(
                    {
                        "material": material,
                        "component": component,
                        "source_shape": "x".join(map(str, native_power.shape)),
                        "target_shape": "x".join(map(str, remapped_power.shape)),
                        "source_power_W": source_power,
                        "target_power_W": target_power_value,
                        "power_relative_error": power_error,
                        "transpose_dot_relative_error": dot_error,
                        "boundary_dual_power_redistributed_fraction": redistributed_fraction,
                    }
                )
                output_payload[f"power_{material}_{component}_W"] = remapped_power.astype(
                    np.float32
                )

            total_power = np.sum(np.stack(target_power_components), axis=0)
            q_thermal = total_power / target_volume
            finite_nonnegative &= bool(
                np.all(np.isfinite(q_thermal)) and np.all(q_thermal >= 0.0)
            )
            target_q[material] = q_thermal
            output_payload[f"Q_{material}_thermal_W_m3"] = q_thermal.astype(np.float32)
            for axis, edges in zip(COMPONENTS, target_edges_tuple, strict=True):
                output_payload[f"{material}_{axis}_edges_m"] = edges

    remap_raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(remap_raw_path, **output_payload)
    remap_sha = _sha256(remap_raw_path)

    source_total = float(sum(row["source_power_W"] for row in rows))
    target_total = float(sum(row["target_power_W"] for row in rows))
    total_power_error = _relative(source_total, target_total)
    source_summary_error = _relative(source_total, source_summary["P_Q_reintegrated_W"])
    gates = {
        "source_SHA_matches_validated_spatial_Q": True,
        "finite_nonnegative_thermal_Q": finite_nonnegative,
        "axis_overlap_column_sum_error_lt_1e-12": maximum_axis_column_error < 1.0e-12,
        "component_power_conservation_lt_1e-12": maximum_power_error < 1.0e-12,
        "total_power_conservation_lt_1e-12": total_power_error < 1.0e-12,
        "transpose_dot_error_lt_1e-12": maximum_dot_error < 1.0e-12,
        "source_power_matches_spatial_summary_lt_1e-6": source_summary_error < 1.0e-6,
        "no_source_deletion_clipping_smoothing_gain_or_global_rescaling": True,
    }
    passed = all(gates.values())
    status = (
        "VALIDATED_FDTDX_SPATIAL_Q_CONSERVATIVE_MATERIAL_OVERLAP_REMAP"
        if passed
        else "FAILED_FDTDX_SPATIAL_Q_CONSERVATIVE_MATERIAL_OVERLAP_REMAP"
    )

    csv_path = output / "fdtdx_material_overlap_remap_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for axis, material in zip(axes, MATERIALS, strict=True):
        edges_x, edges_y, edges_z = target_edges_by_material[material]
        depth_integrated = np.sum(
            target_q[material] * np.diff(edges_z)[None, None, :], axis=2
        )
        mesh = axis.pcolormesh(
            edges_x * 1.0e6,
            edges_y * 1.0e6,
            depth_integrated.T,
            shading="flat",
        )
        axis.set_aspect("equal")
        axis.set_title(f"{material}: remapped total Q")
        axis.set_xlabel("x=b (um)")
        axis.set_ylabel("y=a (um)")
        fig.colorbar(mesh, ax=axis, label="depth-integrated Q (W/m2)")
    q_plot_path = output / "fdtdx_material_overlap_thermal_q.png"
    fig.savefig(q_plot_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    labels = [f"{row['material']} Q{row['component']}" for row in rows]
    errors = [100.0 * row["power_relative_error"] for row in rows]
    redistributed = [
        100.0 * row["boundary_dual_power_redistributed_fraction"] for row in rows
    ]
    axes[0].barh(labels, errors)
    axes[0].set_xlabel("power conservation error (%)")
    axes[0].set_title("Component-wise conservative remap")
    axes[1].barh(labels, redistributed)
    axes[1].set_xlabel("power in boundary-crossing dual support (%)")
    axes[1].set_title("Redistributed inside absorbing material")
    audit_plot_path = output / "fdtdx_material_overlap_remap_audit.png"
    fig.savefig(audit_plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "component-native Yee dual-cell power to material-primal thermal-cell "
            "overlap remap for Au, TaIrTe4, and SiO2; no thermal solve, PTE, Maxwell "
            "adjoint, combined gradient, or optimization"
        ),
        "input_spatial_Q": {
            "summary": str(summary_path),
            "raw_path": str(source_raw_path),
            "bytes": source_raw_path.stat().st_size,
            "sha256": source_sha,
        },
        "output_thermal_Q": {
            "raw_path": str(remap_raw_path),
            "bytes": remap_raw_path.stat().st_size,
            "sha256": remap_sha,
            "committed_to_git": False,
        },
        "source_total_power_W": source_total,
        "target_total_power_W": target_total,
        "total_power_relative_error": total_power_error,
        "maximum_component_power_relative_error": maximum_power_error,
        "maximum_transpose_dot_relative_error": maximum_dot_error,
        "maximum_axis_overlap_column_sum_error": maximum_axis_column_error,
        "maximum_boundary_dual_power_redistributed_fraction": (
            maximum_boundary_redistributed_power_fraction
        ),
        "mapping_contract": (
            "p_source=Q_native*V_dual; p_source is distributed by exact separable "
            "intersection with the absorbing material's primal thermal cells, normalized "
            "only within that source-cell/material intersection; Q_thermal=sum(p)/V_thermal"
        ),
        "gates": gates,
        "next_gate": (
            "solve one explicit heterogeneous Au/TaIrTe4/SiO2 thermal control with this "
            "Q, audit energy balance, then compare against fixed-Q coupled operator"
        ),
    }
    summary_json_path = output / "fdtdx_material_overlap_remap_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# FDTDX spatial-Q material-overlap thermal remap

Status: **{status}**

For every Au, TaIrTe4, and SiO2 `Qx/Qy/Qz` Yee sample, the source power is
first formed as `p=Q*V_dual`. Its component-specific dual-cell bounds are then
intersected with the actual absorbing-material thermal cells. The already
calculated source-cell power is distributed only among those overlaps.

This is not nearest-cell projection and it does not delete a boundary sample,
assign air absorption to TaIrTe4, or apply a global gain. The per-cell overlap
weights sum to one. A boundary-crossing dual cell is handled by its exact
material overlap rather than by an array-index convention.

| metric | value |
|---|---:|
| source total power | {source_total:.12e} W |
| remapped total power | {target_total:.12e} W |
| total conservation error | {100*total_power_error:.12f}% |
| worst component conservation error | {100*maximum_power_error:.12f}% |
| worst transpose dot-test error | {maximum_dot_error:.3e} |
| worst overlap-column error | {maximum_axis_column_error:.3e} |
| largest boundary-dual redistributed power fraction | {100*maximum_boundary_redistributed_power_fraction:.6f}% |

The last quantity is diagnostic: it reports power whose native Yee dual
support crosses a material boundary and is therefore conservatively placed
inside the actual absorbing material. It is not discarded power and is not a
physical air heat source.

The output thermal-Q NPZ is not committed to Git. This checkpoint validates
only the remap and its transpose. No temperature, PTE current, combined
Maxwell gradient, or optimization result is claimed yet.
"""
    report_path = output / "FDTDX_MATERIAL_OVERLAP_THERMAL_REMAP_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = (summary_path, summary_json_path, csv_path, q_plot_path, audit_plot_path, report_path)
    manifest = {
        "status": status,
        "input_raw": summary["input_spatial_Q"],
        "output_raw": summary["output_thermal_Q"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
