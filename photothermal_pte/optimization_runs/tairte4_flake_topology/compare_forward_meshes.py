#!/usr/bin/env python3
"""Conservative native-Q comparison for one frozen density.

The same power-bin operator is used for the 100/50 nm mesh certificate and
the 40/48 um transverse-domain certificate.  Equal-power normalization is a
reported shape diagnostic only; no FDTD artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from photothermal_pte.finite_inverse_design.native_yee_q import trapezoid_weights


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_edges(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    return np.concatenate(
        ([values[0]], 0.5 * (values[:-1] + values[1:]), [values[-1]])
    )


def overlap_fraction(source_edges: np.ndarray, target_edges: np.ndarray) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    source_width = np.diff(source_edges)
    for target in range(target_edges.size - 1):
        low, high = target_edges[target : target + 2]
        first = max(0, int(np.searchsorted(source_edges, low, side="right") - 1))
        last = min(source_width.size - 1, int(np.searchsorted(source_edges, high, side="left")))
        for source in range(first, last + 1):
            overlap = max(0.0, min(high, source_edges[source + 1]) - max(low, source_edges[source]))
            if overlap > 0.0:
                rows.append(target)
                columns.append(source)
                values.append(overlap / source_width[source])
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(target_edges.size - 1, source_width.size),
    )


def map_lateral_power(data: np.lib.npyio.NpzFile, target_edges: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    total = np.zeros((target_edges.size - 1, target_edges.size - 1), dtype=np.float64)
    powers: dict[str, float] = {}
    for component in "xyz":
        q = np.asarray(data[f"Q{component}_W_m3"], dtype=np.float64)
        x = np.asarray(data[f"Q{component}_x_m"], dtype=np.float64)
        y = np.asarray(data[f"Q{component}_y_m"], dtype=np.float64)
        z = np.asarray(data[f"Q{component}_z_m"], dtype=np.float64)
        wx, wy, wz = map(trapezoid_weights, (x, y, z))
        source_power = np.einsum("ijk,k,i,j->ij", q, wz, wx, wy, optimize=True)
        fx = overlap_fraction(dual_edges(x), target_edges)
        fy = overlap_fraction(dual_edges(y), target_edges)
        mapped = fx @ source_power @ fy.T
        total += np.asarray(mapped)
        powers[component] = float(np.sum(source_power))
    return total, powers


def nrmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(second), np.finfo(float).tiny))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-json", required=True, type=Path)
    parser.add_argument("--coarse-npz", required=True, type=Path)
    parser.add_argument("--fine-json", required=True, type=Path)
    parser.add_argument("--fine-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--comparison", choices=("mesh", "domain"), default="mesh")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    coarse_result = json.loads(args.coarse_json.read_text())
    fine_result = json.loads(args.fine_json.read_text())
    if not coarse_result.get("passed") or not fine_result.get("passed"):
        raise RuntimeError("both forward gates must pass before convergence comparison")
    target_edges = np.linspace(-14e-6, 14e-6, 281)
    coarse, coarse_components = map_lateral_power(np.load(args.coarse_npz), target_edges)
    fine, fine_components = map_lateral_power(np.load(args.fine_npz), target_edges)
    coarse_power = float(np.sum(coarse))
    fine_power = float(np.sum(fine))
    power_change = abs(coarse_power - fine_power) / max(abs(fine_power), np.finfo(float).tiny)
    raw_nrmse = nrmse(coarse, fine)
    coarse_normalized = coarse / coarse_power
    fine_normalized = fine / fine_power
    shape_nrmse = nrmse(coarse_normalized, fine_normalized)
    passed = bool(power_change < 0.005 and shape_nrmse < 0.005)
    stem = (
        "tairte4_flake_100nm_50nm"
        if args.comparison == "mesh"
        else "tairte4_flake_40um_48um"
    )
    raw = output / f"{stem}_lateral_Q.npz"
    np.savez_compressed(
        raw,
        common_x_edges_m=target_edges,
        common_y_edges_m=target_edges,
        coarse_power_per_bin_W=coarse,
        fine_power_per_bin_W=fine,
        coarse_normalized=coarse_normalized,
        fine_normalized=fine_normalized,
    )
    status_root = (
        "TAIRTE4_FLAKE_100NM_OPTICAL_MESH"
        if args.comparison == "mesh"
        else "TAIRTE4_FLAKE_40UM_OPTICAL_DOMAIN"
    )
    result = {
        "status": f"VALIDATED_{status_root}" if passed else f"FAILED_{status_root}",
        "passed": passed,
        "scope": "uniform rho=0.5 E||a; native component Q conservatively accumulated to common 100 nm lateral power bins",
        "comparison": args.comparison,
        "candidate_labels": (
            {"coarse_interface_xy_nm": 100.0, "fine_interface_xy_nm": 50.0}
            if args.comparison == "mesh"
            else {"baseline_domain_um": 40.0, "expanded_domain_um": 48.0}
        ),
        "P_Q_coarse_W": coarse_result["P_Q_W"],
        "P_Q_fine_W": fine_result["P_Q_W"],
        "common_grid_integrated_coarse_W": coarse_power,
        "common_grid_integrated_fine_W": fine_power,
        "component_power_coarse_W": coarse_components,
        "component_power_fine_W": fine_components,
        "relative_total_power_change": power_change,
        "raw_lateral_power_NRMSE": raw_nrmse,
        "equal_power_normalized_lateral_shape_NRMSE": shape_nrmse,
        "normalization_use": "diagnostic comparison only; no source or solver artifact was rescaled",
        "gates": {"total_power_change_max": 0.005, "normalized_lateral_shape_NRMSE_max": 0.005},
        "artifact": {"path": str(raw), "size_bytes": raw.stat().st_size, "sha256": sha256(raw)},
    }
    path = output / f"{stem}_{args.comparison}_comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
