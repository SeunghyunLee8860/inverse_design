#!/usr/bin/env python3
"""Offline/CUDA decomposition of a selected-grid combined AD--FD failure.

No Maxwell solve is performed.  The script reuses SHA-pinned base/plus/minus
native-Q artifacts, remaps them conservatively, and evaluates optical-only
and thermal-material-only centered differences with CUDA thermal solves.
"""

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

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (  # noqa: E402
    PersistentCudaCSR,
)
from photothermal_pte.finite_inverse_design.finite_q_mapping import (  # noqa: E402
    nodal_control_volume_edges,
)
from run_production_combined_adfd_smoke import (  # noqa: E402
    SCENARIO,
    boundary_energy,
    build_state,
    map_q,
)
from selected_thermal_density_mapping import (  # noqa: E402
    selected_nodal_to_thermal_cell,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def load_native(path: Path, expected_sha: str) -> dict[str, object]:
    if sha256(path) != expected_sha:
        raise RuntimeError(f"native-Q SHA mismatch: {path}")
    source = np.load(path)
    components: dict[str, np.ndarray] = {}
    coordinates: dict[str, dict[str, np.ndarray]] = {}
    power = 0.0
    for component in "xyz":
        values = np.asarray(source[f"Q{component}_W_m3"], float)
        component_coordinates = {
            axis: np.asarray(source[f"Q{component}_{axis}_m"], float)
            for axis in "xyz"
        }
        edges = tuple(
            nodal_control_volume_edges(component_coordinates[axis])
            for axis in "xyz"
        )
        volume = (
            np.diff(edges[0])[:, None, None]
            * np.diff(edges[1])[None, :, None]
            * np.diff(edges[2])[None, None, :]
        )
        power += float(np.sum(values * volume))
        components[component] = values
        coordinates[component] = component_coordinates
    return {
        "Q_components": components,
        "native_coordinates": coordinates,
        "P_Q_W": power,
    }


def objective(
    q_path: Path,
    q_sha: str,
    rho: np.ndarray,
    cuda_device: int,
) -> dict[str, float]:
    q = load_native(q_path, q_sha)
    data, mapping = map_q(q, design_half_span_m=9.3e-6)
    state = build_state(
        data,
        SCENARIO,
        selected_nodal_to_thermal_cell(rho),
    )
    solve = PersistentCudaCSR(
        state.system.matrix_W_K, cuda_device=cuda_device
    ).solve(
        state.source_power_W,
        relative_tolerance=1.0e-10,
        max_iterations=30000,
    )
    return {
        "objective_A": float(np.dot(state.c_A_K, solve.solution)),
        "P_Q_W": float(q["P_Q_W"]),
        "mapped_power_W": float(mapping["mapped_power_W"]),
        "mapping_relative_error": float(mapping["internal_relative_power_error"]),
        "thermal_residual": float(solve.explicit_relative_residual),
        "thermal_energy_balance": float(boundary_energy(state, solve.solution)),
        "thermal_solve_seconds": float(solve.solve_seconds),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-directory", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.combined_directory.expanduser().resolve()
    result_path = directory / "production_combined_adfd_smoke_result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "FAILED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE":
        raise RuntimeError("input is not the preserved failed combined checkpoint")
    raw_path = Path(result["raw_artifact"]["path"])
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("combined raw artifact SHA mismatch")
    raw = np.load(raw_path)
    rho = np.asarray(raw["rho"], float)
    direction = np.asarray(raw["direction"], float)
    step = float(result["step"])
    plus_rho = rho + step * direction
    minus_rho = rho - step * direction
    q_records = {
        "base": result["base_forward"]["native_Q"],
        "plus": result["FD_pair"]["plus"]["forward"]["native_Q"],
        "minus": result["FD_pair"]["minus"]["forward"]["native_Q"],
    }

    # Four counterfactual combinations isolate the two partial derivatives.
    cases = {
        "plus_Q_base_rho": objective(
            Path(q_records["plus"]["path"]),
            q_records["plus"]["sha256"],
            rho,
            args.cuda_device,
        ),
        "minus_Q_base_rho": objective(
            Path(q_records["minus"]["path"]),
            q_records["minus"]["sha256"],
            rho,
            args.cuda_device,
        ),
        "base_Q_plus_rho": objective(
            Path(q_records["base"]["path"]),
            q_records["base"]["sha256"],
            plus_rho,
            args.cuda_device,
        ),
        "base_Q_minus_rho": objective(
            Path(q_records["base"]["path"]),
            q_records["base"]["sha256"],
            minus_rho,
            args.cuda_device,
        ),
    }
    fd_optical = (
        cases["plus_Q_base_rho"]["objective_A"]
        - cases["minus_Q_base_rho"]["objective_A"]
    ) / (2.0 * step)
    fd_thermal = (
        cases["base_Q_plus_rho"]["objective_A"]
        - cases["base_Q_minus_rho"]["objective_A"]
    ) / (2.0 * step)
    ad_optical = float(np.sum(np.asarray(raw["gradient_optical_A"]) * direction))
    ad_thermal = float(np.sum(np.asarray(raw["gradient_thermal_A"]) * direction))
    fd_combined = float(result["finite_difference_directional_A"])
    ad_combined = float(result["adjoint_directional_A"])
    cross = fd_combined - (fd_optical + fd_thermal)
    diagnostics = {
        "status": "DIAGNOSED_SELECTED_COMBINED_ADFD_FAILURE",
        "Maxwell_solves": 0,
        "thermal_solves": 4,
        "CUDA_thermal_only": True,
        "optimizer_started": False,
        "input": {
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "raw": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "step": step,
        },
        "directional_derivatives_A": {
            "AD_optical": ad_optical,
            "FD_optical_fixed_thermal_material": fd_optical,
            "AD_thermal_material": ad_thermal,
            "FD_thermal_material_fixed_Q": fd_thermal,
            "AD_combined": ad_combined,
            "FD_combined": fd_combined,
            "FD_cross_term": cross,
        },
        "relative_errors": {
            "optical": relative(ad_optical, fd_optical),
            "thermal_material": relative(ad_thermal, fd_thermal),
            "combined": relative(ad_combined, fd_combined),
            "cross_term_relative_to_combined_FD": abs(cross)
            / max(abs(fd_combined), np.finfo(float).tiny),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
