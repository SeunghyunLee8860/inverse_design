from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
NEW_ROOT = HERE.parent
BASELINE = Path(
    os.environ.get(
        "TAIRTE4_PTE_BASELINE",
        str(NEW_ROOT.parent / "pte_electrode_optimizer"),
    )
).expanduser().resolve()
if not (BASELINE / "src" / "tairte4_pte" / "electrical.py").is_file():
    raise FileNotFoundError(
        "baseline pte_electrode_optimizer was not found; set "
        "TAIRTE4_PTE_BASELINE to its checkout path"
    )
sys.path.insert(0, str(BASELINE / "src"))

from tairte4_pte.config import load_config  # noqa: E402
from tairte4_pte.electrical import Electrode, ElectricalModel  # noqa: E402


def current_vector(model: ElectricalModel, temperature: np.ndarray) -> np.ndarray:
    mesh = model.mesh
    tri = mesh.triangles
    grad_temperature = np.einsum(
        "eai,ei->ea", mesh.gradients_m_inv, temperature.reshape(-1)[tri]
    )
    local = -model.thickness_m * mesh.area_m2[:, None] * np.einsum(
        "eai,eab,eb->ei", mesh.gradients_m_inv, model.alpha, grad_temperature
    )
    vector = np.zeros(mesh.nodes_m.shape[0], dtype=float)
    np.add.at(vector, tri.ravel(), local.ravel())
    return vector


def relative_sparse_asymmetry(matrix: sparse.spmatrix) -> float:
    delta = (matrix - matrix.T).tocoo()
    numerator = np.linalg.norm(delta.data) if delta.nnz else 0.0
    denominator = np.linalg.norm(matrix.data)
    return float(numerator / denominator)


def node_signature(nodes: np.ndarray) -> str:
    return ":".join(str(int(v)) for v in nodes)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    config = load_config(BASELINE / "configs" / "per_beam_500nm.json")
    fields = np.load(
        BASELINE / "results" / "per_beam_500nm_final" / "per_beam_fields.npz"
    )
    centers = np.asarray(fields["beam_centers_m"])
    beam_index = int(np.argmin(np.linalg.norm(centers, axis=1)))
    temperature = np.asarray(fields["temperature_nodes_K"][beam_index], float)
    model = ElectricalModel(config)
    result_json = json.loads((
        BASELINE / "results" / "per_beam_500nm_final" / "per_beam_results.json"
    ).read_text(encoding="utf-8"))
    record = result_json["per_beam_results"][beam_index]
    e0_data, e1_data = record["best_electrode_0"], record["best_electrode_1"]
    electrode_0 = Electrode(e0_data["side"], e0_data["center_m"], e0_data["length_m"])
    electrode_1 = Electrode(e1_data["side"], e1_data["center_m"], e1_data["length_m"])
    forward = model.evaluate(electrode_0, electrode_1, temperature)
    swapped = model.evaluate(electrode_1, electrode_0, temperature)
    q = current_vector(model, temperature)
    current_q = float(q @ forward.weighting_potential.reshape(-1))

    fixed = np.unique(np.concatenate((forward.electrode_0_nodes, forward.electrode_1_nodes)))
    free_mask = np.ones(model.mesh.nodes_m.shape[0], dtype=bool)
    free_mask[fixed] = False
    reduced = model.matrix[free_mask][:, free_mask].tocsr()

    doubled_config = copy.deepcopy(config)
    doubled_config["geometry"]["flake_thickness_nm"] *= 2.0
    doubled = ElectricalModel(doubled_config).evaluate(electrode_0, electrode_1, temperature)

    center_rows: list[dict] = []
    fixed_other = Electrode("bottom", 0.0, 8.0e-6)
    for center_um in np.arange(-4.0, 4.0 + 0.0125, 0.025):
        moving = Electrode("top", float(center_um * 1e-6), 6.0e-6)
        value = model.evaluate(moving, fixed_other, temperature)
        center_rows.append({
            "center_um": float(center_um),
            "current_A": value.short_circuit_current_A,
            "objective_A2": value.short_circuit_current_A**2,
            "node_signature": node_signature(value.electrode_0_nodes),
            "node_count": int(value.electrode_0_nodes.size),
        })

    length_rows: list[dict] = []
    for length_um in np.arange(2.0, 10.0 + 0.0125, 0.025):
        moving = Electrode("top", 0.0, float(length_um * 1e-6))
        value = model.evaluate(moving, fixed_other, temperature)
        length_rows.append({
            "length_um": float(length_um),
            "current_A": value.short_circuit_current_A,
            "objective_A2": value.short_circuit_current_A**2,
            "node_signature": node_signature(value.electrode_0_nodes),
            "node_count": int(value.electrode_0_nodes.size),
        })

    base_center = 0.10
    base_length = 6.0
    fd_rows = []
    for h_um in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.49, 0.51):
        plus = model.evaluate(
            Electrode("top", (base_center + h_um) * 1e-6, base_length * 1e-6),
            fixed_other,
            temperature,
        )
        minus = model.evaluate(
            Electrode("top", (base_center - h_um) * 1e-6, base_length * 1e-6),
            fixed_other,
            temperature,
        )
        derivative = (
            plus.short_circuit_current_A**2 - minus.short_circuit_current_A**2
        ) / (2.0 * h_um * 1e-6)
        fd_rows.append({
            "h_um": h_um,
            "central_FD_dI2_dc_A2_m": derivative,
            "plus_signature": node_signature(plus.electrode_0_nodes),
            "minus_signature": node_signature(minus.electrode_0_nodes),
            "same_contact_set": bool(np.array_equal(
                plus.electrode_0_nodes, minus.electrode_0_nodes
            )),
        })

    summary = {
        "baseline_project": str(BASELINE),
        "beam_index": beam_index,
        "beam_center_m": centers[beam_index].tolist(),
        "mesh_shape": list(model.mesh.shape),
        "node_count": int(model.mesh.nodes_m.shape[0]),
        "triangle_count": int(model.mesh.triangles.shape[0]),
        "matrix_nnz": int(model.matrix.nnz),
        "full_matrix_relative_asymmetry": relative_sparse_asymmetry(model.matrix),
        "reduced_matrix_relative_asymmetry": relative_sparse_asymmetry(reduced),
        "hard_current_A": forward.short_circuit_current_A,
        "q_transpose_psi_current_A": current_q,
        "current_reconstruction_relative_error": abs(
            current_q - forward.short_circuit_current_A
        ) / max(abs(forward.short_circuit_current_A), np.finfo(float).tiny),
        "swap_current_A": swapped.short_circuit_current_A,
        "swap_sign_relative_error": abs(
            swapped.short_circuit_current_A + forward.short_circuit_current_A
        ) / max(abs(forward.short_circuit_current_A), np.finfo(float).tiny),
        "psi_swap_max_abs_error": float(np.max(np.abs(
            swapped.weighting_potential - (1.0 - forward.weighting_potential)
        ))),
        "terminal_conductance_S": forward.terminal_conductance_S,
        "double_thickness_current_ratio": (
            doubled.short_circuit_current_A / forward.short_circuit_current_A
        ),
        "double_thickness_conductance_ratio": (
            doubled.terminal_conductance_S / forward.terminal_conductance_S
        ),
        "center_sweep_samples": len(center_rows),
        "center_sweep_unique_contact_sets": len({r["node_signature"] for r in center_rows}),
        "length_sweep_samples": len(length_rows),
        "length_sweep_unique_contact_sets": len({r["node_signature"] for r in length_rows}),
        "finite_difference_steps": fd_rows,
    }
    (HERE / "audit_current.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(HERE / "snapping_center.csv", center_rows)
    write_csv(HERE / "snapping_length.csv", length_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(
        [row["center_um"] for row in center_rows],
        np.asarray([row["current_A"] for row in center_rows]) * 1e9,
    )
    axes[0].set_xlabel("nominal top-contact center (um)")
    axes[0].set_ylabel("Isc (nA)")
    axes[0].set_title("Hard contact: center snapping")
    axes[1].plot(
        [row["length_um"] for row in length_rows],
        np.asarray([row["current_A"] for row in length_rows]) * 1e9,
    )
    axes[1].set_xlabel("nominal top-contact length (um)")
    axes[1].set_ylabel("Isc (nA)")
    axes[1].set_title("Hard contact: length snapping")
    fig.savefig(HERE / "snapping_sweeps.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
