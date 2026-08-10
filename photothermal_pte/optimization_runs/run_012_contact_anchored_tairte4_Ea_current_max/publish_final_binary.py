#!/usr/bin/env python3
"""Publish the exact-binary Run012 certificate without copying raw solver files."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import build_state


HERE = Path(__file__).resolve().parent
RUN_LABEL = "Run012 E∥a"
POLARIZATION_LABEL = "E∥a"
FINALIZATION = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_Ea_exact_binary_finalization_20260810/binary_finalization_result.json"
)
OBJECTIVE = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run012_Ea_exact_binary_objective_20260810/binary_objective_result.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    finalization = json.loads(FINALIZATION.read_text())
    objective = json.loads(OBJECTIVE.read_text())
    if not (finalization.get("passed") and objective.get("passed")):
        raise RuntimeError("refusing to publish a failed exact-binary gate")
    candidate = Path(finalization["candidate"]["path"])
    raw = Path(objective["raw_artifact"]["path"])
    with np.load(candidate) as data:
        rho = np.asarray(data["rho_binary"], dtype=np.uint8)
        exemptions = np.asarray(data["port_boundary_exemptions"], dtype=bool)
    with np.load(raw) as data:
        mapped_q = np.asarray(data["mapped_Q_W_m3"], dtype=float)
        temperature = np.asarray(data["nodal_temperature_K"], dtype=float)
        weighting = np.asarray(data["weighting_potential"], dtype=float)
    state = build_state(rho.astype(float))
    dz = np.diff(state.edges_m[2])
    q_areal = np.sum(mapped_q * dz[None, None, :], axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    design_extent = [-12, 12, -10, 10]
    full_extent = [-12, 12, -12, 12]
    thermal_extent = [-32, 32, -32, 32]
    axes[0, 0].imshow(
        rho.T, origin="lower", extent=design_extent, cmap="gray_r", vmin=0, vmax=1,
        interpolation="nearest", aspect="equal",
    )
    exempt_xy = np.argwhere(exemptions)
    if exempt_xy.size:
        x = -12.0 + exempt_xy[:, 0] * 0.1
        y = -10.0 + exempt_xy[:, 1] * 0.1
        axes[0, 0].scatter(x, y, marker="x", color="red", s=30, label="port-boundary exemption")
        axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[0, 0].set_title(f"Exact-binary {POLARIZATION_LABEL} design (black = TaIrTe₄)")
    axes[0, 0].set_xlabel("Lumerical x = b (µm)")
    axes[0, 0].set_ylabel("Lumerical y = a (µm)")

    image = axes[0, 1].imshow(
        q_areal.T, origin="lower", extent=thermal_extent, cmap="inferno", aspect="equal"
    )
    axes[0, 1].set_title("Conservatively mapped depth-integrated Q")
    axes[0, 1].set_xlabel("x=b (µm)")
    axes[0, 1].set_ylabel("y=a (µm)")
    fig.colorbar(image, ax=axes[0, 1], label="W m⁻²")

    image = axes[0, 2].imshow(
        temperature.T, origin="lower", extent=full_extent, cmap="magma", aspect="equal"
    )
    axes[0, 2].set_title("TaIrTe₄ nodal ΔT")
    axes[0, 2].set_xlabel("x=b (µm)")
    axes[0, 2].set_ylabel("y=a (µm)")
    fig.colorbar(image, ax=axes[0, 2], label="K")

    image = axes[1, 0].imshow(
        weighting.T, origin="lower", extent=full_extent, cmap="viridis", vmin=0, vmax=1,
        aspect="equal",
    )
    axes[1, 0].set_title("Electrical weighting potential (bottom 0, top 1)")
    axes[1, 0].set_xlabel("x=b (µm)")
    axes[1, 0].set_ylabel("y=a (µm)")
    fig.colorbar(image, ax=axes[1, 0])

    continuous = float(objective["reference_continuous_objective_A"])
    binary = float(objective["objective_A"])
    axes[1, 1].bar(["β=64 continuous", "exact binary"], np.asarray([continuous, binary]) * 1e17)
    axes[1, 1].set_ylabel("signed objective (10⁻¹⁷ A at simulated power)")
    axes[1, 1].set_title(f"Binary objective change: {(binary / continuous - 1) * 100:.3f}%")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(
            [
                "FINAL EXACT-BINARY GATES",
                f"interior discrete-opening bad nodes: {finalization['interior_bad_cell_count']}",
                "feature request: 500 nm; realized nodal opening: ~600 nm",
                f"enumerated port-boundary exemptions: {finalization['port_boundary_exemption_count']}",
                f"objective @ 285 µW: {objective['equivalent_objective_at_285uW_A'] * 1e9:.3f} nA",
                f"optical closure: {objective['gates']['optical_closure'] * 100:.5f}%",
                f"Q mapping error: {objective['gates']['Q_mapping_error']:.3e}",
                f"thermal residual: {objective['gates']['thermal_forward_residual']:.3e}",
                f"energy balance: {objective['gates']['thermal_energy_balance']:.3e}",
                "No clipping, smoothing, gain, or rescaling",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=11,
    )
    figure_path = HERE / "final_exact_binary_certificate.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    published = {
        "schema": "contact-anchored-final-exact-binary-published-v1",
        "status": f"VALIDATED_{RUN_LABEL.replace(' ', '_').upper()}_EXACT_BINARY_WITH_ENUMERATED_PORT_BOUNDARY_EXEMPTIONS",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "finalization": finalization,
        "objective": objective,
        "numerical_convergence_and_physical_gate": "passed",
        "global_morphology_claim": "not passed: 9 explicitly enumerated outermost port-boundary cells",
        "interior_feature_claim": (
            "passed: zero interior bad nodes under the conservative ~600 nm "
            "discrete opening; this is not an exact 500 nm certificate"
        ),
    }
    status_path = HERE / "FINAL_BINARY_STATUS.json"
    status_path.write_text(json.dumps(published, indent=2) + "\n")

    report_path = HERE / "FINAL_BINARY_REPORT.md"
    report_path.write_text(
        f"# {RUN_LABEL} exact-binary certificate\n\n"
        "The continuation reached β=64, but that continuous checkpoint was not called final: "
        "it retained 2.84% gray nodes and 89 global morphology violations. One deterministic, "
        "simultaneous active-set repair changed 89/48,441 nodes (0.184%) and produced an exact "
        "0/1 candidate.\n\n"
        "The requested feature size was 500 nm, but the 100 nm nodal grid rounds the "
        "250 nm opening radius up to three offsets. The realized discrete audit is therefore "
        "a conservative 300 nm maximum offset / roughly 600 nm nominal diameter, not an "
        "exact 500 nm certificate. It has **zero interior bad nodes**. The unchanged global audit "
        "reports nine violations, all explicitly enumerated at the outermost nodes where the "
        "fixed top/bottom TaIrTe4 contact phase terminates against exterior left/right void. "
        "They are treated as port-boundary exemptions, not silently counted as a global pass.\n\n"
        f"A fresh GPU Maxwell plus CUDA thermal/electrical solve gives `{objective['objective_A']:.12e} A`, "
        f"or `{objective['equivalent_objective_at_285uW_A'] * 1e9:.6f} nA` at 285 µW. "
        f"This is `{objective['relative_objective_change_from_continuous'] * 100:.4f}%` relative "
        "to the β=64 continuous checkpoint. Optical closure, conservative Q mapping, thermal "
        "residual, energy balance, and electrical weighting residual all pass.\n\n"
        "No Q clipping, smoothing, gain, global rescaling, CPU FDTD fallback, or empirical "
        "objective/gradient rescaling was used. Raw NPZ/FSP files remain outside Git.\n"
    )
    manifest = {
        "schema": "contact-anchored-final-binary-raw-manifest-v1",
        "artifacts": {
            "finalization_result": record(FINALIZATION),
            "binary_candidate": record(candidate),
            "objective_result": record(OBJECTIVE),
            "objective_fields": record(raw),
            "final_forward_FSP": objective["forward"]["project"],
        },
        "published_figure": record(figure_path),
    }
    (HERE / "FINAL_BINARY_RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
