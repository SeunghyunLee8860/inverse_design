#!/usr/bin/env python3
"""Audit whether the finite-187T optical Q can enter a terminal-PTE solve.

This is deliberately an offline, read-only handoff audit.  It never crops or
renormalizes Q and it does not invent a finite flake or electrode geometry.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_Q_11p825um_Eb"
)
NPZ = RAW / "finite_187T_w12_Q.npz"
EXPECTED_NPZ_SHA256 = "c4dbbdce0038a5b3a0c8529dc867a266245ef6d37672e20245790d8429b08f09"
OUTPUT = HERE / "results_finite_187T_thermal_electrical_handoff"

CANDIDATES_UM = (
    ("finite_T_array_footprint", 16.5, 17.0),
    ("legacy_top_bottom_24x20", 24.0, 20.0),
    ("legacy_left_right_20x24", 20.0, 24.0),
    ("diagnostic_30x30", 30.0, 30.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_widths(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, float)
    if coordinates.ndim != 1 or coordinates.size < 2 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("coordinates must be a strictly increasing 1-D array")
    edges = np.empty(coordinates.size + 1, float)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
    edges[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    return np.diff(edges)


def component_containment(data: np.lib.npyio.NpzFile, component: str) -> tuple[float, dict[str, float]]:
    q = np.asarray(data[f"Q{component}_W_m3"], float)
    x = np.asarray(data[f"Q{component}_x_m"], float)
    y = np.asarray(data[f"Q{component}_y_m"], float)
    z = np.asarray(data[f"Q{component}_z_m"], float)
    if q.shape != (x.size, y.size, z.size):
        raise ValueError(f"Q{component} shape/coordinate mismatch")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError(f"Q{component} is not finite and nonnegative")
    wx, wy, wz = dual_widths(x), dual_widths(y), dual_widths(z)
    qxy = np.einsum("ijk,k->ij", q, wz, optimize=True)
    total = float(np.einsum("ij,i,j->", qxy, wx, wy, optimize=True))
    contained: dict[str, float] = {}
    for name, span_x_um, span_y_um in CANDIDATES_UM:
        mx = np.abs(x) <= 0.5 * span_x_um * 1.0e-6
        my = np.abs(y) <= 0.5 * span_y_um * 1.0e-6
        contained[name] = float(
            np.einsum("ij,i,j->", qxy[np.ix_(mx, my)], wx[mx], wy[my], optimize=True)
        )
    return total, contained


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    actual_sha = sha256(NPZ)
    if actual_sha != EXPECTED_NPZ_SHA256:
        raise RuntimeError(f"finite-Q NPZ SHA mismatch: {actual_sha}")

    totals: dict[str, float] = {}
    by_component: dict[str, dict[str, float]] = {}
    with np.load(NPZ, allow_pickle=False) as data:
        for component in "xyz":
            totals[component], by_component[component] = component_containment(data, component)
        x_common = np.asarray(data["common_x_m"], float)
        y_common = np.asarray(data["common_y_m"], float)
        z_common = np.asarray(data["common_z_m"], float)
        q_common = np.asarray(data["Q_common_W_m3"], float)
        qxy_common = np.trapezoid(q_common, z_common, axis=2)

    total_power = float(sum(totals.values()))
    rows = []
    for name, span_x_um, span_y_um in CANDIDATES_UM:
        powers = {component: by_component[component][name] for component in "xyz"}
        contained = float(sum(powers.values()))
        rows.append(
            {
                "scenario": name,
                "span_x_um": span_x_um,
                "span_y_um": span_y_um,
                "P_Qx_inside_W": powers["x"],
                "P_Qy_inside_W": powers["y"],
                "P_Qz_inside_W": powers["z"],
                "P_Q_inside_W": contained,
                "inside_fraction": contained / total_power,
                "outside_fraction": 1.0 - contained / total_power,
                "permitted_as_thermal_crop": False,
            }
        )

    with (OUTPUT / "finite_thermal_electrical_handoff_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    positive = qxy_common[qxy_common > 0.0]
    image = axes[0].pcolormesh(
        x_common * 1e6,
        y_common * 1e6,
        qxy_common.T,
        shading="auto",
        cmap="inferno",
        norm=LogNorm(vmin=max(float(np.max(positive)) * 1e-6, float(np.percentile(positive, 1))), vmax=float(np.max(positive))),
    )
    colors = ("#00ffff", "#00ff55", "#4da6ff", "white")
    for (name, sx, sy), color in zip(CANDIDATES_UM, colors):
        axes[0].add_patch(Rectangle((-sx / 2, -sy / 2), sx, sy, fill=False, ec=color, lw=1.5, label=name))
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Lumerical x=b (um)")
    axes[0].set_ylabel("Lumerical y=a (um)")
    axes[0].set_title("Existing all-material Q; rectangles are audit-only")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.colorbar(image, ax=axes[0], label="depth-integrated Q (W/m2)")

    labels = [row["scenario"].replace("legacy_", "").replace("diagnostic_", "") for row in rows]
    outside_pct = [100.0 * float(row["outside_fraction"]) for row in rows]
    axes[1].barh(labels, outside_pct, color="#b2182b")
    axes[1].set_xlabel("existing optical power outside rectangle (%)")
    axes[1].set_title("Cropping would delete this power")
    axes[1].grid(axis="x", alpha=0.3)
    for index, value in enumerate(outside_pct):
        axes[1].text(value + 0.15, index, f"{value:.3f}%", va="center")

    axes[2].axis("off")
    axes[2].text(0.03, 0.92, "Terminal-PTE geometry still required", fontsize=16, weight="bold")
    required = [
        "finite TaIrTe4 flake footprint",
        "two electrode footprints and terminal polarity",
        "physical SiO2 thickness + Si depth",
        "TaIrTe4/Au and layer-interface thermal G",
        "electrical contact model",
        "finite-flake Maxwell rerun with the same Q gates",
    ]
    for index, item in enumerate(required):
        axes[2].text(0.06, 0.80 - 0.105 * index, f"{index + 1}. {item}", fontsize=12)
    axes[2].text(
        0.03,
        0.08,
        "No source deletion, crop, smoothing, gain, or rescaling.\n"
        "The current Q remains a valid finite-array optical certificate,\n"
        "but it is not yet a finite-flake terminal-PTE source.",
        fontsize=11,
        color="#8b1a1a",
    )
    fig.suptitle("Finite-187T optical Q -> thermal/electrical handoff audit", fontsize=17)
    plot_path = OUTPUT / "finite_thermal_electrical_handoff_audit.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    summary = {
        "status": "BLOCKED_FINITE_PTE_GEOMETRY_UNDEFINED",
        "validated_input": {
            "classification": "finite 11x17 Au inverse-T array on laterally extended TaIrTe4/stack",
            "npz_path": str(NPZ),
            "npz_size_bytes": NPZ.stat().st_size,
            "npz_sha256": actual_sha,
            "P_Q_component_reintegration_W": totals,
            "P_Q_total_reintegrated_W": total_power,
        },
        "diagnostic_rectangles_are_not_promoted_geometries": rows,
        "reason": (
            "The optical TaIrTe4 and lower stack extend through lateral PML, while terminal PTE requires "
            "a finite conducting flake and two specified contacts. Cropping this Q would delete source power "
            "and omit finite-flake/contact Maxwell scattering."
        ),
        "next_required_contract": {
            "finite_flake_footprint": "undefined",
            "electrode_footprints_and_polarity": "undefined",
            "physical_thermal_stack": "must not inherit the 285-nm optical closure without an explicit choice",
            "required_action": "rerun finite-device Maxwell after geometry is fixed, then conservative material-overlap thermal remap",
        },
        "not_run": ["new FDTD", "thermal", "weighting potential", "PTE", "adjoint", "optimization"],
    }
    summary_path = OUTPUT / "finite_thermal_electrical_handoff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = OUTPUT / "FINITE_THERMAL_ELECTRICAL_HANDOFF_REPORT.md"
    report_path.write_text(
        "# Finite-187T optical-to-PTE handoff audit\n\n"
        "Status: `BLOCKED_FINITE_PTE_GEOMETRY_UNDEFINED`\n\n"
        "The validated optical model has a finite 11 x 17 Au inverse-T array, but its "
        "TaIrTe4 layer and lower stack extend laterally through the PML. A terminal-current "
        "calculation instead needs a finite conducting flake and two explicit contacts. "
        "Those dimensions are not defined by the current paper-architecture contract.\n\n"
        "The existing Q was reintegrated without modification. Audit rectangles show how much "
        "of that source would be deleted by an after-the-fact crop:\n\n"
        "| Audit rectangle | Q inside | Q outside |\n|---|---:|---:|\n"
        + "".join(
            f"| {row['scenario']} ({row['span_x_um']:.1f} x {row['span_y_um']:.1f} um) "
            f"| {100*row['inside_fraction']:.3f}% | {100*row['outside_fraction']:.3f}% |\n"
            for row in rows
        )
        + "\nThese are diagnostics, not promoted device geometries. In particular, cropping to "
        "24 x 20 um would delete about 5.37% of the existing optical power and would also "
        "miss finite-flake/contact scattering. No crop, deletion, gain, smoothing, or rescaling "
        "was performed.\n\n"
        "The next physically valid step is to freeze the finite TaIrTe4 footprint, electrode "
        "footprints/polarity, and physical thermal stack; rerun Maxwell with those objects; then "
        "map the unmodified volumetric Q conservatively into the explicit thermal/electrical solve.\n"
    )

    artifacts = []
    for path in (report_path, summary_path, OUTPUT / "finite_thermal_electrical_handoff_cases.csv", plot_path):
        artifacts.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "raw_artifacts_committed_to_git": False,
                "input_raw_artifact": {"path": str(NPZ), "size_bytes": NPZ.stat().st_size, "sha256": actual_sha},
                "published_artifacts": artifacts,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
