#!/usr/bin/env python3
"""Audit the saved analytic/planar/finite-edge Q comparison inputs.

This is deliberately offline.  It does not start Lumerical.  The script
fails closed when an actual planar TaIrTe4-stack Q artifact is unavailable;
an empty-stack control or a finite Device-A polygon is not relabeled as a
planar-stack result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


DEFAULT_ARTIFACT_ROOT = Path(
    "/home/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity"
)
W0_M = 6.5e-6
PAPER_ABSORPTION_A = 0.17673296


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_edges(coordinate: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(coordinate, float)
    middle = 0.5 * (coordinate[:-1] + coordinate[1:])
    return np.concatenate(
        (
            [coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])],
            middle,
            [coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])],
        )
    )


def load_q(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        q = np.asarray(raw["Q_on_W_m3"], float)
        components = {
            name: np.asarray(raw[f"Q{name}_W_m3"], float)
            for name in ("x", "y", "z")
        }
        metadata = json.loads(str(np.asarray(raw["metadata_json"]).ravel()[0]))
    x_edges, y_edges, z_edges = map(dual_edges, (x, y, z))
    volume = (
        np.diff(x_edges)[:, None, None]
        * np.diff(y_edges)[None, :, None]
        * np.diff(z_edges)[None, None, :]
    )
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    areal = np.sum(q * np.diff(z_edges)[None, None, :], axis=2)
    power = float(np.sum(q * volume))
    return {
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "Q_W_m3": q,
        "areal_Q_W_m2": areal,
        "normalized_areal_Q_m2_inv": areal / power,
        "power_W": power,
        "component_power_W": {
            name: float(np.sum(values * volume))
            for name, values in components.items()
        },
        "area_m2": area,
        "metadata": metadata,
    }


def analytic_areal_q(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Exact Gaussian cell average times the y<=x half-plane centre mask."""
    x_edges, y_edges = map(dual_edges, (x_m, y_m))
    scale = np.sqrt(2.0) / W0_M
    int_x = 0.5 * (
        erf(scale * x_edges[1:]) - erf(scale * x_edges[:-1])
    )
    int_y = 0.5 * (
        erf(scale * y_edges[1:]) - erf(scale * y_edges[:-1])
    )
    cell_fraction = np.outer(int_x, int_y)
    xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
    support = yy < xx
    diagonal = np.isclose(xx, yy, atol=1e-15, rtol=0.0)
    occupancy = support.astype(float) + 0.5 * diagonal
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    areal = PAPER_ABSORPTION_A * cell_fraction * occupancy / area
    return areal


def normal_profile(
    x_m: np.ndarray,
    y_m: np.ndarray,
    normalized_areal_q: np.ndarray,
    *,
    tangent_half_width_m: float = 0.10e-6,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
    normal = (-xx + yy) / np.sqrt(2.0)
    tangent = (xx + yy) / np.sqrt(2.0)
    selected = np.abs(tangent) <= tangent_half_width_m
    n = normal[selected]
    values = normalized_areal_q[selected]
    order = np.argsort(n)
    return n[order], values[order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    candidates = {
        "empty_stack_control": (
            args.artifact_root
            / "empty_a_w6p5_dz10_shut1e5_gpu2_20260730"
        ),
        "finite_device_a_polygon": (
            args.artifact_root
            / "finite_center_a_w6p5_dz10_gpu2_20260730"
        ),
        "finite_straight_45_edge_legacy_c": (
            args.artifact_root
            / "straight45_a_w6p5_dz10_L48_gpu4_20260730"
        ),
    }
    rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    for name, directory in candidates.items():
        result_path = directory / "case_result.json"
        q_path = directory / "finite_q_on_artifact.npz"
        result = json.loads(result_path.read_text())
        row: dict[str, Any] = {
            "name": name,
            "case": result.get("case"),
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256(result_path),
            "q_artifact_available": q_path.exists(),
            "q_artifact_path": str(q_path.resolve()) if q_path.exists() else "",
            "q_artifact_sha256": sha256(q_path) if q_path.exists() else "",
            "usable_as_planar_TaIrTe4_stack": False,
        }
        if q_path.exists():
            item = load_q(q_path)
            loaded[name] = item
            row.update(
                {
                    "P_Q_W": item["power_W"],
                    "P_Qx_W": item["component_power_W"]["x"],
                    "P_Qy_W": item["component_power_W"]["y"],
                    "P_Qz_W": item["component_power_W"]["z"],
                    "Qz_fraction": item["component_power_W"]["z"]
                    / max(abs(item["power_W"]), np.finfo(float).tiny),
                }
            )
        if name == "empty_stack_control":
            row["rejection_reason"] = (
                "no TaIrTe4 and no volume-Q artifact; incident/background "
                "control is not a planar TaIrTe4 stack"
            )
        elif name == "finite_device_a_polygon":
            row["rejection_reason"] = (
                "finite digitized Device-A polygon; it contains edges and is "
                "not an edge-free planar stack"
            )
        else:
            row["rejection_reason"] = (
                "valid finite straight-edge diagnostic, but generated with "
                "legacy lossless epsilon_c=16 closure (P_Qz is exactly zero)"
            )
        rows.append(row)

    planar_available = any(
        bool(row["usable_as_planar_TaIrTe4_stack"]) for row in rows
    )
    edge = loaded["finite_straight_45_edge_legacy_c"]
    analytic = analytic_areal_q(edge["x_m"], edge["y_m"])
    analytic_power = float(np.sum(analytic * edge["area_m2"]))
    analytic_normalized = analytic / analytic_power
    edge_n, edge_profile = normal_profile(
        edge["x_m"], edge["y_m"], edge["normalized_areal_Q_m2_inv"]
    )
    analytic_n, analytic_profile = normal_profile(
        edge["x_m"], edge["y_m"], analytic_normalized
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, data, title in (
        (
            axes[0],
            analytic_normalized,
            "Analytic Gaussian–Beer–Lambert\n(equal-power shape basis)",
        ),
        (
            axes[1],
            edge["normalized_areal_Q_m2_inv"],
            "Saved finite 45° edge FDTD\n(legacy $\\epsilon_c=16$)",
        ),
    ):
        image = axis.imshow(
            data.T,
            origin="lower",
            extent=[
                edge["x_m"][0] * 1e6,
                edge["x_m"][-1] * 1e6,
                edge["y_m"][0] * 1e6,
                edge["y_m"][-1] * 1e6,
            ],
            cmap="inferno",
            aspect="equal",
        )
        axis.set_xlim(-12, 12)
        axis.set_ylim(-12, 12)
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("y (µm)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=r"$Q_A/\int Q\,dV$ (m$^{-2}$)")
    axes[2].plot(
        analytic_n * 1e6,
        analytic_profile / max(np.max(analytic_profile), 1e-300),
        label="analytic",
    )
    axes[2].plot(
        edge_n * 1e6,
        edge_profile / max(np.max(edge_profile), 1e-300),
        label="saved finite edge",
    )
    axes[2].set_xlim(-10, 10)
    axes[2].set_xlabel(r"edge-normal $n=(-x+y)/\sqrt{2}$ (µm)")
    axes[2].set_ylabel("profile / own maximum")
    axes[2].set_title("Provenance-only profile comparison")
    axes[2].legend()
    figure.suptitle(
        "Offline saved-Q audit — required planar-stack source is unavailable"
    )
    figure.savefig(args.output_dir / "saved_three_source_q_audit.png", dpi=220)
    plt.close(figure)

    status = (
        "AVAILABLE_THREE_SOURCE_Q_COMPARISON"
        if planar_available
        else "BLOCKED_PLANAR_STACK_Q_ARTIFACT_UNAVAILABLE"
    )
    summary = {
        "status": status,
        "validated": False,
        "new_FDTD_run": False,
        "required_comparison": [
            "paper analytic Gaussian-Beer-Lambert Q",
            "edge-free planar TaIrTe4-stack Lumerical Gaussian Q",
            "finite 45-degree-edge Lumerical Gaussian Q",
        ],
        "planar_stack_artifact_available": planar_available,
        "candidate_audit": rows,
        "available_provenance_only_comparison": {
            "analytic_shape_total_before_normalization": analytic_power,
            "finite_edge_P_Q_W": edge["power_W"],
            "normalization": (
                "each displayed areal distribution divided by its own "
                "integrated Q; no raw artifact was modified"
            ),
            "production_claim": False,
            "reason": (
                "the finite-edge artifact uses the legacy epsilon_c closure "
                "and the required planar-stack artifact is absent"
            ),
        },
        "sequencing": {
            "Figure_3H_I_started": False,
            "reason": (
                "the approved order places the SPCM comparison after the "
                "three-source decomposition; that decomposition is blocked "
                "without a planar-stack artifact and new FDTD was prohibited"
            ),
        },
    }
    (args.output_dir / "saved_three_source_q_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.output_dir / "saved_three_source_q_audit.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0 if status.startswith("BLOCKED_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
