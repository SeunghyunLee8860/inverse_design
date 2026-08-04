#!/usr/bin/env python3
"""Re-evaluate saved Device-A temperatures with three current quadratures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.paper_ir_sanity.coordinate_plot import cell_field
from photothermal_pte.validation.paper_ir_sanity.run_device_a_explicit_thermal_pte import (
    Geometry,
    pte_current_internal_face_bilinear,
    pte_current_strict_centered,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-a", type=Path, required=True)
    parser.add_argument("--case-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def geometry_from_raw(raw: np.lib.npyio.NpzFile) -> Geometry:
    flake = np.asarray(raw["flake_mask"], bool)
    shape = flake.shape
    return Geometry(
        x_edges_m=np.asarray(raw["x_edges_m"], float),
        y_edges_m=np.asarray(raw["y_edges_m"], float),
        z_edges_m=np.asarray(raw["z_edges_m"], float),
        material_id=np.where(flake, 3, 0).astype(np.uint8),
        flake_mask=flake,
        kappa_W_mK=np.ones((*shape, 3), float),
        interface_resistance_m2K_W={
            "x": np.zeros((shape[0] - 1, shape[1], shape[2])),
            "y": np.zeros((shape[0], shape[1] - 1, shape[2])),
            "z": np.zeros((shape[0], shape[1], shape[2] - 1)),
        },
    )


def signed_parts(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, float)
    positive = float(np.sum(array[array > 0.0]))
    negative = float(np.sum(array[array < 0.0]))
    absolute = positive - negative
    net = positive + negative
    return {
        "positive_A": positive,
        "negative_A": negative,
        "net_A": net,
        "absolute_sum_A": absolute,
        "cancellation_factor_abs_net_over_abs_sum": abs(net) / max(
            absolute, np.finfo(float).tiny
        ),
    }


def face_to_cell_map(
    face_x: np.ndarray,
    face_y: np.ndarray,
    shape_xy: tuple[int, int],
) -> np.ndarray:
    cell = np.zeros(shape_xy, float)
    xsum = np.sum(face_x, axis=2)
    ysum = np.sum(face_y, axis=2)
    cell[:-1, :] += 0.5 * xsum
    cell[1:, :] += 0.5 * xsum
    cell[:, :-1] += 0.5 * ysum
    cell[:, 1:] += 0.5 * ysum
    return cell


def evaluate(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fields_path = directory / "thermal_pte_fields.npz"
    summary_path = directory / "summary.json"
    stored_summary = json.loads(summary_path.read_text())
    with np.load(fields_path) as raw:
        geometry = geometry_from_raw(raw)
        temperature = np.asarray(raw["temperature_rise_K"], float)
        psi = np.asarray(raw["weighting_potential"], float)
        legacy_cell = np.asarray(raw["shockley_ramo_integrand_A_m3_3d"], float)
        dx = np.diff(geometry.x_edges_m)
        dy = np.diff(geometry.y_edges_m)
        dz = np.diff(geometry.z_edges_m)
        volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
        legacy_contribution = legacy_cell * volume
        legacy_current = float(np.sum(legacy_contribution))
        strict_current, strict = pte_current_strict_centered(
            temperature, geometry, psi
        )
        face_current, face = pte_current_internal_face_bilinear(
            temperature, geometry, psi
        )
        strict_contribution = np.asarray(strict["cell_contribution_A"], float)
        face_x = np.asarray(face["face_x_contribution_A"], float)
        face_y = np.asarray(face["face_y_contribution_A"], float)
        mask_xy = np.any(geometry.flake_mask, axis=2)
        maps = {
            "x_edges_m": geometry.x_edges_m,
            "y_edges_m": geometry.y_edges_m,
            "mask_xy": mask_xy,
            "legacy_A_per_cell": np.sum(legacy_contribution, axis=2),
            "strict_A_per_cell": np.sum(strict_contribution, axis=2),
            "face_A_per_cell": face_to_cell_map(face_x, face_y, mask_xy.shape),
        }
    reported = float(stored_summary["PTE_current_A_at_requested_incident_power"])
    return {
        "directory": str(directory.resolve()),
        "fields": {
            "path": str(fields_path.resolve()),
            "size_bytes": fields_path.stat().st_size,
            "sha256": sha256(fields_path),
        },
        "legacy": {
            "reported_A": reported,
            "offline_reintegrated_A": legacy_current,
            "reintegration_relative_error": abs(legacy_current - reported)
            / max(abs(reported), np.finfo(float).tiny),
            "signed_parts": signed_parts(legacy_contribution),
            "contract": "stored one-sided/centered cell-gradient volume quadrature",
        },
        "strict_centered": {
            "current_A": strict_current,
            "relative_change_vs_legacy": abs(strict_current - legacy_current)
            / max(abs(legacy_current), np.finfo(float).tiny),
            "signed_parts": signed_parts(strict_contribution),
            "valid_xy_cell_count": strict["valid_xy_cell_count"],
            "masked_flake_xy_cell_count": strict[
                "masked_flake_xy_cell_count"
            ],
            "contract": strict["contract"],
            "production_status": strict["production_status"],
        },
        "internal_face": {
            "current_A": face_current,
            "current_x_A": face["current_x_faces_A"],
            "current_y_A": face["current_y_faces_A"],
            "relative_change_vs_legacy": abs(face_current - legacy_current)
            / max(abs(legacy_current), np.finfo(float).tiny),
            "signed_parts": signed_parts(np.concatenate((face_x.ravel(), face_y.ravel()))),
            "connected_x_face_count": face["connected_x_face_count"],
            "connected_y_face_count": face["connected_y_face_count"],
            "contract": face["contract"],
            "exterior_boundary_half_control_volumes_included": face[
                "exterior_boundary_half_control_volumes_included"
            ],
            "production_status": face["production_status"],
        },
    }, maps


def ratio(cases: dict[str, dict[str, object]], method: str) -> float:
    key = "reported_A" if method == "legacy" else "current_A"
    return abs(float(cases["a"][method][key])) / abs(
        float(cases["b"][method][key])
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict[str, object]] = {}
    maps: dict[str, dict[str, np.ndarray]] = {}
    for polarization, directory in (("a", args.case_a), ("b", args.case_b)):
        cases[polarization], maps[polarization] = evaluate(directory)

    ratios = {method: ratio(cases, method) for method in (
        "legacy", "strict_centered", "internal_face"
    )}
    payload = {
        "status": "DIAGNOSTIC_CURRENT_QUADRATURE_DISAGREEMENT_PRESERVED",
        "scope": (
            "offline reintegration of immutable temperature/weighting fields; "
            "no new FDTD or thermal solve"
        ),
        "axis_mapping": "x=b, y=a",
        "cases": cases,
        "abs_Ia_over_abs_Ib": ratios,
        "paper_digitized_abs_Ia_over_abs_Ib": 0.8365896980461811,
        "interpretation": {
            "legacy_is_not_promoted": True,
            "strict_is_not_promoted": (
                "it obeys the four-neighbour mask but deletes boundary volume"
            ),
            "internal_face_is_not_promoted": (
                "it collocates both differences on common faces but an explicit "
                "contact boundary-face quadrature is still missing"
            ),
            "root_cause_test": (
                "if every quadrature retains abs(Ia)/abs(Ib)>1, the old ratio "
                "cannot be explained solely by the one-sided cell stencil"
            ),
        },
    }
    (args.output_dir / "device_a_current_discretization_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    with (args.output_dir / "device_a_current_discretization_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "polarization", "method", "current_A", "current_nA",
            "abs_Ia_over_abs_Ib",
        ])
        for polarization in ("a", "b"):
            for method in ("legacy", "strict_centered", "internal_face"):
                key = "reported_A" if method == "legacy" else "current_A"
                current = float(cases[polarization][method][key])
                writer.writerow([
                    polarization, method, current, 1e9 * current, ratios[method]
                ])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    methods = ("legacy_A_per_cell", "strict_A_per_cell", "face_A_per_cell")
    titles = ("legacy cell gradient", "strict four-neighbour", "internal-face bilinear")
    maximum = max(
        float(np.nanmax(np.abs(maps[p][m])))
        for p in ("a", "b") for m in methods
    )
    for row, polarization in enumerate(("a", "b")):
        for column, (method, title) in enumerate(zip(methods, titles)):
            values = np.array(maps[polarization][method], copy=True)
            values[~maps[polarization]["mask_xy"]] = np.nan
            image = cell_field(
                axes[row, column],
                maps[polarization]["x_edges_m"] * 1e6,
                maps[polarization]["y_edges_m"] * 1e6,
                values * 1e12,
                cmap="RdBu_r",
                vmin=-maximum * 1e12,
                vmax=maximum * 1e12,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("x=b (um)")
            axes[row, column].set_ylabel("y=a (um)")
            axes[row, column].set_title(f"E||{polarization}: {title}")
            fig.colorbar(image, ax=axes[row, column], label="cell-assigned current (pA)")
    fig.suptitle("Immutable Device-A fields: current-discretization audit")
    fig.savefig(args.output_dir / "DEVICE_A_CURRENT_DISCRETIZATION_MAPS.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    xloc = np.arange(3)
    width = 0.34
    for offset, polarization in ((-0.5, "a"), (0.5, "b")):
        currents = []
        for method in ("legacy", "strict_centered", "internal_face"):
            key = "reported_A" if method == "legacy" else "current_A"
            currents.append(1e9 * float(cases[polarization][method][key]))
        ax.bar(xloc + offset * width, currents, width, label=f"E||{polarization}")
    ax.set_xticks(xloc, ["legacy", "strict centered", "internal face"])
    ax.set_ylabel("signed current (nA)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    twin = ax.twinx()
    twin.plot(xloc, list(ratios.values()), "ko--", label="|Ia|/|Ib|")
    twin.axhline(0.8365896980461811, color="tab:red", linestyle=":", label="paper")
    twin.set_ylabel("|Ia|/|Ib|")
    twin.legend(loc="lower right")
    fig.savefig(args.output_dir / "DEVICE_A_CURRENT_DISCRETIZATION_COMPARISON.png", dpi=210)
    plt.close(fig)

    report = f"""# Device-A current-discretization audit

Status: `{payload['status']}`

No new Maxwell or thermal solve was run. The immutable saved temperature and
weighting-potential arrays were reintegrated with three separately named
quadratures.

| method | abs(Ia)/abs(Ib) |
|---|---:|
| legacy mixed centred/one-sided cell gradient | {ratios['legacy']:.9f} |
| strict four-neighbour cell mask | {ratios['strict_centered']:.9f} |
| common internal-face bilinear | {ratios['internal_face']:.9f} |
| digitized paper target | 0.836589698 |

The strict scheme satisfies the requested masking rule but removes physical
boundary volume. The internal-face scheme collocates T and psi differences on
the same faces, but it omits exterior half-control-volume/contact quadrature.
Neither diagnostic is silently promoted as production. If all three retain a
ratio above one, the old polarization reversal is not caused only by the
legacy one-sided gradient implementation.
"""
    (args.output_dir / "DEVICE_A_CURRENT_DISCRETIZATION_AUDIT.md").write_text(report)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
