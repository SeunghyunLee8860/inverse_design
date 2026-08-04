#!/usr/bin/env python3
"""Build an explicit approximate Figure-3H-to-Device-A registration.

This script performs geometry bookkeeping only.  It does not open Lumerical
or run optical, thermal, electrical, PTE, adjoint, or optimization solves.
The registration is intentionally named an approximation because the paper
does not publish raw SPCM stage coordinates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np


FIG3H_LEFT_PANEL_BOUNDS_PX = np.asarray([[125.0, 251.0], [680.0, 720.0]])
FIG3H_SCALE_BAR_PX = np.asarray([[263.0, 281.0], [375.0, 281.0]])
FIG3H_BLACK_SCAN_LINE_PX = np.asarray([[217.0, 252.0], [217.0, 719.0]])
ASSUMED_SCALE_BAR_UM = 10.0
FIG3I_NOMINAL_PEAK_DISTANCE_UM = 3.0
SPARSE_SCAN_DISTANCES_UM = (-1.0, 1.0, 3.0, 5.0, 7.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig3h-crop", type=Path, required=True)
    parser.add_argument("--base-geometry-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--waist-um", type=float, default=8.75)
    return parser.parse_args()


def affine_pixel_to_device_um(
    points_px: np.ndarray,
    *,
    panel_center_px: np.ndarray,
    flake_center_um: np.ndarray,
    pixels_per_um: float,
) -> np.ndarray:
    """Map Fig. 3H pixels to code coordinates under the named assumption."""

    points_px = np.asarray(points_px, float)
    return np.column_stack(
        (
            flake_center_um[0]
            + (points_px[:, 0] - panel_center_px[0]) / pixels_per_um,
            flake_center_um[1]
            - (points_px[:, 1] - panel_center_px[1]) / pixels_per_um,
        )
    )


def nearest_polygon_point(
    point: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, int, float]:
    best_point = None
    best_index = -1
    best_distance = np.inf
    for index, start in enumerate(vertices):
        stop = vertices[(index + 1) % len(vertices)]
        direction = stop - start
        parameter = float(
            np.clip(
                np.dot(point - start, direction)
                / max(np.dot(direction, direction), np.finfo(float).tiny),
                0.0,
                1.0,
            )
        )
        candidate = start + parameter * direction
        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_point = candidate
            best_index = index
            best_distance = distance
    assert best_point is not None
    return np.asarray(best_point), best_index, best_distance


def source_device_envelope(
    *,
    beam_um: np.ndarray,
    source_span_um: float,
    domain_um: float,
    top_metal_um: np.ndarray,
    bottom_metal_um: np.ndarray,
) -> dict[str, object]:
    half_source = 0.5 * source_span_um
    metal = np.vstack((top_metal_um, bottom_metal_um))
    occupied_min = np.minimum(beam_um - half_source, np.min(metal, axis=0))
    occupied_max = np.maximum(beam_um + half_source, np.max(metal, axis=0))
    shift = -0.5 * (occupied_min + occupied_max)
    shifted_min = occupied_min + shift
    shifted_max = occupied_max + shift
    clearance = 0.5 * domain_um - np.maximum(
        np.abs(shifted_min), np.abs(shifted_max)
    )
    return {
        "domain_um": domain_um,
        "source_span_um": source_span_um,
        "occupied_union_before_shift_um": {
            "min": occupied_min.tolist(),
            "max": occupied_max.tolist(),
        },
        "simulation_origin_shift_um": shift.tolist(),
        "occupied_union_after_shift_um": {
            "min": shifted_min.tolist(),
            "max": shifted_max.tolist(),
        },
        "minimum_PML_clearance_um": {
            "x": float(clearance[0]),
            "y": float(clearance[1]),
        },
        "passes_existing_loader_clearance_gate": bool(np.all(clearance >= 0.5)),
    }


def draw_rgb(ax: plt.Axes, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    ax.pcolormesh(
        np.arange(width + 1),
        np.arange(height + 1),
        image[..., :3],
        shading="flat",
        rasterized=True,
    )
    ax.set_xlim(0.0, float(width))
    ax.set_ylim(float(height), 0.0)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_geometry_json.read_text())
    image = plt.imread(args.fig3h_crop)

    flake = np.asarray(base["flake_vertices_code_um"], float)
    top_metal = np.asarray(base["top_metal_polygon_code_um"], float)
    bottom_metal = np.asarray(base["bottom_metal_polygon_code_um"], float)
    flake_center = 0.5 * (np.min(flake, axis=0) + np.max(flake, axis=0))
    panel_center = 0.5 * (
        FIG3H_LEFT_PANEL_BOUNDS_PX[0] + FIG3H_LEFT_PANEL_BOUNDS_PX[1]
    )
    pixels_per_um = float(
        np.linalg.norm(FIG3H_SCALE_BAR_PX[1] - FIG3H_SCALE_BAR_PX[0])
        / ASSUMED_SCALE_BAR_UM
    )
    scan_endpoints_um = affine_pixel_to_device_um(
        FIG3H_BLACK_SCAN_LINE_PX,
        panel_center_px=panel_center,
        flake_center_um=flake_center,
        pixels_per_um=pixels_per_um,
    )
    scan_x_um = float(np.mean(scan_endpoints_um[:, 0]))

    # The Figure-3I distance origin is not published as a stage coordinate.
    # This named registration maps distance zero to the centre of the Fig. 3H
    # panel and to the digitized flake-bounding-box centre.  Positive distance
    # follows +a (up in the paper panels).
    nominal_beam = np.asarray(
        [scan_x_um, flake_center[1] + FIG3I_NOMINAL_PEAK_DISTANCE_UM]
    )
    scan_positions = [
        {
            "label": f"d{distance:+g}um".replace("+", "p").replace("-", "m"),
            "distance_from_assumed_Fig3I_zero_um": distance,
            "beam_center_code_um": [
                scan_x_um,
                float(flake_center[1] + distance),
            ],
        }
        for distance in SPARSE_SCAN_DISTANCES_UM
    ]

    nearest, segment, distance = nearest_polygon_point(nominal_beam, flake)
    inside = bool(MplPath(flake, closed=True).contains_point(nominal_beam))
    signed_distance = -distance if inside else distance

    envelopes = {
        "unchanged_60um_domain_50um_source": source_device_envelope(
            beam_um=nominal_beam,
            source_span_um=50.0,
            domain_um=60.0,
            top_metal_um=top_metal,
            bottom_metal_um=bottom_metal,
        ),
        "recommended_64um_domain_50um_source": source_device_envelope(
            beam_um=nominal_beam,
            source_span_um=50.0,
            domain_um=64.0,
            top_metal_um=top_metal,
            bottom_metal_um=bottom_metal,
        ),
        "faster_diagnostic_60um_domain_40um_source": source_device_envelope(
            beam_um=nominal_beam,
            source_span_um=40.0,
            domain_um=60.0,
            top_metal_um=top_metal,
            bottom_metal_um=bottom_metal,
        ),
    }
    for envelope in envelopes.values():
        half_span = 0.5 * float(envelope["source_span_um"])
        envelope["ideal_Gaussian_boundary_intensity_over_peak"] = float(
            np.exp(-2.0 * (half_span / args.waist_um) ** 2)
        )

    registered = copy.deepcopy(base)
    registered["status"] = "DEVICE_A_FIG3H_APPROXIMATE_AFFINE_REGISTRATION"
    registered["pre_registered_beam_center_code_um"] = nominal_beam.tolist()
    registered["beam_center_rule"] = (
        "EXPLICIT_APPROXIMATION: align the Figure-3H map-panel centre with "
        "the digitized Figure-2 flake-bounding-box centre, use the visible "
        "10-um scale bar without rotation/shear, and place the nominal "
        "source at Figure-3I distance +3 um along +a"
    )
    registered["fig3h_approximate_affine_registration"] = {
        "classification": "EXPLICIT_APPROXIMATION_NOT_STAGE_METROLOGY",
        "pixel_to_code_axes": {
            "image_right": "+b",
            "image_up": "+a",
            "rotation_or_shear": False,
        },
        "panel_center_px": panel_center.tolist(),
        "panel_center_aligned_to_flake_bbox_center_code_um": flake_center.tolist(),
        "pixels_per_um": pixels_per_um,
        "black_scan_line_endpoints_px": FIG3H_BLACK_SCAN_LINE_PX.tolist(),
        "black_scan_line_endpoints_code_um": scan_endpoints_um.tolist(),
        "Fig3I_distance_zero_assumption": (
            "panel centre equals flake-bounding-box centre along a"
        ),
        "positive_distance_direction": "+a",
        "nominal_peak_distance_um": FIG3I_NOMINAL_PEAK_DISTANCE_UM,
        "nominal_beam_center_code_um": nominal_beam.tolist(),
        "nearest_digitized_flake_boundary_point_um": nearest.tolist(),
        "nearest_digitized_flake_segment_start_index": segment,
        "signed_distance_to_digitized_flake_um": signed_distance,
        "translation_uncertainty_scenarios_um": [
            [0.0, 0.0],
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, -2.0],
            [0.0, 2.0],
        ],
        "sparse_scan_positions": scan_positions,
        "comparison_target": (
            "separate maxima of |Ia| and |Ib| over the registered scan, "
            "not a single arbitrarily selected point"
        ),
    }
    registered["limitations"] = list(registered.get("limitations", [])) + [
        "Figure-3H panel centre to Figure-2 flake-centre alignment is assumed",
        "Figure-3I distance zero is assumed at that common centre",
        "the registered line and source positions are not raw SPCM stage coordinates",
    ]
    registered_path = args.output_dir / "device_a_fig3h_registered_geometry.json"
    registered_path.write_text(json.dumps(registered, indent=2) + "\n")

    payload = {
        "status": "READY_FOR_GPU_PHASE1_WHEN_RESOURCE_AVAILABLE",
        "scope": "offline approximate registration and source/PML envelope audit",
        "new_FDTD_executed": False,
        "thermal_executed": False,
        "PTE_executed": False,
        "registration": registered["fig3h_approximate_affine_registration"],
        "source_and_domain_envelope_audit": envelopes,
        "selected_phase1_numerical_box": {
            "domain_um": 64.0,
            "source_span_um": 50.0,
            "reason": (
                "preserve the existing 50-um source aperture; 60 um fails "
                "the runner's 0.5-um source/electrode-to-PML clearance gate"
            ),
        },
        "phase_order": [
            "new polarization-matched empty-stack a/b references",
            "nominal registered finite Device-A E||a optical gate",
            "nominal registered finite Device-A E||b optical gate",
            "identical material-intersection-density thermal/PTE solves",
            "only if nominal gates pass: sparse d=-1,1,3,5,7 um scan",
            "report separate profile maxima and +/-2 um registration sensitivity",
        ],
        "inputs": {
            "base_geometry_json": str(args.base_geometry_json.resolve()),
            "base_geometry_sha256": sha256(args.base_geometry_json),
            "fig3h_crop": str(args.fig3h_crop.resolve()),
            "fig3h_crop_sha256": sha256(args.fig3h_crop),
        },
    }
    (args.output_dir / "device_a_fig3h_registration_plan.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    with (args.output_dir / "device_a_fig3h_scan_positions.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["label", "distance_um", "beam_b_um", "beam_a_um"])
        for position in scan_positions:
            writer.writerow(
                [
                    position["label"],
                    position["distance_from_assumed_Fig3I_zero_um"],
                    *position["beam_center_code_um"],
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.3), constrained_layout=True)
    draw_rgb(axes[0], image)
    axes[0].plot(
        FIG3H_BLACK_SCAN_LINE_PX[:, 0],
        FIG3H_BLACK_SCAN_LINE_PX[:, 1],
        color="magenta",
        linewidth=2.5,
        label="digitized Fig. 3H scan",
    )
    nominal_px = np.asarray(
        [
            panel_center[0] + (nominal_beam[0] - flake_center[0]) * pixels_per_um,
            panel_center[1] - (nominal_beam[1] - flake_center[1]) * pixels_per_um,
        ]
    )
    axes[0].scatter(
        nominal_px[0], nominal_px[1], s=90, color="cyan", edgecolor="black",
        label="assumed Fig. 3I +3 um point",
    )
    axes[0].set_title("Fig. 3H raster registration assumption")
    axes[0].set_xlabel("crop x (pixel)")
    axes[0].set_ylabel("crop y (pixel)")
    axes[0].legend(fontsize=8)

    closed = np.vstack((flake, flake[0]))
    axes[1].fill(closed[:, 0], closed[:, 1], color="#d7bd70", alpha=0.55)
    axes[1].plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.5)
    axes[1].plot(
        scan_endpoints_um[:, 0], scan_endpoints_um[:, 1],
        color="magenta", linestyle="--", linewidth=2.0,
        label="registered Fig. 3H scan",
    )
    for position in scan_positions:
        center = np.asarray(position["beam_center_code_um"])
        axes[1].scatter(*center, s=42, color="tab:blue")
        axes[1].text(
            center[0] + 0.35, center[1],
            f"{position['distance_from_assumed_Fig3I_zero_um']:+g}",
            fontsize=8,
        )
    axes[1].scatter(
        *nominal_beam, s=100, color="cyan", edgecolor="black", zorder=5,
        label="phase-1 nominal (+3 um)",
    )
    circle = plt.Circle(
        nominal_beam, args.waist_um, fill=False, color="tab:red",
        linewidth=1.5, label=f"w0={args.waist_um:g} um",
    )
    axes[1].add_patch(circle)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-28, 23)
    axes[1].set_ylim(-24, 24)
    axes[1].set_xlabel("b (um)")
    axes[1].set_ylabel("a (um)")
    axes[1].set_title("Named approximate Device-A registration")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.savefig(args.output_dir / "DEVICE_A_FIG3H_APPROX_REGISTRATION.png", dpi=220)
    plt.close(fig)

    report = f"""# Device-A Figure 3H approximate registration plan

Status: `{payload['status']}`

This is a named approximation, not raw SPCM stage metrology. No FDTD,
thermal, PTE, adjoint, AD-FD, or optimization solve was run here.

## Registration assumption

- image right is `+b`; image up is `+a`; no rotation or shear is introduced;
- the visible 10-um bar gives `{pixels_per_um:.6g}` pixels/um;
- the Figure-3H map-panel centre is aligned to the Figure-2 digitized flake
  bounding-box centre;
- Figure-3I distance zero is placed at that common centre;
- the nominal Figure-3I peak coordinate is `+3 um` along `+a`.

The resulting nominal beam centre is
`(b,a)=({nominal_beam[0]:.6f},{nominal_beam[1]:.6f}) um`.  Its signed
distance from the nearest digitized flake boundary is
`{signed_distance:.6f} um` (positive means outside the flake).  This differs
qualitatively from the old, unregistered point placed 3 um *inside* a
non-boundary chord.

## Source/PML consequence

The registered source plus the full digitized electrode envelope does not
fit the old 60-um domain with the unchanged 50-um source span and the
runner's 0.5-um minimum PML-clearance gate.  That case has x clearance
`{envelopes['unchanged_60um_domain_50um_source']['minimum_PML_clearance_um']['x']:.6f} um`.
The phase-1 plan therefore preserves the 50-um source and uses a 64-um
lateral domain, giving x clearance
`{envelopes['recommended_64um_domain_50um_source']['minimum_PML_clearance_um']['x']:.6f} um`.

## Comparison contract

The experimental quantity is obtained from a scan profile.  The final
comparison must use separate maxima of `|Ia|` and `|Ib|` over the same
registered scan, not one arbitrary source point.  Phase 1 runs only the
nominal `+3 um` a/b pair.  The sparse `-1, +1, +3, +5, +7 um` scan is
authorized only after both optical gates pass.
"""
    (args.output_dir / "DEVICE_A_FIG3H_APPROX_REGISTRATION_REPORT.md").write_text(
        report
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
