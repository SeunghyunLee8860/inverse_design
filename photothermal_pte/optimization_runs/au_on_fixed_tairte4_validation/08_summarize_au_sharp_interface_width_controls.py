#!/usr/bin/env python3
"""Summarize sharp-interface Au width controls without promoting an adjoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
CASES = (
    (5.8, "sharp_width_5p8_forward", "strong_direction_closure_failure"),
    (7.8, "sharp_width_7p8_forward", "fd_h0p2_minus"),
    (7.9, "sharp_width_7p9_forward", "fd_h0p1_minus"),
    (7.95, "sharp_width_7p95_forward", "fd_h0p05_minus"),
    (8.05, "sharp_width_8p05_forward", "fd_h0p05_plus"),
    (8.1, "sharp_width_8p1_forward", "fd_h0p1_plus"),
    (8.2, "sharp_width_8p2_forward", "fd_h0p2_plus"),
    (9.8, "sharp_width_9p8_forward", "near_null_h0p2_minus"),
    (10.2, "sharp_width_10p2_forward", "near_null_h0p2_plus"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    loaded: dict[float, dict[str, object]] = {}
    raw_files: list[dict[str, object]] = []
    for width_um, directory, purpose in CASES:
        case_dir = args.raw_root / directory
        result_path = case_dir / "case_result.json"
        result = json.loads(result_path.read_text())
        loaded[width_um] = result
        rows.append(
            {
                "Au_half_x_um": width_um,
                "purpose": purpose,
                "status": result["status"],
                "passed_case_gate": bool(result.get("passed", False)),
                "P_Q_W": result.get("P_Q_W"),
                "P_six_W": result.get("P_six_W"),
                "six_face_closure_relative": result.get("six_face_closure_relative"),
                "auto_shutoff": result.get("log_audit", {}).get("final_auto_shutoff"),
                "solver_wall_time_s": result.get("solver_wall_time_s"),
            }
        )
        for path in sorted(case_dir.glob("*")):
            if path.is_file() and path.suffix in {".fsp", ".npz", ".json", ".log"}:
                raw_files.append(
                    {
                        "case": directory,
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )

    fd_rows: list[dict[str, float]] = []
    for h_um in (0.2, 0.1, 0.05):
        minus = float(loaded[8.0 - h_um]["P_Q_W"])
        plus = float(loaded[8.0 + h_um]["P_Q_W"])
        derivative = (plus - minus) / (2.0 * h_um)
        fd_rows.append(
            {
                "h_um": h_um,
                "P_Q_minus_W": minus,
                "P_Q_plus_W": plus,
                "central_FD_W_per_um": derivative,
                "pair_signal_relative": relative(plus, minus),
            }
        )
    for index, item in enumerate(fd_rows):
        if index == 0:
            item["relative_change_from_previous_h"] = None
        else:
            item["relative_change_from_previous_h"] = relative(
                item["central_FD_W_per_um"],
                fd_rows[index - 1]["central_FD_W_per_um"],
            )

    plateau_h0p2_to_h0p1 = fd_rows[1]["relative_change_from_previous_h"]
    plateau_h0p1_to_h0p05 = fd_rows[2]["relative_change_from_previous_h"]
    all_fd_cases_close = all(
        bool(loaded[width]["passed"])
        for width in (7.8, 7.9, 7.95, 8.05, 8.1, 8.2)
    )
    plateau_passed = bool(
        all_fd_cases_close
        and plateau_h0p2_to_h0p1 < 0.01
        and plateau_h0p1_to_h0p05 < 0.01
    )
    status = (
        "VALIDATED_SHARP_INTERFACE_FORWARD_FD_PLATEAU"
        if plateau_passed
        else "FAILED_SHARP_INTERFACE_FORWARD_FD_PLATEAU_AT_100NM_EDGE_MESH"
    )

    summary = {
        "status": status,
        "scope": "isolated scalar-Au sharp-interface forward controls; no numerical adjoint, thermal, electrical, PTE, or optimization result",
        "baseline_half_width_um": 8.0,
        "moved_boundaries": ["x_min", "x_max"],
        "fixed_boundaries": ["y_min", "y_max", "z_min", "z_max"],
        "gray_Au_air_material_used": False,
        "Q_clipping_smoothing_gain_or_rescaling": False,
        "CPU_FDTD_fallback": False,
        "edge_lateral_mesh_nm": 100.0,
        "Au_z_mesh_nm": 5.0,
        "case_gate": {"six_face_closure_relative_lt": 0.005, "auto_shutoff_lt": 1e-5},
        "plateau_gate": {"successive_central_FD_relative_change_lt": 0.01},
        "cases": rows,
        "central_FD": fd_rows,
        "plateau_h0p2_to_h0p1_relative": plateau_h0p2_to_h0p1,
        "plateau_h0p1_to_h0p05_relative": plateau_h0p1_to_h0p05,
        "failed_strong_direction": {
            "Au_half_x_um": 5.8,
            "six_face_closure_relative": loaded[5.8]["six_face_closure_relative"],
            "reason": "closure exceeds 0.5%; +6.2 um counterpart was not run",
        },
        "decision": {
            "forward_FD_plateau_validated": plateau_passed,
            "shape_adjoint_AD_FD_permitted_at_current_mesh": False,
            "next_minimal_gate": "repeat the h=0.1 and h=0.05 controls with an edge-local 50 nm lateral mesh before numerical shape-adjoint comparison",
        },
    }
    (output / "au_sharp_interface_width_fd_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    with (output / "au_sharp_interface_width_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "au_sharp_interface_width_fd.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fd_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fd_rows)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    widths = [float(row["Au_half_x_um"]) for row in rows if 7.8 <= float(row["Au_half_x_um"]) <= 8.2]
    powers = [float(loaded[width]["P_Q_W"]) for width in widths]
    axes[0].plot(widths, powers, "o-")
    axes[0].set_xlabel("Au half-width (um)")
    axes[0].set_ylabel("P_Q (W)")
    axes[0].set_title("Sharp-interface forward controls")
    axes[0].grid(alpha=0.3)
    hs = [item["h_um"] for item in fd_rows]
    derivatives = [item["central_FD_W_per_um"] for item in fd_rows]
    axes[1].plot(hs, derivatives, "o-")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("central-FD half step h (um)")
    axes[1].set_ylabel("dP_Q/d(half-width) (W/um)")
    axes[1].set_title("No FD plateau at 100 nm edge mesh")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "au_sharp_interface_width_fd_plateau.png", dpi=180)
    plt.close(figure)

    report = f"""# Au sharp-interface width forward-FD checkpoint

Status: `{status}`

This checkpoint is an isolated optical shape control. It is not a numerical
shape-adjoint certificate and contains no thermal, electrical, PTE, or
optimization result.

## Contract

- exact scalar Au at 10 um: `n + ik = 12.1 + 69.2i`
- binary sharp Au/air boundary; no gray material
- symmetric x-normal faces moved about an 8.0 um half-width baseline
- fixed 20 um y span and 50 nm Au thickness
- 100 nm lateral edge mesh and 5 nm Au z mesh
- GPU FDTD only; six PML; no Q clipping, smoothing, gain, or rescaling

All six cases used in the central differences passed the individual optical
closure and auto-shutoff gates. The central derivatives were:

- h=0.20 um: `{fd_rows[0]['central_FD_W_per_um']:.12e} W/um`
- h=0.10 um: `{fd_rows[1]['central_FD_W_per_um']:.12e} W/um`
- h=0.05 um: `{fd_rows[2]['central_FD_W_per_um']:.12e} W/um`

The 0.20 -> 0.10 um change is `{100.0 * plateau_h0p2_to_h0p1:.6f}%`; the
0.10 -> 0.05 um change is `{100.0 * plateau_h0p1_to_h0p05:.6f}%`. The latter
fails the 1% plateau gate. A 50 nm boundary motion is below the present
100 nm lateral edge mesh, so this result is preserved as a mesh-resolution
failure rather than normalized or promoted.

The earlier 5.8 um strong-direction case also remains fail-closed because its
six-face closure was `{100.0 * float(loaded[5.8]['six_face_closure_relative']):.6f}%`.

## Decision

The density route remains rejected because uniform binary imported Au
diverged. The exact-binary sharp-interface route remains physically viable,
but its numerical AD-FD gate is not yet passed. The minimum next calculation
is an edge-local 50 nm lateral-mesh repeat of h=0.10 and 0.05 um, followed by
the numerical boundary-adjoint comparison only if the forward FD plateaus.
"""
    (output / "AU_SHARP_INTERFACE_WIDTH_FD_REPORT.md").write_text(report)

    manifest = {
        "status": status,
        "raw_files_committed": False,
        "generation_commands": [
            "python 06_run_au_sharp_interface_width_control.py --au-half-x-um <value> --output-dir <raw_case> --gpu-device 'GPU 6'"
        ],
        "raw_files": raw_files,
    }
    (output / "AU_SHARP_INTERFACE_WIDTH_RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "raw_artifacts": len(raw_files)}, indent=2))
    return 0 if plateau_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
