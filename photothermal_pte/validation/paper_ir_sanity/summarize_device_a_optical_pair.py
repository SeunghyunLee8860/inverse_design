#!/usr/bin/env python3
"""Summarize the completed Device-A a/b optical pair without changing Q."""

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

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)


TARGET_POWER_W = 285e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(case_dir: Path, audit_path: Path, label: str) -> tuple[dict, dict]:
    result_path = case_dir / "case_result.json"
    artifact_path = case_dir / "finite_q_on_artifact.npz"
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"{label} optical case is not complete")
    run = result["run_result"]
    incident_unit = float(run["normalization"]["incident_power_W_at_1_W_m2"])
    scale = TARGET_POWER_W / incident_unit
    with np.load(artifact_path, allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        q = np.asarray(raw["Q_on_W_m3"], float)
    bounds = result["pre_run_contract"]["geometry"][
        "pabs_nominal_control_volume_bounds_m"
    ]
    wz = runner.bounded_dual_cell_weights(z, *bounds["z"])
    areal = np.einsum("k,ijk->ij", wz, q, optimize=True) * scale
    metrics = {
        "polarization": label,
        "status": result["status"],
        "auto_shutoff": run["auto_shutoff"]["final_value"],
        "P_Q_unit_W": run["P_Q_W"],
        "P_six_unit_W": run["P_six_face_W"],
        "six_face_closure": run["six_face_relative_closure"],
        "incident_power_unit_W": incident_unit,
        "scale_to_285uW": scale,
        "P_Q_at_285uW_W": run["P_Q_W"] * scale,
        "P_six_at_285uW_W": run["P_six_face_W"] * scale,
        "component_power_unit_W": run["component_power_W"],
        "component_power_at_285uW_W": {
            key: value * scale for key, value in run["component_power_W"].items()
        },
        "Q_hotspot": run["Q_hotspot"],
        "negative_Q_voxel_count": run["negative_Q_voxel_count"],
        "minimum_Q_W_m3": run["minimum_Q_W_m3"],
        "maximum_field_index_coordinate_mismatch_m": run[
            "native_Yee_mesh_audit"
        ]["independent_field_index_pairing"]["maximum_coordinate_mismatch_m"],
        "material_support_power_unit_W": audit[
            "power_at_unit_central_intensity_W"
        ],
        "artifact": {
            "path": str(artifact_path.resolve()),
            "size_bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        },
        "FSP": {
            "path": str((case_dir / "finite_2um_optical_q.fsp").resolve()),
            "size_bytes": (case_dir / "finite_2um_optical_q.fsp").stat().st_size,
            "sha256": sha256(case_dir / "finite_2um_optical_q.fsp"),
        },
        "result": {
            "path": str(result_path.resolve()),
            "sha256": sha256(result_path),
        },
        "audit": {
            "path": str(audit_path.resolve()),
            "sha256": sha256(audit_path),
        },
    }
    return metrics, {"x_m": x, "y_m": y, "areal_W_m2": areal}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-a", type=Path, required=True)
    parser.add_argument("--case-b", type=Path, required=True)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a, field_a = load_case(args.case_a, args.audit_a, "a")
    b, field_b = load_case(args.case_b, args.audit_b, "b")
    ratio = b["P_Q_at_285uW_W"] / a["P_Q_at_285uW_W"]
    payload = {
        "status": "VALIDATED_DEVICE_A_SINGLE_POSITION_OPTICAL_PAIR",
        "illumination_contract": {
            "wavelength_m": 11e-6,
            "incident_power_each_polarization_W": TARGET_POWER_W,
            "source": "scalar Gaussian with explicitly assumed physical w0=12 um",
            "domain": "60x60 um, six 24-layer PML, no periodic/Bloch",
            "mesh": "50 nm local x/y and 5 nm TaIrTe4 z, conformal variant 1",
            "polarization_dependent_Q_matching_or_rescaling": False,
        },
        "a": a,
        "b": b,
        "ratios": {
            "P_Q_b_over_a_at_equal_285uW_incident": ratio,
            "P_six_b_over_a_at_equal_285uW_incident": (
                b["P_six_at_285uW_W"] / a["P_six_at_285uW_W"]
            ),
        },
        "gates": {
            "a_closure_lt_0p5_percent": a["six_face_closure"] < 0.005,
            "b_closure_lt_0p5_percent": b["six_face_closure"] < 0.005,
            "a_auto_shutoff_lt_1e_5": a["auto_shutoff"] < 1e-5,
            "b_auto_shutoff_lt_1e_5": b["auto_shutoff"] < 1e-5,
            "no_negative_Q": (
                a["negative_Q_voxel_count"] == 0
                and b["negative_Q_voxel_count"] == 0
            ),
        },
        "no_Q_modification": (
            "no clipping, smoothing, gain, global rescaling, tiling, source "
            "deletion, or polarization matching"
        ),
        "thermal_run_in_this_summary": False,
        "PTE_run_in_this_summary": False,
    }
    if not all(payload["gates"].values()):
        payload["status"] = "FAILED_DEVICE_A_SINGLE_POSITION_OPTICAL_PAIR"
    (args.output_dir / "device_a_optical_pair_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    with (args.output_dir / "device_a_optical_pair_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "polarization",
                "P_Q_unit_W",
                "P_six_unit_W",
                "closure",
                "auto_shutoff",
                "P_Q_at_285uW_W",
                "Qx_unit_W",
                "Qy_unit_W",
                "Qz_unit_W",
                "TaIrTe4_fraction",
                "Ti_fraction",
                "Au_fraction",
                "conformal_fraction",
            ]
        )
        for case in (a, b):
            support = case["material_support_power_unit_W"]
            total = support["common_grid_total_W"]
            writer.writerow(
                [
                    case["polarization"],
                    case["P_Q_unit_W"],
                    case["P_six_unit_W"],
                    case["six_face_closure"],
                    case["auto_shutoff"],
                    case["P_Q_at_285uW_W"],
                    case["component_power_unit_W"]["x"],
                    case["component_power_unit_W"]["y"],
                    case["component_power_unit_W"]["z"],
                    support["TaIrTe4_W"] / total,
                    support["Ti_W"] / total,
                    support["Au_W"] / total,
                    support["conformal_interface_ambiguous_W"] / total,
                ]
            )

    common_vmax = max(
        float(np.max(field_a["areal_W_m2"])),
        float(np.max(field_b["areal_W_m2"])),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for ax, label, field in zip(axes, ("E || a", "E || b"), (field_a, field_b)):
        image = ax.pcolormesh(
            field["x_m"] * 1e6,
            field["y_m"] * 1e6,
            field["areal_W_m2"].T,
            shading="nearest",
            cmap="inferno",
            vmin=0.0,
            vmax=common_vmax,
        )
        ax.set(
            title=label,
            xlabel="code x = crystal b [um]",
            ylabel="code y = crystal a [um]",
            aspect="equal",
        )
        fig.colorbar(image, ax=ax, label="depth-integrated Q at 285 uW [W/m2]")
    fig.savefig(args.output_dir / "DEVICE_A_OPTICAL_Q_PAIR.png", dpi=220)
    plt.close(fig)

    report = f"""# Device A single-position optical pair

Status: `{payload['status']}`

Both cases use the same physical incident power of 285 uW. Their matching
empty-stack references are polarization-specific measurements of the same
source geometry; Q is not matched or rescaled between polarizations.

| Polarization | P_Q at 285 uW [W] | P_six [W] | closure | auto-shutoff |
|---|---:|---:|---:|---:|
| E parallel a | {a['P_Q_at_285uW_W']:.12e} | {a['P_six_at_285uW_W']:.12e} | {a['six_face_closure']:.4%} | {a['auto_shutoff']:.6e} |
| E parallel b | {b['P_Q_at_285uW_W']:.12e} | {b['P_six_at_285uW_W']:.12e} | {b['six_face_closure']:.4%} | {b['auto_shutoff']:.6e} |

The equal-incident-power absorbed-power ratio is
`P_Q,b/P_Q,a = {ratio:.9f}`. This optical checkpoint is not yet the terminal
current comparison. Metal/interface thermal handling is reported separately.
"""
    (args.output_dir / "DEVICE_A_OPTICAL_PAIR_REPORT.md").write_text(report)
    manifest = {
        "raw_artifacts_are_not_committed": True,
        "a": {"Q_NPZ": a["artifact"], "FSP": a["FSP"]},
        "b": {"Q_NPZ": b["artifact"], "FSP": b["FSP"]},
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_OPTICAL_PAIR.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": payload["status"], **payload["ratios"]}, indent=2))
    return 0 if payload["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
