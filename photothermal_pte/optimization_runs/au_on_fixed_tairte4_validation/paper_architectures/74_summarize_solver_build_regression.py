#!/usr/bin/env python3
"""Publish the R1 versus R1.2 periodic local-loss build regression."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results_periodic_T_Z_solver_build_regression"
RAW = Path("/home/seunghyun/tairte4/raw_artifacts")
CASES = {
    "T_R1p2_10nm": RAW
    / "paper_tairte4_architectures_sio2_si_reduced/T2024_MIR_4750_bare_Eb/"
    "T2024_TaIrTe4_optical_smoke.json",
    "T_R1_10nm": RAW
    / "periodic_T_Z_six_polarization_20260822/selected_Q_diagnostics/"
    "T_planar_10nm_mesh_control_corrected_runres/T2024_TaIrTe4_optical_smoke.json",
    "T_R1_25nm": RAW
    / "periodic_T_Z_six_polarization_20260822/selected_Q_diagnostics/"
    "T_planar_25nm_mesh_control_v2/T2024_TaIrTe4_optical_smoke.json",
    "Z_R1_25nm": RAW
    / "periodic_T_Z_six_polarization_20260822/selected_Q/Z/x_b/"
    "Z2022_M2_selected_Q.json",
    "Z_R1p2_25nm": RAW
    / "periodic_T_Z_six_polarization_20260822/selected_Q_diagnostics/"
    "Z_xb_5p3um_25nm_r1p2_build_control/Z2022_M2_selected_Q.json",
}
ROOTS = {
    "R1": Path("/opt/lumerical/v261"),
    "R1p2": Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for name, path in CASES.items():
        payload = json.loads(path.read_text())
        source = float(payload["source_power_W"])
        pflux = float(payload["P_flux_absorbed_W"])
        pq = float(payload["P_Q_pabs_periodic_W"])
        rows.append(
            {
                "case": name,
                "solver_version": payload["solver_version"],
                "mesh_nm": payload.get("lateral_mesh_nm")
                or float(payload["mesh_runsetup"]["min_dx_m"]) * 1e9,
                "P_flux_W": pflux,
                "P_Q_W": pq,
                "A_flux": pflux / source,
                "A_Q": pq / source,
                "closure_relative": float(payload["closure_relative"]),
                "status": payload["status"],
            }
        )
        artifacts.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    builds: dict[str, object] = {}
    for label, root in ROOTS.items():
        engine = root / "bin/fdtd-engine"
        api = root / "api/python/lumapi.py"
        builds[label] = {
            "root": str(root),
            "engine_size_bytes": engine.stat().st_size,
            "engine_sha256": sha256(engine),
            "engine_mtime": engine.stat().st_mtime,
            "lumapi_size_bytes": api.stat().st_size,
            "lumapi_sha256": sha256(api),
            "lumapi_mtime": api.stat().st_mtime,
        }

    by_name = {str(row["case"]): row for row in rows}
    gates = {
        "R1p2_T_closure_lt_0p5pct": by_name["T_R1p2_10nm"]["closure_relative"] < 0.005,
        "R1p2_Z_closure_lt_0p5pct": by_name["Z_R1p2_25nm"]["closure_relative"] < 0.005,
        "R1_T_controls_fail_closure": all(
            by_name[name]["closure_relative"] >= 0.005
            for name in ("T_R1_10nm", "T_R1_25nm")
        ),
        "R1_Z_fails_closure": by_name["Z_R1_25nm"]["closure_relative"] >= 0.005,
        "R1_10_to_25nm_flux_difference_lt_0p5pct": abs(
            by_name["T_R1_10nm"]["P_flux_W"] - by_name["T_R1_25nm"]["P_flux_W"]
        )
        / abs(by_name["T_R1_10nm"]["P_flux_W"])
        < 0.005,
    }
    status = (
        "VALIDATED_PERIODIC_Q_REQUIRES_V261_R1P2_BUILD_8P35P4522"
        if all(gates.values())
        else "FAILED_SOLVER_BUILD_REGRESSION_AUDIT"
    )
    summary = {
        "status": status,
        "scope": "periodic optical flux/local-Q build regression; no thermal/weighting/PTE",
        "interpretation": (
            "The displayed v261 label hid distinct R1 and R1.2 builds. "
            "R1.2 closes T and centered-Z local loss; R1 does not."
        ),
        "cases": rows,
        "solver_builds": builds,
        "gates": gates,
        "production_rule": "periodic reference-Q artifacts must use solver version 8.35.4522",
        "no_clipping_smoothing_gain_or_rescaling": True,
    }
    (OUTPUT / "PERIODIC_Q_SOLVER_BUILD_REGRESSION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (OUTPUT / "periodic_q_solver_build_regression.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )

    labels = [str(row["case"]) for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(positions - 0.18, [row["A_flux"] for row in rows], 0.36, label="flux")
    axes[0].bar(positions + 0.18, [row["A_Q"] for row in rows], 0.36, label="local Q")
    axes[0].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_ylabel("absorbed fraction")
    axes[0].legend()
    axes[0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0].set_title("Flux and local-loss result")
    axes[1].bar(positions, [100 * row["closure_relative"] for row in rows])
    axes[1].axhline(0.5, color="r", linestyle="--", label="0.5% gate")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("flux-Q closure (%)")
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1].set_title("Closure by solver build")
    axes[1].legend()
    fig.suptitle("v261 is not one numerical build: R1 versus R1.2")
    fig.savefig(OUTPUT / "periodic_q_solver_build_regression.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Periodic optical-Q solver-build regression",
        "",
        f"Status: `{status}`",
        "",
        "The same `v261` directory name referred to two different solver builds. "
        "The older R1 build fails the flux/local-loss identity for both compact T and "
        "centered Z, whereas R1.2 passes. This is not a 10-versus-25-nm mesh effect: "
        "the two R1 T fluxes agree within 0.5% and both fail closure.",
        "",
        "| case | solver | mesh (nm) | A_flux | A_Q | closure | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['solver_version']} | {row['mesh_nm']:.3g} | "
            f"{row['A_flux']:.6f} | {row['A_Q']:.6f} | "
            f"{row['closure_relative']:.3%} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "Production rule: use and record `8.35.4522` (2026 R1.2) for all "
            "periodic local-Q certificates. Raw FSP/NPZ files remain outside Git.",
            "",
            "No thermal, weighting-field, PTE, adjoint, or optimization calculation "
            "is included in this audit.",
        ]
    )
    (OUTPUT / "PERIODIC_Q_SOLVER_BUILD_REGRESSION_REPORT.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
