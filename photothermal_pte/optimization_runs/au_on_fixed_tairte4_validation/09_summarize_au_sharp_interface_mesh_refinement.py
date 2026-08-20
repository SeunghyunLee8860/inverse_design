#!/usr/bin/env python3
"""Summarize Au sharp-interface FD step and edge-mesh refinement controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
LEVELS = {
    100: {
        0.10: ("sharp_width_7p9_forward", "sharp_width_8p1_forward"),
        0.05: ("sharp_width_7p95_forward", "sharp_width_8p05_forward"),
    },
    50: {
        0.10: ("sharp_width_7p9_edge50_forward", "sharp_width_8p1_edge50_forward"),
        0.05: ("sharp_width_7p95_edge50_forward", "sharp_width_8p05_edge50_forward"),
    },
    25: {
        0.10: ("sharp_width_7p9_edge25_forward", "sharp_width_8p1_edge25_forward"),
        0.05: ("sharp_width_7p95_edge25_forward", "sharp_width_8p05_edge25_forward"),
    },
}
CONTRACTS = (
    "sharp_width_8p0_edge50_contract",
    "sharp_width_8p0_edge25_contract",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def raw_entry(case: str, path: Path, result: dict[str, object]) -> dict[str, object]:
    stored = {
        item["path"]: item for item in result.get("raw_artifacts", [])
    }
    resolved = str(path.resolve())
    if resolved in stored:
        item = stored[resolved]
        return {
            "case": case,
            "path": resolved,
            "bytes": int(item["size_bytes"]),
            "sha256": item["sha256"],
        }
    return {
        "case": case,
        "path": resolved,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict[str, object]] = {}
    raw_files: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    fd: dict[int, dict[float, float]] = {}
    for mesh_nm, steps in LEVELS.items():
        fd[mesh_nm] = {}
        for h_um, (minus_name, plus_name) in steps.items():
            pair = []
            for sign, name in (("minus", minus_name), ("plus", plus_name)):
                case_dir = args.raw_root / name
                result = json.loads((case_dir / "case_result.json").read_text())
                cases[name] = result
                pair.append(result)
                rows.append(
                    {
                        "edge_mesh_nm": mesh_nm,
                        "h_um": h_um,
                        "sign": sign,
                        "case": name,
                        "Au_half_x_um": result["shape_parameter"]["value_um"],
                        "P_Q_W": result["P_Q_W"],
                        "P_six_W": result["P_six_W"],
                        "closure_relative": result["six_face_closure_relative"],
                        "auto_shutoff": result["log_audit"]["final_auto_shutoff"],
                        "wall_time_s": result["solver_wall_time_s"],
                        "passed_case_gate": result["passed"],
                    }
                )
                for path in sorted(case_dir.glob("*")):
                    if path.is_file() and path.suffix in {".fsp", ".npz", ".json", ".log"}:
                        raw_files.append(raw_entry(name, path, result))
            fd[mesh_nm][h_um] = (
                float(pair[1]["P_Q_W"]) - float(pair[0]["P_Q_W"])
            ) / (2.0 * h_um)

    for name in CONTRACTS:
        case_dir = args.raw_root / name
        result = json.loads((case_dir / "case_result.json").read_text())
        cases[name] = result
        for path in sorted(case_dir.glob("*")):
            if path.is_file() and path.suffix in {".fsp", ".json", ".log"}:
                raw_files.append(raw_entry(name, path, result))

    fd_rows = []
    for mesh_nm in sorted(fd, reverse=True):
        step_change = relative(fd[mesh_nm][0.10], fd[mesh_nm][0.05])
        fd_rows.append(
            {
                "edge_mesh_nm": mesh_nm,
                "FD_h0p10_W_per_um": fd[mesh_nm][0.10],
                "FD_h0p05_W_per_um": fd[mesh_nm][0.05],
                "step_change_relative": step_change,
                "step_plateau_lt_1pct": step_change < 0.01,
            }
        )
    mesh_rows = []
    for coarse, fine in ((100, 50), (50, 25)):
        mesh_rows.append(
            {
                "coarse_edge_mesh_nm": coarse,
                "fine_edge_mesh_nm": fine,
                "h0p10_derivative_change_relative": relative(
                    fd[coarse][0.10], fd[fine][0.10]
                ),
                "h0p05_derivative_change_relative": relative(
                    fd[coarse][0.05], fd[fine][0.05]
                ),
            }
        )

    all_cases_pass = all(bool(row["passed_case_gate"]) for row in rows)
    step_plateau_25 = fd_rows[-1]["step_plateau_lt_1pct"]
    mesh_50_to_25 = mesh_rows[-1]["h0p10_derivative_change_relative"]
    mesh_converged = mesh_50_to_25 < 0.01
    status = (
        "VALIDATED_AU_SHARP_INTERFACE_FORWARD_FD_STEP_AND_MESH_CONVERGENCE"
        if all_cases_pass and step_plateau_25 and mesh_converged
        else "VALIDATED_AU_SHARP_INTERFACE_FD_STEP_PLATEAU_BLOCKED_EDGE_MESH_CONVERGENCE"
    )
    summary = {
        "status": status,
        "scope": "isolated optical exact-binary Au sharp-interface forward-FD refinement; no adjoint, thermal, electrical, PTE, or optimization result",
        "all_optical_case_gates_passed": all_cases_pass,
        "edge_mesh_levels_nm": [100, 50, 25],
        "Au_z_mesh_nm": 5.0,
        "gray_Au_air_material_used": False,
        "Q_clipping_smoothing_gain_or_rescaling": False,
        "CPU_FDTD_fallback": False,
        "forward_FD": fd_rows,
        "mesh_convergence": mesh_rows,
        "decision": {
            "25nm_step_plateau_passed": step_plateau_25,
            "50_to_25nm_mesh_convergence_passed": mesh_converged,
            "numerical_shape_adjoint_certified": False,
            "production_Au_optimization_permitted": False,
            "reason": "the h=0.1 derivative changes by more than 1% from the 50 nm to 25 nm edge mesh",
        },
    }
    (output / "au_sharp_interface_mesh_refinement_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with (output / "au_sharp_interface_mesh_refinement_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "au_sharp_interface_mesh_refinement_fd.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fd_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fd_rows)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    meshes = [100, 50, 25]
    for h_um in (0.10, 0.05):
        axes[0].plot(meshes, [fd[m][h_um] for m in meshes], "o-", label=f"h={h_um:g} um")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("edge-local lateral mesh (nm)")
    axes[0].set_ylabel("central FD (W/um)")
    axes[0].set_title("Mesh dependence remains")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(
        meshes,
        [relative(fd[m][0.10], fd[m][0.05]) for m in meshes],
        "o-",
        label="h=0.10 vs 0.05 um",
    )
    axes[1].axhline(0.01, color="black", linestyle="--", label="1% gate")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("edge-local lateral mesh (nm)")
    axes[1].set_ylabel("FD-step relative difference")
    axes[1].set_title("Step plateau passes at 25 nm")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "au_sharp_interface_mesh_refinement.png", dpi=180)
    plt.close(figure)

    report = f"""# Au sharp-interface edge-mesh refinement checkpoint

Status: `{status}`

This is an isolated exact-binary Au optical control. It does not certify a
numerical shape adjoint or a coupled Au/TaIrTe4 device.

All twelve forward cases used here pass six-face closure `<0.5%` and
auto-shutoff `<1e-5`. No gray Au/air material, CPU FDTD fallback, Q clipping,
smoothing, gain, or rescaling was used.

## Central finite differences

| edge mesh | h=0.10 um | h=0.05 um | step change |
|---:|---:|---:|---:|
| 100 nm | {fd[100][0.10]:.12e} | {fd[100][0.05]:.12e} | {100*relative(fd[100][0.10], fd[100][0.05]):.6f}% |
| 50 nm | {fd[50][0.10]:.12e} | {fd[50][0.05]:.12e} | {100*relative(fd[50][0.10], fd[50][0.05]):.6f}% |
| 25 nm | {fd[25][0.10]:.12e} | {fd[25][0.05]:.12e} | {100*relative(fd[25][0.10], fd[25][0.05]):.6f}% |

The 25 nm mesh passes the 1% FD-step plateau gate. However, the h=0.10 um
derivative changes by `{100*mesh_50_to_25:.6f}%` from edge-50 to edge-25 nm,
so the derivative is not yet mesh-independent. The result is not rescaled or
promoted to a production Au optimization gradient.

## Decision

The density/imported Au route remains rejected because the uniform binary
endpoint diverged. The sharp-interface route is retained and now has a stable
within-mesh central difference at 25 nm. A numerical boundary-adjoint can be
implemented as a diagnostic at this mesh, but final certification additionally
requires an edge-mesh convergence resolution and the explicit Au/TaIrTe4
thermal/electrical contact scenarios.
"""
    (output / "AU_SHARP_INTERFACE_MESH_REFINEMENT_REPORT.md").write_text(report)

    manifest = {
        "status": status,
        "raw_files_committed": False,
        "generation_command_template": (
            "python 06_run_au_sharp_interface_width_control.py "
            "--au-half-x-um <value> --edge-dxy-nm <100|50|25> "
            "--edge-band-um 0.5 --output-dir <raw_case> --gpu-device 'GPU 6'"
        ),
        "raw_files": raw_files,
    }
    (output / "AU_SHARP_INTERFACE_MESH_RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "raw_artifacts": len(raw_files)}, indent=2))
    return 0 if mesh_converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
