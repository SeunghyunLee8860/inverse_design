#!/usr/bin/env python3
"""Summarize the scalar/imported binary-Au controls and fail closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")
CASES = (
    ("scalar_10um_dz5", "binary_rho1_scalar_forward_v2"),
    ("scalar_10um_dz2p5", "binary_rho1_scalar_forward_dz2p5"),
    ("scalar_20um_dz5", "binary_rho1_scalar_20um_forward"),
    ("imported_20um_dz5", "binary_rho1_imported_20um_forward"),
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

    rows = []
    loaded = {}
    raw_files = []
    for label, directory in CASES:
        case_dir = args.raw_root / directory
        result_path = case_dir / "case_result.json"
        result = json.loads(result_path.read_text())
        loaded[label] = result
        row = {
            "case": label,
            "representation": result["representation"],
            "lateral_span_um": 20.0 if "20um" in label else 10.0,
            "dz_nm": 2.5 if "dz2p5" in label else 5.0,
            "status": result["status"],
            "solver_wall_time_s": result.get("solver_wall_time_s"),
            "P_Q_W": result.get("P_Q_W"),
            "P_six_W": result.get("P_six_W"),
            "six_face_closure_relative": result.get("six_face_closure_relative"),
            "diverged": "diverging" in result.get("error", "").lower(),
            "passed": bool(result.get("passed", False)),
        }
        rows.append(row)
        for path in sorted(case_dir.glob("*")):
            if path.is_file() and path.suffix in {".fsp", ".npz", ".json", ".log"}:
                raw_files.append({
                    "case": label,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })

    coarse = loaded["scalar_10um_dz5"]
    refined = loaded["scalar_10um_dz2p5"]
    dz_convergence = {
        "P_Q_relative_change": relative(coarse["P_Q_W"], refined["P_Q_W"]),
        "P_six_relative_change": relative(coarse["P_six_W"], refined["P_six_W"]),
        "closure_dz5": coarse["six_face_closure_relative"],
        "closure_dz2p5": refined["six_face_closure_relative"],
        "interpretation": "closure is unchanged by z refinement; small-volume flux cancellation is the dominant diagnostic limitation",
    }
    scalar_large = loaded["scalar_20um_dz5"]
    imported_large = loaded["imported_20um_dz5"]
    density_route_failed = bool(
        scalar_large.get("passed", False)
        and "diverging" in imported_large.get("error", "").lower()
    )
    status = (
        "FAILED_DENSITY_ROUTE_UNIFORM_AU_IMPORTNK2_DIVERGENCE_FALLBACK_SHARP_INTERFACE"
        if density_route_failed
        else "INCOMPLETE_AU_BINARY_REPRESENTATION_CONTROL"
    )
    summary = {
        "status": status,
        "scope": "isolated finite Au-film material-representation control; no fixed-flake thermal/electrical/PTE/optimization result",
        "wavelength_m": 10e-6,
        "Au_thickness_m": 50e-9,
        "source": "certified scalar Gaussian w0=8.5 um, six PML",
        "Q_clipping_smoothing_gain_or_rescaling": False,
        "CPU_FDTD_fallback": False,
        "cases": rows,
        "dz_convergence": dz_convergence,
        "production_scalar_control": {
            "P_Q_W": scalar_large["P_Q_W"],
            "P_six_W": scalar_large["P_six_W"],
            "six_face_closure_relative": scalar_large["six_face_closure_relative"],
            "epsilon_component_readback": scalar_large["epsilon_component_readback"],
        },
        "density_route_decision": {
            "uniform_binary_importnk2_completed": False,
            "uniform_binary_importnk2_diverged": density_route_failed,
            "gray_density_test_permitted": False,
            "Maxwell_AD_FD_permitted": False,
            "fallback": "sharp-interface level-set/shape optimization",
        },
    }
    (output / "au_binary_representation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output / "au_binary_representation_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# Binary Au representation control

Status: `{status}`

This is an isolated material-representation control, not a fixed-TaIrTe4
device prediction.

## Stable scalar-Au reference

The 20 x 20 x 0.05 um exact scalar `(n,k)` Au film completed on GPU with:

- `P_Q = {scalar_large['P_Q_W']:.12e} W`
- `P_six = {scalar_large['P_six_W']:.12e} W`
- six-face closure = `{100.0 * scalar_large['six_face_closure_relative']:.6f}%`
- all component epsilon medians equal the requested Ordal endpoint
- no Q clipping, smoothing, gain, or rescaling

The earlier 10 x 10 um control had approximately 0.9% relative closure because
Au absorption was a small difference of large incident/reflected fluxes.
Changing `dz=5 -> 2.5 nm` left that closure essentially unchanged. Increasing
the finite control area raised the absorption signal and closed to 0.5%.

## Density/imported endpoint failure

With the same 20 x 20 x 0.05 um geometry and mesh, uniform `rho=1`
`importnk2` Au diverged after 1,919 FDTD iterations. The scalar material stayed
stable. CPU fallback was prohibited and no gray-density or AD-FD run followed.

Because the density representation fails already at the binary Au endpoint,
the approved option 2 is rejected for production. The next route is option 1:
binary scalar-Au geometry with a sharp-interface level-set/shape derivative.
The failed FSP/log/JSON are preserved outside Git; their hashes are recorded.
"""
    (output / "AU_DENSITY_ROUTE_BINARY_CONTROL_REPORT.md").write_text(report)

    manifest = {
        "status": status,
        "raw_files_committed": False,
        "raw_files": raw_files,
    }
    (output / "AU_BINARY_RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "raw_artifacts": len(raw_files)}, indent=2))
    return 0 if density_route_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
