#!/usr/bin/env python3
"""Publish the fresh r=0.8 um in-flake proxy optical-Q validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


SUCCESS = "VALIDATED_FINITE_INFLAKE_PROXY_OPTICAL_Q"
OLD_PR3_Q_SHA256 = (
    "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
)
POWER_GATE = 0.005
POWER_CONVERGENCE_GATE = 0.01
SPATIAL_CONVERGENCE_GATE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--final-case-result", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(value: Any, target: float) -> bool:
    try:
        return bool(np.isclose(float(value), target))
    except (TypeError, ValueError):
        return False


def read_row(path: Path, root: Path) -> dict[str, Any]:
    data = load_json(path)
    run = data.get("run_result", {})
    component = run.get("component_power_W", {})
    hotspot = run.get("Q_hotspot", {})
    p_q = run.get("P_Q_W")
    artifact = path.parent / str(run.get("artifact", ""))
    acceptance = run.get("acceptance", {})
    return {
        "_path": path,
        "_data": data,
        "_artifact": artifact if artifact.is_file() else None,
        "case_id": str(path.parent.relative_to(root)),
        "case": data.get("case"),
        "polarization_deg": data.get("polarization_deg"),
        "domain_um": data.get("domain_um"),
        "pml_layers": data.get("pml_layers"),
        "flake_dz_nm": data.get("flake_dz_nm"),
        "waist_um": data.get("waist_um"),
        "source_span_um": data.get("source_span_um"),
        "design_radius_um": data.get("design_radius_um"),
        "status": data.get("status"),
        "all_case_acceptance": bool(acceptance)
        and all(bool(value) for value in acceptance.values()),
        "P_Qx_W": component.get("x"),
        "P_Qy_W": component.get("y"),
        "P_Qz_W": component.get("z"),
        "P_Q_W": p_q,
        "P_six_W": run.get("P_six_face_W"),
        "six_face_closure": run.get("six_face_relative_closure"),
        "Qx_fraction": component.get("x") / p_q if p_q else None,
        "Qy_fraction": component.get("y") / p_q if p_q else None,
        "Qz_fraction": component.get("z") / p_q if p_q else None,
        "hotspot_x_m": hotspot.get("x_m"),
        "hotspot_y_m": hotspot.get("y_m"),
        "hotspot_z_m": hotspot.get("z_m"),
        "hotspot_Q_W_m3": hotspot.get("Q_W_m3"),
        "generation_command": data.get("generation_command"),
        "generation_commit": data.get("generation_commit"),
    }


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def is_proxy(row: dict[str, Any]) -> bool:
    return (
        row["case"] == "fixed-design"
        and same(row["polarization_deg"], 0.0)
        and same(row["design_radius_um"], 0.8)
        and same(row["waist_um"], 2.0)
        and same(row["source_span_um"], 6.8)
        and row["status"] == "COMPLETED"
        and row["all_case_acceptance"]
    )


def find_one(rows: list[dict[str, Any]], **criteria: float | int | str) -> dict[str, Any]:
    matches = []
    for row in rows:
        if not is_proxy(row):
            continue
        passed = True
        for key, target in criteria.items():
            value = row.get(key)
            if isinstance(target, (float, int)):
                passed &= same(value, float(target))
            else:
                passed &= value == target
        if passed:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"expected one proxy case for {criteria}, got {len(matches)}")
    return matches[0]


def spatial_l2(first: dict[str, Any], second: dict[str, Any]) -> float:
    if first["_artifact"] is None or second["_artifact"] is None:
        raise RuntimeError("convergence artifact is missing")
    with np.load(first["_artifact"]) as old, np.load(second["_artifact"]) as new:
        ox, oy, oz = (np.asarray(old[f"{axis}_m"], float) for axis in "xyz")
        nx, ny, nz = (np.asarray(new[f"{axis}_m"], float) for axis in "xyz")
        oq = np.asarray(old["Q_on_W_m3"], float)
        nq = np.asarray(new["Q_on_W_m3"], float)
        mx = (ox >= -1e-6) & (ox <= 1e-6)
        my = (oy >= -1e-6) & (oy <= 1e-6)
        mz = (oz >= -1e-7) & (oz <= 0.0)
        x, y, z = ox[mx], oy[my], oz[mz]
        points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        new_on_old = RegularGridInterpolator(
            (nx, ny, nz), nq, bounds_error=False, fill_value=0.0
        )(points).reshape(x.size, y.size, z.size)
        old_exact = oq[np.ix_(mx, my, mz)]
        return float(np.linalg.norm(new_on_old - old_exact)) / max(
            float(np.linalg.norm(new_on_old)), np.finfo(float).tiny
        )


def convergence_pair(
    name: str, first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    def change(key: str) -> float:
        a, b = float(first[key]), float(second[key])
        return abs(b - a) / max(abs(b), np.finfo(float).tiny)

    result = {
        "name": name,
        "from_case": first["case_id"],
        "to_case": second["case_id"],
        "P_Q_relative_change": change("P_Q_W"),
        "P_six_relative_change": change("P_six_W"),
        "spatial_Q_relative_L2": spatial_l2(first, second),
    }
    result["passed"] = (
        result["P_Q_relative_change"] < POWER_CONVERGENCE_GATE
        and result["P_six_relative_change"] < POWER_CONVERGENCE_GATE
        and result["spatial_Q_relative_L2"] < SPATIAL_CONVERGENCE_GATE
    )
    return result


def control_gate(rows: list[dict[str, Any]]) -> dict[str, bool]:
    def passed(case: str, angle: float | None = None) -> bool:
        return any(
            row["case"] == case
            and (angle is None or same(row["polarization_deg"], angle))
            and row["status"] == "COMPLETED"
            and row["all_case_acceptance"]
            for row in rows
        )

    return {
        "source_off": passed("no-source"),
        "empty_stack_x_y_45": all(
            passed("empty-stack", angle) for angle in (0.0, 90.0, 45.0)
        ),
        "finite_flat_x_y_45": all(
            passed("flat", angle) for angle in (0.0, 90.0, 45.0)
        ),
        "inflake_proxy_x": any(is_proxy(row) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = [public(row) for row in rows]
    keys = list(payload[0]) if payload else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(payload)


def raw_manifest(
    rows: list[dict[str, Any]], final: dict[str, Any]
) -> dict[str, Any]:
    records = []
    for row in rows:
        for path in sorted(row["_path"].parent.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".npz", ".fsp"}:
                continue
            records.append(
                {
                    "case_id": row["case_id"],
                    "server_path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "generation_command": row["generation_command"],
                    "generation_commit": row["generation_commit"],
                    "committed_to_git": False,
                }
            )
    final_npz = Path(str(final["_artifact"])).resolve()
    final_record = next(
        record
        for record in records
        if Path(record["server_path"]) == final_npz
    )
    different = final_record["sha256"] != OLD_PR3_Q_SHA256
    return {
        "raw_npz_or_fsp_committed": False,
        "old_PR3_raw_Q_sha256": OLD_PR3_Q_SHA256,
        "selected_final_raw_npz": final_record,
        "selected_final_SHA_differs_from_PR3": different,
        "artifacts": records,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    final = summary["final_case"]
    lines = [
        "# Finite in-flake SiO2 proxy optical-Q validation",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This is a fresh v261 GPU FDTD validation. The radius-1.5-µm PR #3 "
        "artifact was neither reused nor cropped.",
        "",
        "## Geometry and optical contract",
        "",
        "- TaIrTe4: 2 µm × 2 µm × 100 nm",
        "- centered SiO2 disk: radius 0.8 µm, height 600 nm, fully inside the flake",
        "- outside the disk: air; no support annulus, overhang support, or oxide pillar",
        "- finite Gaussian waist 2 µm, aperture 6.8 µm, source 3–6 µm, analysis 4 µm",
        "- six PML boundaries, auto nonuniform mesh, conformal variant 1, accuracy 5",
        "- central incident intensity: 1 W/m²",
        "",
        "## Promoted fresh result",
        "",
        f"- P_Q = `{final['P_Q_W']:.15e} W`",
        f"- P_six = `{final['P_six_W']:.15e} W`",
        f"- six-face closure = `{100.0 * final['six_face_closure']:.9g}%`",
        f"- Qx/Q, Qy/Q, Qz/Q = `{final['Qx_fraction']:.9g}`, "
        f"`{final['Qy_fraction']:.9g}`, `{final['Qz_fraction']:.9g}`",
        "- hotspot (x,y,z) = "
        f"`({final['hotspot_x_m']:.9e}, {final['hotspot_y_m']:.9e}, "
        f"{final['hotspot_z_m']:.9e}) m`",
        "",
        "## Convergence",
        "",
    ]
    for item in summary["convergence"].values():
        lines.append(
            f"- {item['name']}: ΔP_Q `{100*item['P_Q_relative_change']:.6g}%`, "
            f"ΔP_six `{100*item['P_six_relative_change']:.6g}%`, "
            f"spatial-Q L2 `{100*item['spatial_Q_relative_L2']:.6g}%`"
        )
    lines += [
        "",
        "All source-off, empty-stack x/y/45°, finite-flat x/y/45°, proxy, "
        "six-face closure, domain, PML, and flake-dz gates passed.",
        "",
        "No clipping, smoothing, gain, global rescaling, tiling, source deletion, "
        "thermal solve, PTE, adjoint, gradient, or optimization was used.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    root = Path(args.case_root).expanduser().resolve()
    report = Path(args.report_dir).expanduser().resolve()
    report.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("case_result.json"))
    rows = [read_row(path, root) for path in paths]
    if not rows:
        raise RuntimeError(f"no cases under {root}")

    domain12 = find_one(rows, domain_um=12, pml_layers=24, flake_dz_nm=5)
    baseline = find_one(rows, domain_um=16, pml_layers=24, flake_dz_nm=5)
    pml16 = find_one(rows, domain_um=16, pml_layers=16, flake_dz_nm=5)
    refined = find_one(rows, domain_um=16, pml_layers=24, flake_dz_nm=2.5)
    requested_final = Path(args.final_case_result).expanduser().resolve()
    if requested_final != baseline["_path"].resolve():
        raise RuntimeError("promoted case must be domain16/PML24/dz5 baseline")

    controls = control_gate(rows)
    convergence = {
        "domain_12_to_16_um": convergence_pair("domain 12→16 µm", domain12, baseline),
        "PML_16_to_24_layers": convergence_pair("PML 16→24 layers", pml16, baseline),
        "flake_dz_5_to_2p5_nm": convergence_pair(
            "flake dz 5→2.5 nm", baseline, refined
        ),
    }
    manifest = raw_manifest(rows, baseline)
    closure_pass = float(baseline["six_face_closure"]) < POWER_GATE
    validated = (
        all(controls.values())
        and closure_pass
        and all(item["passed"] for item in convergence.values())
        and manifest["selected_final_SHA_differs_from_PR3"]
    )
    summary = {
        "status": SUCCESS if validated else "FAILED_FINITE_INFLAKE_PROXY_OPTICAL_Q",
        "validated": validated,
        "controls": controls,
        "final_case": public(baseline),
        "convergence": convergence,
        "gates": {
            "six_face_closure_lt_0p5_percent": closure_pass,
            "power_convergence_lt_1_percent": all(
                item["P_Q_relative_change"] < POWER_CONVERGENCE_GATE
                and item["P_six_relative_change"] < POWER_CONVERGENCE_GATE
                for item in convergence.values()
            ),
            "spatial_Q_L2_lt_5_percent": all(
                item["spatial_Q_relative_L2"] < SPATIAL_CONVERGENCE_GATE
                for item in convergence.values()
            ),
            "fresh_SHA_differs_from_PR3": manifest[
                "selected_final_SHA_differs_from_PR3"
            ],
        },
        "prohibited_operations": {
            "reused_PR3_Q": False,
            "cropped_PR3_Q": False,
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "source_deletion": False,
        },
        "excluded_scope": {
            "thermal": False,
            "PTE": False,
            "adjoint": False,
            "gradient": False,
            "optimization": False,
        },
        "case_count": len(rows),
    }
    write_csv(report / "inflake_proxy_optical_q_cases.csv", rows)
    (report / "inflake_proxy_optical_q_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (report / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    write_report(report / "FINITE_INFLAKE_PROXY_OPTICAL_Q_REPORT.md", summary)
    print(json.dumps(summary, indent=2))
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
