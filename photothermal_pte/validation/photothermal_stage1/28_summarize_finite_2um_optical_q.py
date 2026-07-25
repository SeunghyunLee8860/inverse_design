#!/usr/bin/env python3
"""Summarize finite 2 um optical-Q controls and convergence without rerunning FDTD."""

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
from scipy.interpolate import RegularGridInterpolator


POWER_LIMIT = 0.005
CONVERGENCE_LIMIT = 0.01
SPATIAL_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--final-case-result")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_row(path: Path, root: Path) -> dict[str, Any]:
    data = load_json(path)
    run = data.get("run_result", {})
    components = run.get("component_power_W", {})
    hotspot = run.get("Q_hotspot", {})
    acceptance = run.get("acceptance", {})
    return {
        "_path": path,
        "_data": data,
        "case_id": str(path.parent.relative_to(root)),
        "case": data.get("case"),
        "polarization_deg": data.get("polarization_deg"),
        "domain_um": data.get("domain_um"),
        "pml_layers": data.get("pml_layers"),
        "flake_dz_nm": data.get("flake_dz_nm"),
        "source_span_um": data.get("source_span_um"),
        "waist_um": data.get("waist_um"),
        "status": data.get("status"),
        "generation_commit": data.get("generation_commit"),
        "P_Qx_W": components.get("x"),
        "P_Qy_W": components.get("y"),
        "P_Qz_W": components.get("z"),
        "P_Q_W": run.get("P_Q_W"),
        "P_six_face_W": run.get("P_six_face_W"),
        "closure": run.get("six_face_relative_closure"),
        "sigma_abs_m2": run.get("absorption_cross_section_m2"),
        "sigma_over_Ageo": run.get("normalized_absorption_cross_section"),
        "Q_hotspot_W_m3": hotspot.get("Q_W_m3"),
        "source_central_intensity_native_W_m2": run.get(
            "incident_reference", {}
        ).get(
            "central_incident_intensity_W_m2",
            run.get("normalization", {}).get(
                "measured_source_intensity_native_W_m2"
            ),
        ),
        "all_case_acceptance": bool(acceptance)
        and all(bool(value) for value in acceptance.values()),
        "artifact_path": (
            str(path.parent / run["artifact"]) if run.get("artifact") else None
        ),
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def completed_material(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["case"] in ("flat", "fixed-design")
        and row["status"] == "COMPLETED"
        and row["all_case_acceptance"]
    ]


def same(value: Any, target: float) -> bool:
    try:
        return bool(np.isclose(float(value), target))
    except (TypeError, ValueError):
        return False


def select_sweep(
    rows: list[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in completed_material(rows)
        if row["case"] == "fixed-design"
        and same(row["polarization_deg"], 0.0)
    ]
    if kind == "domain":
        candidates = [
            row
            for row in candidates
            if row["pml_layers"] == 24
            and same(row["flake_dz_nm"], 5.0)
            and same(row["waist_um"], 2.0)
            and same(row["source_span_um"], 6.8)
        ]
        key = "domain_um"
    elif kind == "pml":
        candidates = [
            row
            for row in candidates
            if same(row["domain_um"], 8.0)
            and same(row["flake_dz_nm"], 5.0)
            and same(row["waist_um"], 2.0)
            and same(row["source_span_um"], 6.8)
        ]
        key = "pml_layers"
    elif kind == "mesh":
        candidates = [
            row
            for row in candidates
            if same(row["domain_um"], 8.0)
            and row["pml_layers"] == 24
            and same(row["waist_um"], 2.0)
            and same(row["source_span_um"], 6.8)
        ]
        key = "flake_dz_nm"
        return sorted(candidates, key=lambda row: float(row[key]), reverse=True)
    elif kind == "waist":
        candidates = [
            row
            for row in candidates
            if same(row["domain_um"], 8.0)
            and row["pml_layers"] == 24
            and same(row["flake_dz_nm"], 5.0)
            and same(row["source_span_um"], 6.8)
        ]
        key = "waist_um"
    else:
        raise ValueError(kind)
    return sorted(candidates, key=lambda row: float(row[key]))


def relative_changes(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    changes = []
    for previous, current in zip(rows, rows[1:]):
        a = float(previous[key])
        b = float(current[key])
        changes.append(
            {
                "from": previous["case_id"],
                "to": current["case_id"],
                "relative_change": abs(b - a)
                / max(abs(b), np.finfo(float).tiny),
            }
        )
    return changes


def spatial_change(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any] | None:
    if not previous["artifact_path"] or not current["artifact_path"]:
        return None
    old_path = Path(previous["artifact_path"])
    new_path = Path(current["artifact_path"])
    if not old_path.is_file() or not new_path.is_file():
        return None
    with np.load(old_path) as old, np.load(new_path) as new:
        ox, oy, oz = (np.asarray(old[key], float) for key in ("x_m", "y_m", "z_m"))
        nx, ny, nz = (np.asarray(new[key], float) for key in ("x_m", "y_m", "z_m"))
        old_q = np.asarray(old["Q_on_W_m3"], float)
        new_q = np.asarray(new["Q_on_W_m3"], float)
        mask_x = (ox >= -1e-6) & (ox <= 1e-6)
        mask_y = (oy >= -1e-6) & (oy <= 1e-6)
        mask_z = (oz >= -1e-7) & (oz <= 0.0)
        x, y, z = ox[mask_x], oy[mask_y], oz[mask_z]
        points = np.stack(
            np.meshgrid(x, y, z, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        interpolator = RegularGridInterpolator(
            (nx, ny, nz),
            new_q,
            bounds_error=False,
            fill_value=0.0,
        )
        new_on_old = interpolator(points).reshape(x.size, y.size, z.size)
        old_exact = old_q[np.ix_(mask_x, mask_y, mask_z)]
        difference = new_on_old - old_exact
        return {
            "from": previous["case_id"],
            "to": current["case_id"],
            "relative_L2": float(np.linalg.norm(difference))
            / max(float(np.linalg.norm(new_on_old)), np.finfo(float).tiny),
            "relative_L1": float(np.sum(np.abs(difference)))
            / max(float(np.sum(np.abs(new_on_old))), np.finfo(float).tiny),
            "comparison_shape": list(old_exact.shape),
        }


def summarize_sweep(
    rows: list[dict[str, Any]], kind: str
) -> dict[str, Any]:
    selected = select_sweep(rows, kind)
    scalar = {
        key: relative_changes(selected, key)
        for key in (
            "P_Q_W",
            "P_six_face_W",
            "sigma_abs_m2",
            "Q_hotspot_W_m3",
        )
        if all(row.get(key) is not None for row in selected)
    }
    spatial = [
        result
        for result in (
            spatial_change(previous, current)
            for previous, current in zip(selected, selected[1:])
        )
        if result is not None
    ]
    last_scalar = [
        values[-1]["relative_change"]
        for values in scalar.values()
        if values
    ]
    return {
        "kind": kind,
        "cases": [public_row(row) for row in selected],
        "scalar_changes": scalar,
        "spatial_changes": spatial,
        "scalar_converged_lt_1_percent": bool(last_scalar)
        and max(last_scalar) < CONVERGENCE_LIMIT,
        "spatial_converged_lt_5_percent": bool(spatial)
        and max(
            spatial[-1]["relative_L1"], spatial[-1]["relative_L2"]
        )
        < SPATIAL_LIMIT,
    }


def plot_sweep(
    output: Path,
    sweep: dict[str, Any],
    x_key: str,
    xlabel: str,
) -> None:
    cases = sweep["cases"]
    if not cases:
        return
    x = [float(row[x_key]) for row in cases]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, [row["P_Q_W"] for row in cases], "o-", label="P_Q")
    axes[0].plot(
        x, [row["P_six_face_W"] for row in cases], "s--", label="P_six"
    )
    axes[0].set(xlabel=xlabel, ylabel="power (W)", title=f"{sweep['kind']} power")
    axes[0].legend()
    axes[1].plot(
        x, [100.0 * row["closure"] for row in cases], "o-", label="closure"
    )
    axes[1].axhline(0.5, color="red", ls="--", label="0.5% gate")
    axes[1].set(xlabel=xlabel, ylabel="closure (%)", title="six-face closure")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / f"{sweep['kind']}_convergence.png", dpi=180)
    plt.close(figure)


def raw_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts = []
    seen: set[Path] = set()
    for row in rows:
        directory = row["_path"].parent
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            artifacts.append(
                {
                    "server_path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "generation_command": row["_data"].get(
                        "generation_command"
                    ),
                    "generation_commit": row["_data"].get(
                        "generation_commit"
                    ),
                    "reproduction": (
                        f"checkout {row['_data'].get('generation_commit')} "
                        f"and run: {row['_data'].get('generation_command')}"
                    ),
                }
            )
    return {
        "large_raw_files_committed": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public = [public_row(row) for row in rows]
    keys = list(public[0]) if public else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(public)


def main() -> int:
    args = parse_args()
    root = Path(args.case_root).expanduser().resolve()
    output = Path(args.report_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("case_result.json"))
    if not paths:
        raise RuntimeError(f"no case_result.json found under {root}")
    rows = [case_row(path, root) for path in paths]
    write_cases_csv(output / "finite_2um_optical_q_cases.csv", rows)

    sweeps = {
        kind: summarize_sweep(rows, kind)
        for kind in ("domain", "pml", "mesh", "waist")
    }
    for kind, key, label in (
        ("domain", "domain_um", "lateral domain (µm)"),
        ("pml", "pml_layers", "PML layers"),
        ("mesh", "flake_dz_nm", "flake dz (nm)"),
        ("waist", "waist_um", "Gaussian waist (µm)"),
    ):
        plot_sweep(output, sweeps[kind], key, label)

    controls = {
        "no_source_pass": any(
            row["case"] == "no-source"
            and row["status"] == "COMPLETED"
            and row["all_case_acceptance"]
            for row in rows
        ),
        "empty_stack_x_y_45_pass": all(
            any(
                row["case"] == "empty-stack"
                and same(row["polarization_deg"], angle)
                and row["status"] == "COMPLETED"
                and row["all_case_acceptance"]
                for row in rows
            )
            for angle in (0.0, 45.0, 90.0)
        ),
        "flat_x_y_45_pass": all(
            any(
                row["case"] == "flat"
                and same(row["polarization_deg"], angle)
                and row["status"] == "COMPLETED"
                and row["all_case_acceptance"]
                for row in rows
            )
            for angle in (0.0, 45.0, 90.0)
        ),
        "fixed_x_pass": any(
            row["case"] == "fixed-design"
            and same(row["polarization_deg"], 0.0)
            and row["status"] == "COMPLETED"
            and row["all_case_acceptance"]
            for row in rows
        ),
    }
    final = None
    if args.final_case_result:
        final_path = Path(args.final_case_result).expanduser().resolve()
        final = public_row(case_row(final_path, root))
    convergence_pass = all(
        sweeps[kind]["scalar_converged_lt_1_percent"]
        and sweeps[kind]["spatial_converged_lt_5_percent"]
        for kind in ("domain", "pml", "mesh")
    )
    waist_characterized = len(sweeps["waist"]["cases"]) >= 3
    validated = (
        all(controls.values())
        and convergence_pass
        and waist_characterized
        and final is not None
        and final["all_case_acceptance"]
    )
    summary = {
        "validated": validated,
        "controls": controls,
        "convergence": sweeps,
        "waist_characterized": waist_characterized,
        "final_case": final,
        "case_count": len(rows),
        "large_raw_files_committed": False,
        "limits": {
            "six_face_closure": POWER_LIMIT,
            "successive_scalar_convergence": CONVERGENCE_LIMIT,
            "successive_spatial_L1_L2": SPATIAL_LIMIT,
        },
    }
    (output / "finite_2um_optical_q_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    manifest = raw_manifest(rows)
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
