#!/usr/bin/env python3
"""Fail-closed orchestration for the isolated photothermal stage-1 pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import config_stage1 as config
from lumerical_api import (
    create_output_directory,
    environment_record,
    jsonable,
    select_installation,
    write_json,
)


STAGES = (
    ("probe", "00_probe_heat_api.py"),
    ("analytic", "01_validate_heat_analytic.py"),
    ("fdtd", "02_export_fdtd_qon.py"),
    ("steady", "03_import_qon_heat_steady.py"),
    ("scaling", "04_validate_heat_scaling.py"),
    ("transient", "05_validate_heat_transient.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    for name, _ in STAGES:
        parser.add_argument(f"--run-{name}", action="store_true")
    parser.add_argument("--pipeline-placeholder", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def config_snapshot() -> dict[str, Any]:
    result = {}
    for name in dir(config):
        if name.isupper():
            result[name] = jsonable(getattr(config, name))
    return result


def run_stage(
    script: Path,
    *,
    version: str,
    output: Path,
    hide: bool,
    placeholder: bool,
    log,
) -> dict[str, Any]:
    command = [
        sys.executable, str(script),
        "--lumerical-version", version,
        "--output-dir", str(output),
        "--hide-gui" if hide else "--no-hide-gui",
    ]
    if placeholder and script.name == "03_import_qon_heat_steady.py":
        command.append("--pipeline-placeholder")
    print(f"\n===== {script.name} =====", file=log, flush=True)
    print("command: " + " ".join(command), file=log, flush=True)
    result = subprocess.run(
        command, cwd=config.REPOSITORY_ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, file=log, flush=True)
    return {
        "script": script.name,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
    }


def report_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    analytic = read_json(output / "analytic_heat" / "analytic_summary.json")
    if analytic and analytic.get("finest"):
        finest = analytic["finest"]
        for metric, threshold in (
            ("relative_error_deltaT_max", "< 0.01"),
            ("normalized_rmse", "< 0.01"),
            ("energy_error", "< 0.01"),
        ):
            rows.append({
                "test": "analytic HEAT", "metric": metric,
                "value": finest.get(metric), "threshold": threshold,
                "pass": finest.get(metric, float("inf")) < 0.01,
                "notes": f"fine Tmax={finest.get('T_max_numeric_K')} K",
            })
    optical = read_json(output / "fdtd_qon" / "fdtd_absorption_summary.json")
    if optical and optical.get("absorption"):
        value = optical["absorption"].get("energy_error")
        rows.append({
            "test": "FDTD Q_on", "metric": "volume-vs-flux energy error",
            "value": value, "threshold": f"< {config.OPTICAL_ENERGY_ERROR_LIMIT}",
            "pass": value is not None and value < config.OPTICAL_ENERGY_ERROR_LIMIT,
            "notes": "installed pabs_adv with x/y periodic correction",
        })
    steady = read_json(output / "heat_steady" / "heat_steady_summary.json")
    if steady and steady.get("import"):
        value = steady["import"].get("import_power_error")
        rows.append({
            "test": "FDTD -> HEAT", "metric": "import power error",
            "value": value, "threshold": f"< {config.HEAT_IMPORT_POWER_ERROR_LIMIT}",
            "pass": value is not None and value < config.HEAT_IMPORT_POWER_ERROR_LIMIT,
            "notes": steady.get("placeholder_label") or "physical properties only",
        })
    scaling = read_json(output / "heat_scaling" / "heat_scaling_summary.json")
    if scaling and scaling.get("scaling"):
        fits = scaling["scaling"].get("fits", {})
        for metric, fit in fits.items():
            rows.append({
                "test": "HEAT scaling", "metric": f"{metric} R2 / slope error",
                "value": f"{fit.get('R2')} / {fit.get('slope_relative_error')}",
                "threshold": f"> {config.HEAT_SCALING_R2_LIMIT} / < {config.HEAT_SCALING_SLOPE_ERROR_LIMIT}",
                "pass": (
                    fit.get("R2", -float("inf")) > config.HEAT_SCALING_R2_LIMIT
                    and fit.get("slope_relative_error", float("inf"))
                    < config.HEAT_SCALING_SLOPE_ERROR_LIMIT
                ),
                "notes": "",
            })
    transient = read_json(output / "heat_transient" / "transient_summary.json")
    if transient:
        rows.append({
            "test": "HEAT transient", "metric": "overall",
            "value": transient.get("status"), "threshold": "all transient criteria",
            "pass": bool(transient.get("validated")),
            "notes": ", ".join(transient.get("missing_rho_cp", [])),
        })
    return rows


def write_report(output: Path, stage_records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    environment = read_json(output / "environment.json") or {}
    lines = [
        "# Photothermal stage-1 validation report",
        "",
        f"- Output: `{output}`",
        f"- Selected Lumerical: `{environment.get('selected_version', 'unknown')}`",
        f"- Lumerical root: `{environment.get('lumerical_root', 'unknown')}`",
        "- Scope: one-way FDTD -> Q_on -> HEAT only; no optimizer, adjoint, PTE current, or electrode weighting field.",
        "",
        "| Test | Metric | Value | Acceptance threshold | Pass/Fail | Notes |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['test']} | {row['metric']} | {row['value']} | "
            f"{row['threshold']} | {'PASS' if row['pass'] else 'FAIL'} | {row['notes']} |"
        )
    lines.extend(["", "## Stage execution", ""])
    for stage in stage_records:
        lines.append(f"- `{stage['script']}`: return code {stage['returncode']}")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- A missing design/Si/SiO2 thermal conductivity stops the physical HEAT run.",
        "- A TaIrTe4 conductivity tensor that does not round-trip through the installed HEAT API stops the run; it is never replaced silently by a scalar.",
        f"- `{config.PLACEHOLDER_LABEL}` artifacts are numerical plumbing only and are not scientific results.",
        "- Lateral adiabatic boundaries describe a periodic validation cell, not a finite-device PTE model.",
    ])
    (output / "stage1_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output = create_output_directory(args.output_dir)
    installation = select_installation(args.lumerical_version)
    environment = environment_record(installation, working_directory=config.REPOSITORY_ROOT)
    environment["runner_arguments"] = vars(args)
    write_json(output / "environment.json", environment)
    write_json(output / "config_snapshot.json", config_snapshot())
    shutil.copy2(config.HERE / "config_stage1.py", output / "config_stage1.py")
    diff = subprocess.run(
        ["git", "diff", "--", "."], cwd=config.REPOSITORY_ROOT,
        text=True, capture_output=True, check=False,
    )
    (output / "repository_diff.patch").write_text(diff.stdout)

    explicitly_selected = any(getattr(args, f"run_{name}") for name, _ in STAGES)
    selected = {
        name for name, _ in STAGES
        if (getattr(args, f"run_{name}") if explicitly_selected else True)
    }
    records: list[dict[str, Any]] = []
    stopped_reason = None
    try:
        with (output / "all_errors.log").open("w") as log:
            for name, filename in STAGES:
                if name not in selected:
                    continue
                if name == "transient":
                    missing = []
                    for material, rho, cp in (
                        ("design", config.DESIGN_RHO_KG_M3, config.DESIGN_CP_J_KG_K),
                        ("TaIrTe4", config.TAIRTE4_RHO_KG_M3, config.TAIRTE4_CP_J_KG_K),
                        ("SiO2", config.SIO2_RHO_KG_M3, config.SIO2_CP_J_KG_K),
                        ("Si", config.SI_RHO_KG_M3, config.SI_CP_J_KG_K),
                    ):
                        if rho is None:
                            missing.append(f"{material}.rho")
                        if cp is None:
                            missing.append(f"{material}.cp")
                    if missing:
                        skipped = {
                            "script": filename, "returncode": 3,
                            "status": "skipped_missing_material_properties",
                            "missing_rho_cp": missing,
                        }
                        records.append(skipped)
                        write_json(
                            output / "heat_transient" / "transient_summary.json",
                            {"status": skipped["status"], "validated": False,
                             "executed": False, "missing_rho_cp": missing},
                        )
                        continue
                record = run_stage(
                    config.HERE / filename, version=installation.version_key,
                    output=output, hide=args.hide_gui,
                    placeholder=args.pipeline_placeholder, log=log,
                )
                records.append(record)
                if record["returncode"] != 0:
                    stopped_reason = f"{filename} returned {record['returncode']}"
                    # The analytic validation is an absolute gate. All later
                    # physical stages are also fail-closed on their prerequisites.
                    break
    except Exception as exc:
        stopped_reason = f"{type(exc).__name__}: {exc}"
        records.append({
            "script": "runner", "returncode": 1,
            "exception": stopped_reason, "traceback": traceback.format_exc(),
        })

    rows = report_rows(output)
    write_report(output, records, rows)
    all_pass = stopped_reason is None and all(
        record.get("returncode") in (0, 3) for record in records
    )
    summary = {
        "status": "passed" if all_pass else "stopped",
        "validated": bool(all_pass and rows and all(row["pass"] for row in rows if row["test"] != "HEAT transient")),
        "selected_version": installation.version_key,
        "output_directory": str(output),
        "stage_records": records,
        "metrics": rows,
        "stopped_reason": stopped_reason,
        "placeholder_mode": args.pipeline_placeholder,
    }
    write_json(output / "stage1_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
