#!/usr/bin/env python3
"""Summarize the failed strict-time full-SiO2 FDTD diagnostic offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRACE = re.compile(
    r"(?P<progress>[0-9.]+)% complete\. Elapsed simulation time: "
    r"(?P<time>[0-9.eE+-]+) secs\..*?Auto Shutoff: "
    r"(?P<shutoff>[0-9.eE+-]+)"
)
COMPLETED_TIME = re.compile(
    r"Completed [0-9]+ iterations, or (?P<time>[0-9.eE+-]+)s "
    r"of Simulation Time"
)
WALL_TIME = re.compile(r"Overall wall time measurements in seconds: (?P<time>[0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    trace = [
        {
            "progress_percent": float(match.group("progress")),
            "simulation_time_s": float(match.group("time")),
            "auto_shutoff": float(match.group("shutoff")),
        }
        for match in TRACE.finditer(text)
    ]
    if not trace:
        raise RuntimeError(f"no auto-shutoff trace found in {path}")
    completed = COMPLETED_TIME.search(text)
    wall = WALL_TIME.search(text)
    return {
        "path": str(path.resolve()),
        "trace": trace,
        "completed_simulation_time_s": (
            float(completed.group("time")) if completed else None
        ),
        "wall_time_s": float(wall.group("time")) if wall else None,
        "auto_shutoff_early_termination": (
            "Early termination of simulation, the autoshutoff criteria are satisfied."
            in text
        ),
        "electromagnetic_field_divergence": (
            "the electromagnetic fields are diverging" in text
        ),
        "misleading_success_footer_after_divergence": (
            "the electromagnetic fields are diverging" in text
            and "Simulation completed successfully" in text
        ),
        "minimum_logged_auto_shutoff": min(
            item["auto_shutoff"] for item in trace
        ),
        "maximum_logged_auto_shutoff": max(
            item["auto_shutoff"] for item in trace
        ),
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-case-dir", type=Path, required=True)
    parser.add_argument("--strict-case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "legacy_auto_1e-5": parse_log(
            args.old_case_dir / "finite_2um_optical_q_p0.log"
        ),
        "matched_strict_auto_1e-6": parse_log(
            args.strict_case_dir / "finite_2um_optical_q_p0.log"
        ),
    }
    old = cases["legacy_auto_1e-5"]
    strict = cases["matched_strict_auto_1e-6"]
    old_stop_s = float(old["completed_simulation_time_s"])
    strict_divergence_s = float(strict["completed_simulation_time_s"])
    strict_before_old_stop = [
        item for item in strict["trace"] if item["simulation_time_s"] <= old_stop_s
    ]
    strict_after_old_stop = [
        item for item in strict["trace"] if item["simulation_time_s"] > old_stop_s
    ]
    first_strict_after_old = min(
        strict_after_old_stop,
        key=lambda item: item["simulation_time_s"],
    )
    peak_strict = max(strict["trace"], key=lambda item: item["auto_shutoff"])

    rows: list[dict[str, Any]] = []
    for label, case in cases.items():
        rows.extend({"case": label, **item} for item in case["trace"])
    csv_path = args.output_dir / "full_sio2_strict_time_trace.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    for label, case, color in (
        ("legacy 1e-5 (early stop)", old, "tab:orange"),
        ("matched strict 1e-6 (diverged)", strict, "tab:blue"),
    ):
        times_ps = np.asarray(
            [item["simulation_time_s"] for item in case["trace"]]
        ) * 1.0e12
        values = np.asarray([item["auto_shutoff"] for item in case["trace"]])
        axis.plot(times_ps, values, marker="o", ms=3, lw=1.3, label=label, color=color)
    axis.axhline(1.0e-5, ls="--", color="tab:orange", alpha=0.7, label="1e-5 gate")
    axis.axhline(1.0e-6, ls="--", color="tab:green", alpha=0.7, label="1e-6 gate")
    axis.axvline(
        old_stop_s * 1.0e12,
        ls=":",
        color="black",
        alpha=0.8,
        label="legacy stop",
    )
    axis.axvline(
        strict_divergence_s * 1.0e12,
        ls=":",
        color="tab:red",
        alpha=0.8,
        label="divergence",
    )
    axis.set_yscale("log")
    axis.set_xlabel("FDTD simulation time (ps)")
    axis.set_ylabel("logged auto-shutoff observable")
    axis.set_title("Full-SiO2 E||a temporal diagnostic: early stop precedes divergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    plot_path = args.output_dir / "full_sio2_strict_time_trace.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    old_result = json.loads((args.old_case_dir / "case_result.json").read_text())
    strict_result = json.loads(
        (args.strict_case_dir / "case_result.json").read_text()
    )
    summary = {
        "status": "BLOCKED_LUMERICAL_FDTD_LATE_TIME_DIVERGENCE",
        "new_thermal_PTE_adjoint_or_optimization_run": False,
        "legacy_case": {
            "status": old_result["status"],
            "requested_auto_shutoff": 1.0e-5,
            "stopped_simulation_time_s": old_stop_s,
            "stopped_fraction_of_requested_4ps": old_stop_s / 4.0e-12,
            "final_reported_auto_shutoff": 9.68584e-6,
            "P_Q_W": old_result["run_result"]["P_Q_W"],
            "P_six_W": old_result["run_result"]["P_six_face_W"],
            "closure": old_result["run_result"]["six_face_relative_closure"],
        },
        "strict_matched_case": {
            "status": strict_result["status"],
            "requested_auto_shutoff": 1.0e-6,
            "divergence": strict["electromagnetic_field_divergence"],
            "divergence_simulation_time_s": strict_divergence_s,
            "divergence_fraction_of_requested_4ps": strict_divergence_s / 4.0e-12,
            "wall_time_s": strict["wall_time_s"],
            "minimum_logged_auto_shutoff": strict["minimum_logged_auto_shutoff"],
            "maximum_logged_auto_shutoff": strict["maximum_logged_auto_shutoff"],
            "strict_last_sample_at_or_before_legacy_stop": strict_before_old_stop[-1],
            "strict_first_sample_after_legacy_stop": first_strict_after_old,
            "strict_peak_logged_sample": peak_strict,
            "Pabs_bounds_on_native_solver_mesh_lt_1fm": strict_result[
                "pre_run_contract"
            ]["checks"]["Pabs_bounds_on_native_solver_mesh_lt_1fm"],
            "final_Q_or_flux_published": False,
            "reason": "solver terminated for diverging electromagnetic fields before a converged monitor result",
            "rise_interpretation": (
                "late-time field growth; the log alone does not distinguish "
                "delayed source content from numerical-instability growth"
            ),
        },
        "interpretation": (
            "the legacy 1e-5 run stopped at 0.7367254 ps before the strict "
            "trace rose by more than ten orders of magnitude and diverged at "
            "1.62632 ps; the legacy full-SiO2 Q is therefore an early-stop "
            "diagnostic, not a production Maxwell heat source"
        ),
        "gates": {
            "strict_run_completed_without_divergence": False,
            "strict_auto_shutoff_lt_1e_minus_6": False,
            "final_Q_and_six_face_closure_available": False,
            "thermal_progression_allowed": False,
        },
    }
    summary_path = args.output_dir / "full_sio2_strict_time_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    manifest = {
        "raw_artifacts_committed_to_git": False,
        "artifacts": [
            artifact(args.old_case_dir / "case_result.json"),
            artifact(args.old_case_dir / "finite_2um_optical_q_p0.log"),
            artifact(args.strict_case_dir / "case_result.json"),
            artifact(args.strict_case_dir / "finite_2um_optical_q_p0.log"),
            artifact(args.strict_case_dir / "finite_2um_optical_q.fsp"),
        ],
    }
    manifest_path = args.output_dir / "FULL_SIO2_STRICT_TIME_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = f"""# Full-SiO2 strict-time FDTD diagnostic

Status: `BLOCKED_LUMERICAL_FDTD_LATE_TIME_DIVERGENCE`

The old `1e-5` run stopped at `{old_stop_s * 1e12:.6f} ps`
(`{100 * old_stop_s / 4e-12:.3f}%` of the requested 4 ps window).  The matched
strict run continued through the same apparent minimum, then its logged
auto-shutoff observable rose to `{strict['maximum_logged_auto_shutoff']:.6g}`
and the GPU solver terminated for diverging electromagnetic fields at
`{strict_divergence_s * 1e12:.6f} ps` (`{100 * strict_divergence_s / 4e-12:.3f}%`).

The log alone does not distinguish delayed source content from growth of a
numerically unstable electromagnetic mode.  It does establish that the old
early-stop monitor result did not test this late-time interval.

The strict run's Pabs bounds were on native mesh planes within 1 fm.  It did
not produce a converged final Q or face-flux result, so no strict closure is
reported and no thermal, PTE, adjoint, or optimization run follows.

The old `P_Q={old_result['run_result']['P_Q_W']:.12e} W` and closure
`{100 * old_result['run_result']['six_face_relative_closure']:.6f}%` are preserved as early-stop
diagnostics only.  They are not promoted or rescaled.

The time trace alone does not distinguish delayed source content from the
onset of numerical-instability growth.  It does establish that the old
early-stop artifact never tested the interval in which the strict run failed.

The engine log prints a generic successful-completion footer after explicitly
reporting divergence; the LumAPI run exception and explicit divergence line,
not that footer, determine the fail-closed status.
"""
    (args.output_dir / "FULL_SIO2_STRICT_TIME_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
