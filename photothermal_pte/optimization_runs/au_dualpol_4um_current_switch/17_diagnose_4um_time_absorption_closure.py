#!/usr/bin/env python3
"""Separate time-window and discrete-ADE absorption closure at partial factor 8."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    EPS0_F_PER_M,
    electric_yee_dual_volumes,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.historical_checkpoint import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    load_densities,
    sha256 as checkpoint_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
    au_material_fraction,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    PARTIAL_MATERIAL_Z,
    mesh_context,
    variant_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import (
    raw_path,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    DEVICE_CERTIFICATE,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    require_single_visible_gpu,
    sha256,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_time_absorption_closure"
RAW = raw_path("time_absorption_closure")
FACTOR = 8
POLARIZATION = "Eb"
DENSITY_CASE = "eta_0.35"
TIME_CASES = ((24, 4), (32, 4), (40, 4), (40, 8))
GATE = 5.0e-3


def case_name(total_periods: int, window_periods: int) -> str:
    return f"total_{total_periods:03d}_window_{window_periods:03d}"


def load_case_cache(
    total_periods: int,
    window_periods: int,
    runsetup_sha256: str,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
] | None:
    manifest_path = OUT / f"{case_name(total_periods, window_periods)}.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_TIME_ABSORPTION_CASE":
        return None
    if manifest.get("runsetup_sha256") != runsetup_sha256:
        return None
    if manifest.get("time_case") != [total_periods, window_periods]:
        return None
    raw = manifest.get("raw_artifact")
    if not isinstance(raw, dict):
        return None
    artifact_path = Path(str(raw.get("path", "")))
    if not artifact_path.is_file():
        return None
    if sha256(artifact_path) != raw.get("sha256"):
        raise RuntimeError(f"cached time-case artifact hash mismatch: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        expected = {"q_au", "q_tairte4", "volume_au", "volume_tairte4"}
        if set(archive.files) != expected:
            raise RuntimeError(
                f"cached time-case artifact has keys {archive.files}, expected {expected}"
            )
        q = {
            "au": np.asarray(archive["q_au"]),
            "tairte4": np.asarray(archive["q_tairte4"]),
        }
        volumes = {
            "au": np.asarray(archive["volume_au"]),
            "tairte4": np.asarray(archive["volume_tairte4"]),
        }
    if not all(np.all(np.isfinite(value)) for value in (*q.values(), *volumes.values())):
        raise RuntimeError(f"cached time-case artifact is non-finite: {artifact_path}")
    row = manifest.get("row")
    if not isinstance(row, dict):
        raise RuntimeError(f"cached time-case row is invalid: {manifest_path}")
    return row, q, volumes


def save_case_cache(
    total_periods: int,
    window_periods: int,
    runsetup_sha256: str,
    row: dict[str, object],
    q: dict[str, np.ndarray],
    volumes: dict[str, np.ndarray],
) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    stem = case_name(total_periods, window_periods)
    artifact_path = RAW / f"{stem}_{runsetup_sha256[:16]}.npz"
    temporary = artifact_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            q_au=q["au"],
            q_tairte4=q["tairte4"],
            volume_au=volumes["au"],
            volume_tairte4=volumes["tairte4"],
        )
    temporary.replace(artifact_path)
    manifest = {
        "status": "COMPLETE_TIME_ABSORPTION_CASE",
        "runsetup_sha256": runsetup_sha256,
        "time_case": [total_periods, window_periods],
        "row": row,
        "raw_artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256(artifact_path),
            "tracked_by_git": False,
        },
    }
    manifest_path = OUT / f"{stem}.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(float).tiny)


def discrete_susceptibility(
    coefficient_triplet: tuple[float, float, float], omega: float, dt: float
) -> complex:
    c1, c2, c3 = (
        np.float32(value) for value in coefficient_triplet
    )
    theta = np.float32(omega * dt)
    z_minus = np.exp(np.complex64(-1j * theta))
    z_plus = np.exp(np.complex64(1j * theta))
    return complex(np.complex64(c3) / (z_minus - c1 - c2 * z_plus))


def weighted_q_nrmse(
    late: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    volumes: dict[str, np.ndarray],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for material in ("au", "tairte4"):
        volume = volumes[material]
        difference = late[material] - previous[material]
        numerator += float(np.sum(difference**2 * volume))
        denominator += float(np.sum(late[material] ** 2 * volume))
    return math.sqrt(numerator) / max(math.sqrt(denominator), np.finfo(float).tiny)


def run_case(
    rho: np.ndarray, total_periods: int, window_periods: int
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    with mesh_context(FACTOR, PARTIAL_MATERIAL_Z):
        model = optical_model.build_model(
            POLARIZATION,
            total_periods=total_periods,
            window_periods=window_periods,
            include_adjoint_source=False,
        )
    jax, jnp, fdtdx = model["jax"], model["jnp"], model["fdtdx"]
    au_slice = model["slices"]["au_design"]
    au_c3 = float(model["coefficients"]["au"][2])

    def forward(density):
        fraction = au_material_fraction(density)
        c3 = model["fixed_c3"]
        for component in range(3):
            c3 = c3.at[(0, component, *au_slice)].set(
                au_c3 * fraction[:, :, None]
            )
        arrays = (
            model["base"]
            .reset()
            .aset("dispersive_c1", model["fixed_c1"])
            .aset("dispersive_c2", model["fixed_c2"])
            .aset("dispersive_c3", c3)
        )
        return fdtdx.run_fdtd(
            arrays, model["placed"], model["config"], model["key"],
            show_progress=False,
        )[1]

    density = jnp.asarray(rho, dtype=jnp.float32)
    start = time.perf_counter()
    solve = jax.jit(forward).lower(density).compile()
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    output = solve(density)
    jax.block_until_ready(output.detector_states["au_late"]["phasor"])
    forward_s = time.perf_counter() - start

    eta0 = float(fdtdx.constants.eta0)
    prefactor = 0.5 * float(model["omega_rad_s"]) * EPS0_F_PER_M * eta0**2
    fraction = np.asarray(au_material_fraction(rho), dtype=np.float64)
    target_imag = {
        "au": float(model["epsilon"]["au"].imag),
        "tairte4": np.asarray(
            [
                model["epsilon"]["tairte4"][axis].imag
                for axis in ("b", "a", "c")
            ],
            dtype=np.float64,
        )[:, None, None, None],
    }
    discrete_imag = {
        "au": discrete_susceptibility(
            model["coefficients"]["au"],
            float(model["omega_rad_s"]),
            float(model["config"].time_step_duration),
        ).imag,
        "tairte4": np.asarray(
            [
                discrete_susceptibility(
                    model["coefficients"][axis],
                    float(model["omega_rad_s"]),
                    float(model["config"].time_step_duration),
                ).imag
                for axis in ("b", "a", "c")
            ],
            dtype=np.float64,
        )[:, None, None, None],
    }
    volumes = {
        "au": electric_yee_dual_volumes(
            model["grid"], model["slices"]["au_design"]
        ),
        "tairte4": electric_yee_dual_volumes(
            model["grid"], model["slices"]["fixed_tairte4"]
        ),
    }
    q_by_basis: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    powers: dict[str, dict[str, float]] = {}
    for basis, imag in (("target", target_imag), ("discrete_ADE", discrete_imag)):
        q_by_basis[basis] = {}
        powers[basis] = {}
        for window in ("previous", "late"):
            e_au = np.asarray(
                output.detector_states[f"au_{window}"]["phasor"][0, 0]
            )
            e_ta = np.asarray(
                output.detector_states[f"tairte4_{window}"]["phasor"][0, 0]
            )
            q = {
                "au": prefactor * imag["au"] * fraction[None, :, :, None]
                * np.abs(e_au) ** 2,
                "tairte4": prefactor * imag["tairte4"] * np.abs(e_ta) ** 2,
            }
            q_by_basis[basis][window] = q
            powers[basis][window] = float(
                sum(
                    np.sum(q[material] * volumes[material])
                    for material in ("au", "tairte4")
                )
            )
    closed_td = eta0 * float(
        np.mean(
            np.asarray(
                output.detector_states["material_flux_td"]["poynting_flux"]
            )[:, 0]
        )
    )
    closed_phasor = eta0 * float(
        np.asarray(
            model["placed"]["material_flux"].compute_net_flux(
                output.detector_states["material_flux"]
            )
        )[0]
    )
    target_late = powers["target"]["late"]
    discrete_late = powers["discrete_ADE"]["late"]
    metrics = {
        "target_Q_window_power_change_relative": relative(
            target_late, powers["target"]["previous"]
        ),
        "target_Q_window_spatial_NRMSE": weighted_q_nrmse(
            q_by_basis["target"]["late"],
            q_by_basis["target"]["previous"],
            volumes,
        ),
        "target_Q_vs_discrete_ADE_relative": relative(target_late, discrete_late),
        "target_Q_vs_closed_td_relative": relative(target_late, closed_td),
        "discrete_ADE_Q_vs_closed_td_relative": relative(discrete_late, closed_td),
        "target_Q_vs_closed_phasor_relative": relative(target_late, closed_phasor),
        "closed_td_vs_phasor_relative": relative(closed_td, closed_phasor),
    }
    gates = {name: value < GATE for name, value in metrics.items()}
    row: dict[str, object] = {
        "total_periods": total_periods,
        "window_periods": window_periods,
        "time_steps_total": int(model["config"].time_steps_total),
        "dt_s": float(model["config"].time_step_duration),
        "compile_s": compile_s,
        "forward_s": forward_s,
        "P_target_previous_W": powers["target"]["previous"],
        "P_target_late_W": target_late,
        "P_discrete_ADE_late_W": discrete_late,
        "P_closed_td_W": closed_td,
        "P_closed_phasor_W": closed_phasor,
        **metrics,
        "all_0p5pct_gates_pass": all(gates.values()),
        "gates": gates,
        "discrete_susceptibility": {
            "Au": [
                discrete_susceptibility(
                    model["coefficients"]["au"],
                    float(model["omega_rad_s"]),
                    float(model["config"].time_step_duration),
                ).real,
                discrete_imag["au"],
            ],
            "TaIrTe4_imag_bac": discrete_imag["tairte4"].reshape(3).tolist(),
        },
    }
    return row, q_by_basis["target"]["late"], volumes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    density = load_densities()[DENSITY_CASE]
    runsetup = {
        "status": "AUDITED_TIME_ABSORPTION_CLOSURE_RUNSETUP_NOT_SOLVED",
        "scope": "historical worst-closure partial-z geometry reinterpreted with shared-linear Au",
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "device_contract_sha256": sha256(DEVICE_CERTIFICATE),
        "mesh": variant_audit(FACTOR, PARTIAL_MATERIAL_Z),
        "polarization": POLARIZATION,
        "density_case": DENSITY_CASE,
        "density_min_max_mean": [
            float(np.min(density)),
            float(np.max(density)),
            float(np.mean(density)),
        ],
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "sha256": checkpoint_sha256(),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "time_cases": [list(value) for value in TIME_CASES],
        "gate_relative": GATE,
        "au_material_fraction": material_fraction_audit(),
        "incident_normalization": (
            "not required for scale-invariant Q/closed-flux and window comparisons; "
            "no downstream current is evaluated"
        ),
        "closed_surface_phasor_window": (
            "rectangular switch-only window matching the time-domain monitor; "
            "the pinned FDTDX closed-surface phasor detector omits Tukey weights"
        ),
        "raw_arrays": {
            "written_outside_git": not args.audit_only,
            "root": str(RAW.resolve()),
            "purpose": "per-case restart cache for Q-map convergence comparisons",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    runsetup_path = OUT / "TIME_ABSORPTION_CLOSURE_RUNSETUP.json"
    runsetup_path.write_text(
        json.dumps(runsetup, indent=2) + "\n", encoding="utf-8"
    )
    if args.audit_only:
        print(json.dumps(runsetup, indent=2), flush=True)
        return 0
    require_single_visible_gpu()
    runsetup_sha = sha256(runsetup_path)
    rows = []
    stored_q = {}
    comparison_volumes = None
    for total_periods, window_periods in TIME_CASES:
        print(
            f"[time] total={total_periods} window={window_periods}", flush=True
        )
        cached = load_case_cache(total_periods, window_periods, runsetup_sha)
        if cached is None:
            row, q, volumes = run_case(density, total_periods, window_periods)
            save_case_cache(
                total_periods,
                window_periods,
                runsetup_sha,
                row,
                q,
                volumes,
            )
        else:
            print("[time] resuming verified cached case", flush=True)
            row, q, volumes = cached
        rows.append(row)
        stored_q[(total_periods, window_periods)] = q
        comparison_volumes = volumes
        print(json.dumps(row, indent=2), flush=True)
    reference_key = TIME_CASES[-1]
    reference_q = stored_q[reference_key]
    if comparison_volumes is None:
        raise RuntimeError("time diagnostic produced no comparison volumes")
    comparisons = []
    reference_row = rows[-1]
    for row in rows[:-1]:
        key = (int(row["total_periods"]), int(row["window_periods"]))
        comparisons.append(
            {
                "case": list(key),
                "reference": list(reference_key),
                "target_Q_power_change_relative": relative(
                    float(row["P_target_late_W"]),
                    float(reference_row["P_target_late_W"]),
                ),
                "target_Q_spatial_NRMSE": weighted_q_nrmse(
                    stored_q[key],
                    reference_q,
                    comparison_volumes,
                ),
                "closed_td_power_change_relative": relative(
                    float(row["P_closed_td_W"]),
                    float(reference_row["P_closed_td_W"]),
                ),
            }
        )
    status = (
        "DIAGNOSED_TIME_ABSORPTION_CLOSURE_ALL_GATES_PASS"
        if all(bool(row["all_0p5pct_gates_pass"]) for row in rows)
        else "BLOCKED_TIME_OR_ABSORPTION_CLOSURE"
    )
    summary = {
        "status": status,
        "scope": "diagnostic only on unconfirmed device and partial factor-8 z mesh",
        "runsetup": runsetup,
        "cases": rows,
        "comparisons_to_40_period_8_window": comparisons,
        "promotion": {"is_mesh_certificate": False, "is_gradient_certificate": False},
    }
    (OUT / "TIME_ABSORPTION_CLOSURE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = [
        {key: value for key, value in row.items() if not isinstance(value, dict)}
        for row in rows
    ]
    with (OUT / "time_absorption_closure_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# Time-window and absorption-closure diagnostic",
        "",
        f"Status: `{status}`",
        "",
        "Shared-linear Au is evaluated on the historical factor-8 partial-z",
        "eta=0.35 Eb case that had the worst Q/closed-flux closure. No incident",
        "rescaling or downstream current is used.",
        "",
        "| total periods | window periods | Q window power | Q spatial | target/discrete Q | target Q/TD flux | discrete Q/TD flux | TD/phasor flux | all gates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['total_periods']} | {row['window_periods']} | "
            f"{100*row['target_Q_window_power_change_relative']:.4f}% | "
            f"{100*row['target_Q_window_spatial_NRMSE']:.4f}% | "
            f"{100*row['target_Q_vs_discrete_ADE_relative']:.4f}% | "
            f"{100*row['target_Q_vs_closed_td_relative']:.4f}% | "
            f"{100*row['discrete_ADE_Q_vs_closed_td_relative']:.4f}% | "
            f"{100*row['closed_td_vs_phasor_relative']:.4f}% | "
            f"{row['all_0p5pct_gates_pass']} |"
        )
    lines.extend(
        [
            "",
            "This is not a mesh certificate: the device contract is unconfirmed",
            "and only the historical partial material-z factor-8 grid is tested.",
        ]
    )
    (OUT / "TIME_ABSORPTION_CLOSURE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "cases": rows}, indent=2), flush=True)
    return 0 if status.startswith("DIAGNOSED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
