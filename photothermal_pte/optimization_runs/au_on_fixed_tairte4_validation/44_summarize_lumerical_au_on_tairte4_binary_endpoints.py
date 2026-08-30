#!/usr/bin/env python3
"""Publish exact-binary Lumerical Au-on-fixed-TaIrTe4 endpoint controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RESULTS = HERE / "results"
ADFD_SUMMARY = RESULTS / "au_on_fixed_tairte4_optical_adfd_summary.json"
SUMMARY = RESULTS / "lumerical_au_on_tairte4_binary_endpoints_summary.json"
CASES = RESULTS / "lumerical_au_on_tairte4_binary_endpoints_cases.csv"
PLOT = RESULTS / "lumerical_au_on_tairte4_binary_endpoints.png"
REPORT = RESULTS / "LUMERICAL_AU_ON_TAIRTE4_BINARY_ENDPOINTS_REPORT.md"
MANIFEST = RESULTS / "LUMERICAL_AU_ON_TAIRTE4_BINARY_ENDPOINTS_MANIFEST.json"
RUNNER = HERE / "43_run_lumerical_au_on_tairte4_binary_endpoint.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complex_relative_error(requested: list[float], realized: list[float]) -> float:
    target = complex(*requested)
    value = complex(*realized)
    return float(abs(value - target) / max(abs(target), 1e-300))


def q_array_audit(npz_path: Path) -> dict[str, object]:
    with np.load(npz_path) as data:
        components = {}
        for component in "xyz":
            array = np.asarray(data[f"Q{component}_W_m3"], float)
            components[component] = {
                "shape": list(array.shape),
                "all_finite": bool(np.all(np.isfinite(array))),
                "negative_cell_count": int(np.count_nonzero(array < 0.0)),
                "minimum_W_m3": float(np.min(array)),
                "maximum_W_m3": float(np.max(array)),
            }
    return {
        "components": components,
        "passed": bool(
            all(
                row["all_finite"] and row["negative_cell_count"] == 0
                for row in components.values()
            )
        ),
    }


def aggregate_partition(case: dict[str, object]) -> dict[str, float]:
    totals = {"TaIrTe4": 0.0, "Au": 0.0, "residual": 0.0}
    for component in "xyz":
        row = case["epsilon_component_readback"][component][
            "geometric_power_partition_W"
        ]
        totals["TaIrTe4"] += float(row["TaIrTe4"])
        totals["Au"] += float(row["Au"])
        totals["residual"] += float(row["unassigned_or_interface_residual"])
    return totals


def raw_npz(case: dict[str, object]) -> Path:
    matches = [
        Path(row["path"])
        for row in case["raw_artifacts"]
        if str(row["path"]).endswith(".npz")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one raw NPZ, found {matches}")
    return matches[0]


def validate_case(case: dict[str, object], endpoint: int) -> dict[str, object]:
    if case["status"] != "COMPLETED_COMPLEX_MATERIAL_FORWARD_CONTROL":
        raise RuntimeError(f"endpoint {endpoint}: {case['status']}")
    if not case.get("passed", False):
        raise RuntimeError(f"endpoint {endpoint} did not pass its forward gate")
    if int(case["material"]["au_endpoint"]) != endpoint:
        raise RuntimeError(f"endpoint label mismatch for {endpoint}")

    gpu = case["endpoint_crosscheck_contract"]["GPU_log_evidence"]
    partition = aggregate_partition(case)
    npz_path = raw_npz(case)
    q_audit = q_array_audit(npz_path)
    material_errors = {}
    for component in "xyz":
        readback = case["epsilon_component_readback"][component]
        material_errors[f"TaIrTe4_{component}"] = complex_relative_error(
            readback["requested_tairte4_epsilon"],
            readback["tairte4_epsilon_interior_median"],
        )
        if endpoint:
            material_errors[f"Au_{component}"] = complex_relative_error(
                readback["requested_au_epsilon"],
                readback["au_epsilon_interior_median"],
            )

    artifact_hashes_match = True
    for row in case["raw_artifacts"]:
        path = Path(row["path"])
        artifact_hashes_match &= bool(
            path.is_file()
            and path.stat().st_size == int(row["size_bytes"])
            and sha256(path) == row["sha256"]
        )
    checks = {
        "six_face_closure_lt_0p5pct": float(case["six_face_closure_relative"]) < 0.005,
        "auto_shutoff_lt_1e_minus_5": float(case["log_audit"]["final_auto_shutoff"]) < 1e-5,
        "GPU_execution_proven": bool(gpu["passed"]),
        "material_readback_lt_0p5pct": max(material_errors.values()) < 0.005,
        "Q_finite_and_nonnegative": bool(q_audit["passed"]),
        "raw_artifact_hashes_match": bool(artifact_hashes_match),
        "no_Q_clipping_smoothing_gain_or_rescaling": not any(
            case["built_source_contract"]["Q_processing"].values()
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "material_relative_errors": material_errors,
        "partition_W": partition,
        "Q_array_audit": q_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--au0", type=Path, required=True)
    parser.add_argument("--au1", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        0: json.loads(args.au0.resolve().read_text(encoding="utf-8")),
        1: json.loads(args.au1.resolve().read_text(encoding="utf-8")),
    }
    validation = {endpoint: validate_case(case, endpoint) for endpoint, case in cases.items()}
    source_match = abs(float(cases[1]["source_power_W"]) - float(cases[0]["source_power_W"])) / abs(
        float(cases[0]["source_power_W"])
    )
    mesh_match = cases[0]["mesh_after_runsetup"] == cases[1]["mesh_after_runsetup"]
    source_contract_match = cases[0]["built_source_contract"]["source"] == cases[1]["built_source_contract"]["source"]
    pair_checks = {
        "source_power_relative_difference_lt_1e_minus_12": source_match < 1e-12,
        "mesh_readback_exact_match": mesh_match,
        "source_contract_exact_match": source_contract_match,
    }
    passed = all(row["passed"] for row in validation.values()) and all(pair_checks.values())
    status = (
        "VALIDATED_LUMERICAL_AU_TAIRTE4_BINARY_ENDPOINTS"
        if passed
        else "FAILED_LUMERICAL_AU_TAIRTE4_BINARY_ENDPOINTS"
    )
    adfd = json.loads(ADFD_SUMMARY.read_text(encoding="utf-8"))

    rows = []
    for endpoint, case in cases.items():
        partition = validation[endpoint]["partition_W"]
        p_q = float(case["P_Q_W"])
        rows.append(
            {
                "au_endpoint": endpoint,
                "source_power_W": float(case["source_power_W"]),
                "P_Q_W": p_q,
                "P_six_W": float(case["P_six_W"]),
                "absorbed_fraction": p_q / float(case["source_power_W"]),
                "six_face_closure_relative": float(case["six_face_closure_relative"]),
                "auto_shutoff": float(case["log_audit"]["final_auto_shutoff"]),
                "solver_wall_time_s": float(case["solver_wall_time_s"]),
                "GPU_index": int(case["endpoint_crosscheck_contract"]["GPU_log_evidence"]["detected_gpu_index"]),
                "P_Qx_W": float(case["Q_component_power_W"]["x"]),
                "P_Qy_W": float(case["Q_component_power_W"]["y"]),
                "P_Qz_W": float(case["Q_component_power_W"]["z"]),
                "P_TaIrTe4_geometric_W": partition["TaIrTe4"],
                "P_Au_geometric_W": partition["Au"],
                "P_interface_residual_W": partition["residual"],
                "interface_residual_fraction_of_PQ": partition["residual"] / p_q,
                "passed": validation[endpoint]["passed"],
            }
        )
    with CASES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": status,
        "scope": "exact-binary Lumerical endpoint cross-check for an Au optical nanostructure on fixed anisotropic TaIrTe4",
        "Au_is_nanostructure_not_electrode": True,
        "cases": {str(key): value for key, value in cases.items()},
        "validation": {str(key): value for key, value in validation.items()},
        "pair_checks": pair_checks,
        "source_power_relative_difference": source_match,
        "endpoint_effect": {
            "P_Q_relative_change_Au1_vs_Au0": (
                float(cases[1]["P_Q_W"]) / float(cases[0]["P_Q_W"]) - 1.0
            ),
            "absorbed_fraction_Au0": rows[0]["absorbed_fraction"],
            "absorbed_fraction_Au1": rows[1]["absorbed_fraction"],
        },
        "differentiable_route": {
            "status": adfd["status"],
            "solver": "FDTDX/JAX causal fixed-grid dispersive Maxwell",
            "max_total_strong_relative_error_finest_step": adfd["results"]["max_total_strong_relative_error_finest_step"],
            "max_total_gradient_l2_normalized_error_finest_step": adfd["results"]["max_total_gradient_l2_normalized_error_finest_step"],
            "cross_solver_absolute_power_equivalence_claimed": False,
            "reason": "The compact FDTDX AD-FD control and the 48-um Lumerical Gaussian endpoint use different geometry and source normalization.",
        },
        "limitations": [
            "This endpoint check does not validate a Lumerical gray/imported-metal or moving-boundary adjoint.",
            "The geometric TaIrTe4/Au partition leaves conformal/interface Yee power as a signed residual; it is reported and not deleted or reassigned.",
            "Thermal contact, electrical collection, PTE current, and Au topology optimization are not run here.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = ["Au absent", "exact Au"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(labels, [row["P_Q_W"] / row["source_power_W"] for row in rows])
    axes[0, 0].set_ylabel("P_Q / incident source power")
    axes[0, 0].set_title("Absorbed fraction (raw; no rescaling)")
    width = 0.24
    positions = np.arange(2)
    for index, component in enumerate("xyz"):
        axes[0, 1].bar(
            positions + (index - 1) * width,
            [row[f"P_Q{component}_W"] * 1e15 for row in rows],
            width,
            label=f"Q{component}",
        )
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].set_ylabel("component power (fW)")
    axes[0, 1].set_title("Native-Yee component absorption")
    axes[0, 1].legend()
    bottoms = np.zeros(2)
    for key, label in (
        ("P_TaIrTe4_geometric_W", "TaIrTe4 geometric"),
        ("P_Au_geometric_W", "Au geometric"),
        ("P_interface_residual_W", "interface/residual"),
    ):
        values = np.asarray([row[key] * 1e15 for row in rows])
        axes[1, 0].bar(labels, values, bottom=bottoms, label=label)
        bottoms += values
    axes[1, 0].set_ylabel("partitioned power (fW)")
    axes[1, 0].set_title("Geometric material partition (nothing reassigned)")
    axes[1, 0].legend()
    axes[1, 1].bar(labels, [row["six_face_closure_relative"] * 100 for row in rows], label="closure")
    axes[1, 1].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("six-face closure (%)")
    axes[1, 1].set_title("Energy closure")
    axes[1, 1].legend()
    fig.suptitle("Lumerical exact-binary Au nanostructure endpoints on fixed TaIrTe4")
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)

    report = f"""# Lumerical exact-binary Au nanostructure endpoints on fixed TaIrTe4

Status: `{status}`

## What was validated

Au is the **designable optical nanoantenna/nanocube material**, not an
electrode.  Two v261 GPU FDTD cases use the same 10 um scalar Gaussian,
`w0=8.5 um`, 48 x 48 um lateral domain, six PML boundaries, fixed anisotropic
TaIrTe4, component-specific native-Yee absorption, and exact scalar material
endpoints.  The only material difference is absence/presence of a 10 x 10 x
0.05 um Au block in direct face contact with the fixed flake.

| endpoint | P_Q (W) | P_six (W) | closure | auto-shutoff | P_Q/source | GPU |
|---|---:|---:|---:|---:|---:|---:|
| Au absent | {rows[0]['P_Q_W']:.12e} | {rows[0]['P_six_W']:.12e} | {rows[0]['six_face_closure_relative']:.6%} | {rows[0]['auto_shutoff']:.3e} | {rows[0]['absorbed_fraction']:.6%} | {rows[0]['GPU_index']} |
| exact Au | {rows[1]['P_Q_W']:.12e} | {rows[1]['P_six_W']:.12e} | {rows[1]['six_face_closure_relative']:.6%} | {rows[1]['auto_shutoff']:.3e} | {rows[1]['absorbed_fraction']:.6%} | {rows[1]['GPU_index']} |

The source powers match to `{source_match:.3e}` relative.  Adding this exact
Au block changes total absorbed power by
`{summary['endpoint_effect']['P_Q_relative_change_Au1_vs_Au0']:.6%}`.  This is
a raw optical consequence of reflection, field redistribution, Au loss, and
changed TaIrTe4 loss; it is not a fitted or equal-power-normalized result.

## Material and numerical gates

- TaIrTe4 axes: Lumerical `x=b`, `y=a`, `z=c=b` repository closure.
- exact Au at 10 um: `epsilon=-4642.23+1674.64i`.
- maximum component material readback error: `{max(max(row['material_relative_errors'].values()) for row in validation.values()):.3e}`.
- all raw `Qx/Qy/Qz` cells are finite and nonnegative.
- source, mesh, source power, GPU log, auto-shutoff, material readback, raw
  artifact hashes, and six-face closure pass fail-closed gates.
- no Q clipping, smoothing, gain, global rescaling, or material-power
  reassignment is used.

The component-grid geometric masks account for TaIrTe4 and Au interiors.  A
conformal/interface residual of `{rows[1]['interface_residual_fraction_of_PQ']:.6%}`
of `P_Q` remains in the Au-present case.  It is reported as a residual rather
than being deleted or forcibly assigned to either material.

## What this resolves—and what it does not

This closes exact-binary Lumerical endpoint stability, fitted material
readback, native-Yee Q extraction, GPU execution, and energy closure for Au
on fixed TaIrTe4.  Independently, the fixed-grid causal FDTDX/JAX route has
status `{adfd['status']}` with finest-step strong directional error
`{adfd['results']['max_total_strong_relative_error_finest_step']:.6%}`.

The two solvers do **not** use the same compact geometry or absolute source
normalization, so this report does not claim cross-solver equality of raw
power.  It also does not rehabilitate the failed Lumerical moving/conformal
metal boundary derivative or gray `importnk2` route.  The promoted
differentiable optical path remains the causal fixed-grid dispersive route;
thermal, electrical/PTE, and production optimization require their own next
gates.

## Reproduction

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  {RUNNER.relative_to(REPOSITORY)} --au-endpoint 0 \\
  --gpu-device 'GPU 4' --output-dir /external/raw/au0

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  {RUNNER.relative_to(REPOSITORY)} --au-endpoint 1 \\
  --gpu-device 'GPU 4' --output-dir /external/raw/au1

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  {Path(__file__).resolve().relative_to(REPOSITORY)} \\
  --au0 /external/raw/au0/case_result.json \\
  --au1 /external/raw/au1/case_result.json
```
"""
    REPORT.write_text(report, encoding="utf-8")

    published = [RUNNER, Path(__file__).resolve(), SUMMARY, CASES, PLOT, REPORT]
    raw_artifacts = []
    for endpoint, case in cases.items():
        for row in case["raw_artifacts"]:
            raw_artifacts.append({"endpoint": endpoint, **row, "committed": False})
    manifest = {
        "status": status,
        "generation_command": (
            f"44_summarize_lumerical_au_on_tairte4_binary_endpoints.py "
            f"--au0 {args.au0.resolve()} --au1 {args.au1.resolve()}"
        ),
        "raw_artifacts": raw_artifacts,
        "published_artifacts": [
            {
                "path": str(path.relative_to(REPOSITORY)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in published
        ],
        "failed_attempts_preserved_external_to_git": [
            "/home/seunghyun/tairte4/raw_au_on_tairte4_endpoint/au0",
            "/home/seunghyun/tairte4/raw_au_on_tairte4_endpoint/au1",
            "/home/seunghyun/tairte4/raw_au_on_tairte4_endpoint/au1_retry",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(status)
    print(REPORT)
    print(SUMMARY)
    print(CASES)
    print(PLOT)
    print(MANIFEST)


if __name__ == "__main__":
    main()
