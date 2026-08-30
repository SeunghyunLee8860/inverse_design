#!/usr/bin/env python3
"""End-to-end combined PTE directional AD--FD smoke.

For one strongest local direction this runner recomputes, at rho +/- h*d,

    FDTDX Maxwell Q -> native-Yee audit -> conservative material remap
    -> explicit 3-D thermal solve -> Au-aware electrical weighting/current.

The analytic/adjoint directional derivative is the sum of the independently
validated Maxwell-source, direct thermal/contact, and direct
electrical/weighting density gradients.  No optimization is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STAGE49 = HERE / "49_validate_fdtdx_lumerical_binary_endpoints.py"
STAGE63 = HERE / "63_summarize_fdtdx_spatial_q_export.py"
STAGE64 = HERE / "64_validate_fdtdx_material_overlap_thermal_remap.py"
STAGE67 = HERE / "67_validate_explicit_thermal_weighting_fixed_spatial_q_adfd.py"

OPTICAL_STATUS = "VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT"
DIRECT_STATUS = "VALIDATED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD"
SPATIAL_STATUS = "VALIDATED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_ARTIFACT"
REMAP_STATUS = "VALIDATED_FDTDX_SPATIAL_Q_CONSERVATIVE_MATERIAL_OVERLAP_REMAP"
STATUS_PASS = "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_DIRECTIONAL_ADFD"
STATUS_FAIL = "FAILED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_DIRECTIONAL_ADFD"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{tail}"
        )


def _evaluate_perturbation(
    *,
    label: str,
    rho: np.ndarray,
    output: Path,
    raw_root: Path,
    scenario: str,
    cuda_device: int,
    reuse_existing: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    case_output = output / label
    forward_output = case_output / "forward"
    spatial_output = case_output / "spatial_audit"
    remap_output = case_output / "remap_audit"
    for path in (forward_output, spatial_output, remap_output):
        path.mkdir(parents=True, exist_ok=True)
    density_path = raw_root / f"rho_{label}.npz"
    spatial_raw = raw_root / f"native_yee_q_{label}.npz"
    remap_raw = raw_root / f"thermal_q_{label}.npz"
    if density_path.exists():
        with np.load(density_path, allow_pickle=False) as saved_density:
            saved_rho = np.asarray(saved_density["rho"], dtype=np.float64)
        if not np.allclose(saved_rho, rho, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(f"Existing density checkpoint mismatch for {label}")
    else:
        np.savez_compressed(density_path, rho=rho.astype(np.float32))

    generation_json = forward_output / "fdtdx_substrate_spatial_native_yee_q_export.json"
    reused_forward = False
    if reuse_existing and generation_json.exists() and spatial_raw.exists():
        generation = json.loads(generation_json.read_text(encoding="utf-8"))
        expected_raw = generation.get("raw_artifact", {})
        expected_density = generation.get("density_input", {})
        if generation.get("status") != "VALIDATED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_EXPORT":
            raise RuntimeError(f"Existing forward status mismatch for {label}")
        if _sha256(spatial_raw) != expected_raw.get("sha256"):
            raise RuntimeError(f"Existing spatial-Q SHA mismatch for {label}")
        if _sha256(density_path) != expected_density.get("sha256"):
            raise RuntimeError(f"Existing density SHA mismatch for {label}")
        fdtdx_seconds = 0.0
        reused_forward = True
    else:
        command = [
            sys.executable,
            str(STAGE49),
            "--output-dir",
            str(forward_output),
            "--gradient-smoke",
            "--spatial-q-export",
            "--include-substrate",
            "--matched-substrate-interface-grid",
            "--substrate-total-periods",
            "16",
            "--substrate-window-periods",
            "4",
            "--gradient-checkpoints",
            "16",
            "--density-npz",
            str(density_path),
            "--spatial-q-raw-path",
            str(spatial_raw),
        ]
        start = perf_counter()
        _run_logged(command, case_output / "01_fdtdx_forward.log")
        fdtdx_seconds = perf_counter() - start

    _run_logged(
        [
            sys.executable,
            str(STAGE63),
            "--result-json",
            str(generation_json),
            "--raw-npz",
            str(spatial_raw),
            "--output-dir",
            str(spatial_output),
        ],
        case_output / "02_spatial_audit.log",
    )
    spatial_summary_path = spatial_output / "fdtdx_spatial_native_yee_q_summary.json"
    spatial_summary = json.loads(spatial_summary_path.read_text(encoding="utf-8"))
    if spatial_summary.get("status") != SPATIAL_STATUS:
        raise RuntimeError(f"Fail-closed spatial-Q status for {label}")

    _run_logged(
        [
            sys.executable,
            str(STAGE64),
            "--spatial-summary-json",
            str(spatial_summary_path),
            "--raw-spatial-npz",
            str(spatial_raw),
            "--raw-remap-npz",
            str(remap_raw),
            "--output-dir",
            str(remap_output),
        ],
        case_output / "03_remap_audit.log",
    )
    remap_summary_path = remap_output / "fdtdx_material_overlap_remap_summary.json"
    remap_summary = json.loads(remap_summary_path.read_text(encoding="utf-8"))
    if remap_summary.get("status") != REMAP_STATUS:
        raise RuntimeError(f"Fail-closed conservative-remap status for {label}")

    stage67 = _load(STAGE67, f"au_stage70_direct_{label}")
    forward = stage67._load(stage67.STAGE65, f"au_stage70_forward_{label}")
    electrical = stage67._load(stage67.STAGE54, f"au_stage70_electrical_{label}")
    coupled = stage67._load(stage67.STAGE62, f"au_stage70_coupled_{label}")
    topology = stage67._load(
        forward.TOPOLOGY_THERMAL, f"au_stage70_topology_{label}"
    )
    fvm = stage67._load(
        Path(stage67.__file__).parents[2]
        / "validation"
        / "photothermal_stage1"
        / "anisotropic_heat_fvm.py",
        f"au_stage70_fvm_{label}",
    )
    overlap = stage67._load(forward.STAGE64, f"au_stage70_overlap_{label}")
    state = forward._thermal_state(
        rho,
        forward.G_TA_SIO2_SCENARIOS[scenario],
        topology,
        fvm,
    )
    with np.load(remap_raw, allow_pickle=False) as remap:
        _, source_power, mapping = forward._map_thermal_q(remap, state, overlap)
    thermal_start = perf_counter()
    evaluated = stage67._evaluate(
        rho,
        source_power,
        scenario,
        cuda_device,
        need_gradient=False,
        forward=forward,
        electrical=electrical,
        coupled=coupled,
        topology=topology,
        fvm=fvm,
    )
    thermal_seconds = perf_counter() - thermal_start
    case = {
        "label": label,
        "objective_A": evaluated["objective_A"],
        "P_Q_W": float(np.sum(source_power)),
        "Tmax_K": float(np.max(evaluated["temperature"])),
        "thermal_residual": evaluated["thermal_residual"],
        "thermal_energy_balance": evaluated["thermal_energy_balance"],
        "electrical_residual": evaluated["electrical_residual"],
        "electrical_terminal_balance": evaluated["electrical_balance"],
        "mapping": mapping,
        "fdtdx_forward_seconds": fdtdx_seconds,
        "fdtdx_forward_reused_from_SHA_checkpoint": reused_forward,
        "thermal_electrical_seconds": thermal_seconds,
        "density_artifact": {
            "path": str(density_path),
            "bytes": density_path.stat().st_size,
            "sha256": _sha256(density_path),
            "committed_to_git": False,
        },
        "spatial_Q_artifact": spatial_summary["raw_artifact"],
        "thermal_Q_artifact": remap_summary["output_thermal_Q"],
    }
    raw = {
        "temperature": evaluated["temperature"],
        "weighting": evaluated["weighting"],
        "source_power": source_power,
    }
    return case, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-summary-json", required=True, type=Path)
    parser.add_argument("--optical-gradient-npz", required=True, type=Path)
    parser.add_argument("--direct-summary-json", required=True, type=Path)
    parser.add_argument("--direct-gradient-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    parser.add_argument(
        "--scenario", choices=("thermally_grown", "evaporated"), default="thermally_grown"
    )
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only validation requires CUDA_VISIBLE_DEVICES")
    if args.h <= 0.0:
        raise ValueError(args.h)

    output = args.output_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    raw_output = args.raw_output_npz.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    optical_summary = json.loads(
        args.optical_summary_json.resolve().read_text(encoding="utf-8")
    )
    direct_summary = json.loads(
        args.direct_summary_json.resolve().read_text(encoding="utf-8")
    )
    optical_raw_path = args.optical_gradient_npz.expanduser().resolve()
    direct_raw_path = args.direct_gradient_npz.expanduser().resolve()
    if optical_summary.get("status") != OPTICAL_STATUS:
        raise RuntimeError("Fail-closed optical-source gradient status")
    if direct_summary.get("status") != DIRECT_STATUS:
        raise RuntimeError("Fail-closed direct thermal/electrical gradient status")
    if _sha256(optical_raw_path) != optical_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed optical gradient SHA")
    if _sha256(direct_raw_path) != direct_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed direct gradient SHA")
    if optical_summary["spatial_weight"]["scenario"] != args.scenario:
        raise RuntimeError("Optical source-adjoint scenario mismatch")

    with np.load(optical_raw_path, allow_pickle=False) as optical_raw:
        rho = np.asarray(optical_raw["rho"], dtype=np.float64)
        gradient_optical = np.asarray(optical_raw["gradient_A"], dtype=np.float64)
    with np.load(direct_raw_path, allow_pickle=False) as direct_raw:
        direct_rho = np.asarray(direct_raw["rho"], dtype=np.float64)
        gradient_thermal = np.asarray(
            direct_raw[f"gradient_thermal_{args.scenario}_A"], dtype=np.float64
        )
        gradient_electrical = np.asarray(
            direct_raw[f"gradient_electrical_{args.scenario}_A"], dtype=np.float64
        )
    if not np.allclose(rho, direct_rho, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("Baseline density mismatch between derivative checkpoints")
    gradient_direct = gradient_thermal + gradient_electrical
    gradient_total = gradient_optical + gradient_direct
    gradient_norm = float(np.linalg.norm(gradient_total))
    if not np.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise RuntimeError("Invalid combined gradient")
    direction = gradient_total / gradient_norm
    rho_plus = rho + args.h * direction
    rho_minus = rho - args.h * direction
    if np.min(rho_minus) <= 0.0 or np.max(rho_plus) >= 1.0:
        raise RuntimeError("Combined central FD would require clipping")

    plus, plus_raw = _evaluate_perturbation(
        label="plus",
        rho=rho_plus,
        output=output,
        raw_root=raw_root,
        scenario=args.scenario,
        cuda_device=args.cuda_device,
    )
    minus, minus_raw = _evaluate_perturbation(
        label="minus",
        rho=rho_minus,
        output=output,
        raw_root=raw_root,
        scenario=args.scenario,
        cuda_device=args.cuda_device,
    )
    fd = (plus["objective_A"] - minus["objective_A"]) / (2.0 * args.h)
    ad_optical = float(np.sum(gradient_optical * direction))
    ad_thermal = float(np.sum(gradient_thermal * direction))
    ad_electrical = float(np.sum(gradient_electrical * direction))
    ad_total = ad_optical + ad_thermal + ad_electrical
    strong_error = _relative(ad_total, fd)
    normalized_error = abs(ad_total - fd) / gradient_norm
    base_objective = float(direct_summary["scenarios"][args.scenario]["objective_A"])
    midpoint_error = _relative(
        0.5 * (plus["objective_A"] + minus["objective_A"]), base_objective
    )
    worst_residual = max(
        plus["thermal_residual"],
        plus["electrical_residual"],
        minus["thermal_residual"],
        minus["electrical_residual"],
    )
    worst_energy = max(
        plus["thermal_energy_balance"], minus["thermal_energy_balance"]
    )
    worst_terminal = max(
        plus["electrical_terminal_balance"], minus["electrical_terminal_balance"]
    )
    gates = {
        "input_status_and_SHA_chain_validated": True,
        "unclipped_central_FD": True,
        "both_spatial_Q_and_remap_subgates_validated": True,
        "strong_direction_relative_error_lt_1pct": strong_error < 0.01,
        "gradient_l2_normalized_error_lt_1pct": normalized_error < 0.01,
        "central_midpoint_objective_error_lt_0p5pct": midpoint_error < 0.005,
        "linear_residual_lt_1e-8": worst_residual < 1.0e-8,
        "thermal_energy_balance_lt_1pct": worst_energy < 0.01,
        "electrical_terminal_balance_lt_1pct": worst_terminal < 0.01,
        "GPU_FDTDX_and_GPU_linear_solves_no_CPU_fallback": True,
        "no_Q_clipping_smoothing_gain_or_global_rescaling": True,
        "no_gradient_rescaling": True,
    }
    passed = all(gates.values())
    status = STATUS_PASS if passed else STATUS_FAIL

    np.savez_compressed(
        raw_output,
        rho=rho.astype(np.float32),
        direction=direction,
        gradient_optical_A=gradient_optical,
        gradient_thermal_A=gradient_thermal,
        gradient_electrical_A=gradient_electrical,
        gradient_total_A=gradient_total,
        rho_plus=rho_plus.astype(np.float32),
        rho_minus=rho_minus.astype(np.float32),
        temperature_plus_K=plus_raw["temperature"].astype(np.float32),
        temperature_minus_K=minus_raw["temperature"].astype(np.float32),
        weighting_plus=plus_raw["weighting"].astype(np.float32),
        weighting_minus=minus_raw["weighting"].astype(np.float32),
    )
    raw_sha = _sha256(raw_output)
    row = {
        "scenario": args.scenario,
        "direction": "combined_adjoint_aligned",
        "h": args.h,
        "AD_optical_A": ad_optical,
        "AD_thermal_A": ad_thermal,
        "AD_electrical_A": ad_electrical,
        "AD_total_A": ad_total,
        "FD_total_A": fd,
        "strong_relative_error": strong_error,
        "gradient_l2_normalized_error": normalized_error,
        "objective_plus_A": plus["objective_A"],
        "objective_minus_A": minus["objective_A"],
    }
    csv_path = output / "full_combined_pte_directional_adfd.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for axis, image, title in (
        (axes[0, 0], rho, "baseline rho"),
        (axes[0, 1], gradient_optical, "Maxwell-source gradient"),
        (axes[0, 2], gradient_direct, "direct thermal + electrical gradient"),
        (axes[1, 0], gradient_total, "combined gradient"),
    ):
        if "rho" in title:
            plotted = axis.imshow(image.T, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        else:
            scale = max(float(np.max(np.abs(image))), np.finfo(float).tiny)
            plotted = axis.imshow(
                image.T, origin="lower", cmap="coolwarm", vmin=-scale, vmax=scale
            )
        axis.set_title(title)
        axis.set_xlabel("x=b design index")
        axis.set_ylabel("y=a design index")
        fig.colorbar(plotted, ax=axis)
    limit = 1.1 * max(abs(ad_total), abs(fd), 1e-30)
    axes[1, 1].plot([-limit, limit], [-limit, limit], "k--")
    axes[1, 1].scatter([fd], [ad_total], s=80)
    axes[1, 1].set_xlim(-limit, limit)
    axes[1, 1].set_ylim(-limit, limit)
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_xlabel("end-to-end central FD (A)")
    axes[1, 1].set_ylabel("combined AD (A)")
    axes[1, 1].set_title(f"error={100*strong_error:.5f}%")
    axes[1, 2].bar(
        ["optical", "thermal", "electrical", "total", "FD"],
        [ad_optical, ad_thermal, ad_electrical, ad_total, fd],
    )
    axes[1, 2].tick_params(axis="x", rotation=25)
    axes[1, 2].set_ylabel("directional derivative (A)")
    axes[1, 2].set_title("chain-rule decomposition")
    fig.suptitle(status.replace("_", " "), fontsize=12)
    plot_path = output / "full_combined_pte_directional_adfd.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "one strongest-direction end-to-end central AD-FD through FDTDX Maxwell Q, "
            "native-Yee conservative remap, explicit 3-D thermal material/contact, "
            "and Au-aware electrical weighting/current; no optimization"
        ),
        "scenario": args.scenario,
        "gradient_decomposition": {
            "optical_source_norm_A": float(np.linalg.norm(gradient_optical)),
            "thermal_direct_norm_A": float(np.linalg.norm(gradient_thermal)),
            "electrical_direct_norm_A": float(np.linalg.norm(gradient_electrical)),
            "combined_norm_A": gradient_norm,
        },
        "directional_AD_FD": row,
        "baseline_objective_A": base_objective,
        "central_midpoint_objective_relative_error": midpoint_error,
        "plus": plus,
        "minus": minus,
        "worst_linear_residual": worst_residual,
        "worst_thermal_energy_balance": worst_energy,
        "worst_electrical_terminal_balance": worst_terminal,
        "gates": gates,
        "raw_artifact": {
            "path": str(raw_output),
            "bytes": raw_output.stat().st_size,
            "sha256": raw_sha,
            "committed_to_git": False,
        },
        "next_gate": (
            "if validated, add multiple independent directions/steps before latent "
            "filter/projection validation and any optimization"
        ),
    }
    summary_path = output / "full_combined_pte_directional_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Full combined FDTDX--thermal--weighting PTE directional AD--FD

Status: **{status}**

This smoke recomputes the entire forward chain at `rho +/- {args.h} d`: FDTDX
native-Yee Au/TaIrTe4/SiO2 Q, conservative material-overlap remap, explicit
3-D thermal transport/contact, and Au-aware electrical weighting/current.
The analytic derivative is the unscaled sum of Maxwell-source, direct thermal,
and direct electrical/weighting branches.

| quantity | value |
|---|---:|
| optical-source AD contribution | {ad_optical:.12e} A |
| direct thermal AD contribution | {ad_thermal:.12e} A |
| direct electrical/weighting AD contribution | {ad_electrical:.12e} A |
| combined AD | {ad_total:.12e} A |
| end-to-end central FD | {fd:.12e} A |
| strong-direction error | {100*strong_error:.9f}% |
| gradient-L2-normalized error | {100*normalized_error:.9f}% |
| central midpoint objective error | {100*midpoint_error:.9f}% |
| worst linear residual | {worst_residual:.3e} |
| worst thermal energy balance | {100*worst_energy:.9f}% |
| worst terminal balance | {100*worst_terminal:.9f}% |

No Q clipping, smoothing, gain, global rescaling, density clipping, or gradient
rescaling is used. Raw FDTDX/thermal artifacts stay outside Git and are pinned
by SHA in the manifest. A one-direction smoke is not yet a multi-direction or
latent/filter/projection certificate and does not authorize optimization.
"""
    report_path = output / "FULL_COMBINED_PTE_DIRECTIONAL_ADFD_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = [summary_path, csv_path, plot_path, report_path]
    manifest = {
        "status": status,
        "raw_artifact": summary["raw_artifact"],
        "input_SHAs": {
            "optical_gradient": _sha256(optical_raw_path),
            "direct_gradient": _sha256(direct_raw_path),
        },
        "perturbation_artifacts": {
            "plus": {
                "density": plus["density_artifact"],
                "spatial_Q": plus["spatial_Q_artifact"],
                "thermal_Q": plus["thermal_Q_artifact"],
            },
            "minus": {
                "density": minus["density_artifact"],
                "spatial_Q": minus["spatial_Q_artifact"],
                "thermal_Q": minus["thermal_Q_artifact"],
            },
        },
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "row": row, "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
