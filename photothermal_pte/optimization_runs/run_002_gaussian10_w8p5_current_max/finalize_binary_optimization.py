#!/usr/bin/env python3
"""Promote Run 002 only after the exact binary solver reevaluation passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


STATUS = "COMPLETED_FULLY_BINARIZED_500NM_CONSTRAINED_OPTIMIZATION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--evaluation-raw", type=Path, required=True)
    parser.add_argument("--projected-source-raw", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--global-iteration", type=int, required=True)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    result_path = args.evaluation_result.expanduser().resolve()
    raw_path = args.evaluation_raw.expanduser().resolve()
    result = json.loads(result_path.read_text())
    if result.get("status") != "VALIDATED_THRESHOLDED_BINARY_GPU_MAXWELL_CUDA_THERMAL" or not result.get("passed"):
        raise RuntimeError("thresholded-binary GPU/CUDA reevaluation did not pass")
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("thresholded-binary raw SHA mismatch")
    exact = result["exact_binary_audit"]
    if not exact["solid_pass"] or not exact["void_pass"]:
        raise RuntimeError("exact 500 nm binary DRC did not pass")
    data = np.load(raw_path)
    rho = np.asarray(data["rho_binary"], float)
    if not np.array_equal(np.unique(rho), np.asarray([0.0, 1.0])):
        raise RuntimeError("promoted final density is not exactly {0,1}")
    q = np.asarray(data["Q_total_W_m3"], float)
    dz = np.diff(np.asarray(data["z_edges_m"], float))
    qxy = np.sum(q * dz[None, None, :], axis=2)
    temperature = np.asarray(data["thermal_temperature_grid_K"], float)
    with np.errstate(all="ignore"):
        txy = np.nanmax(temperature, axis=2)
    extent_design = [-9.3, 9.3, -9.3, 9.3]
    x_edges = np.asarray(data["x_edges_m"], float) * 1.0e6
    y_edges = np.asarray(data["y_edges_m"], float) * 1.0e6
    extent_thermal = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    images = (
        (rho.T, extent_design, "final exact binary design", "gray"),
        (qxy.T, extent_thermal, r"depth-integrated $Q$ (W/m$^2$)", "inferno"),
        (txy.T, extent_thermal, "maximum temperature through z (K)", "magma"),
    )
    for ax, (field, extent, title, cmap) in zip(axes, images):
        image = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    final_plot = run / "plots/final_thresholded_binary_solver_validation.png"
    fig.savefig(final_plot, dpi=180)
    plt.close(fig)

    summary_path = run / "results/beta_continuation_summary.json"
    summary = json.loads(summary_path.read_text())
    generated = datetime.now(timezone.utc).isoformat()
    summary.update({
        "status": STATUS,
        "generated_at_utc": generated,
        "final_beta_before_threshold": args.beta,
        "final_global_iteration": args.global_iteration,
        "final_binary_solver_evaluation": result,
        "final_binary_raw_artifact": artifact(raw_path),
        "final_binary_plot": str(final_plot),
    })
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report = run / "results/BETA_CONTINUATION_REPORT.md"
    report.write_text(
        "# Run 002 fully binarized constrained optimization\n\n"
        f"Status: `{STATUS}`\n\n"
        f"Final continuation beta before exact thresholding: `{args.beta:g}`. "
        f"Final accepted global iteration: `{args.global_iteration}`.\n\n"
        f"Thresholded-binary evaluated FOM: `{result['objective_A_per_incident_W']:.12e} A/W`.\n\n"
        f"Exact solid violations: `{exact['solid_bad_cell_count']}`; exact void violations: "
        f"`{exact['void_bad_cell_count']}`. The final density contains exactly `0` and `1`.\n\n"
        "The final candidate was rerun with GPU Maxwell and CUDA thermal/PTE solvers. It was "
        "not promoted from an offline threshold alone, and no posthoc density repair, optical "
        "gain, Q clipping, smoothing, or rescaling was used.\n"
    )
    status_path = run / "STATUS.json"
    status = json.loads(status_path.read_text())
    status.update({
        "status": STATUS,
        "last_updated_utc": generated,
        "optimization_started": True,
        "message": "Fully binary exact-500-nm solid/void design passed final GPU Maxwell and CUDA thermal/PTE reevaluation.",
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    config_path = run / "run_config.json"
    config_text = config_path.read_text()
    config_text, count = re.subn(
        r'("driver_status"\s*:\s*")[^"]+("\s*)',
        rf'\g<1>{STATUS}\g<2>',
        config_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("run_config driver status replacement failed")
    config_path.write_text(config_text)
    manifest_path = run / "manifests/RAW_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["beta_continuation"]["status"] = STATUS
    manifest["beta_continuation"]["thresholded_binary_final"] = {
        "evaluation_result": artifact(result_path),
        "evaluation_NPZ": artifact(raw_path),
        "projected_source_NPZ": artifact(args.projected_source_raw.expanduser().resolve()),
    }
    manifest["current_promoted_status"] = STATUS
    manifest["current_promoted_at_utc"] = generated
    manifest["active_work_status"] = STATUS
    manifest["active_work_updated_at_utc"] = generated
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": STATUS,
        "objective_A_per_incident_W": result["objective_A_per_incident_W"],
        "exact_binary_audit": exact,
        "plot": str(final_plot),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
