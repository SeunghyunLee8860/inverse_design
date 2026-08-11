#!/usr/bin/env python3
"""Publish the Run-002 thermal/PTE to native-Yee pullback gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
MANIFEST = HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--native-q", type=Path, required=True)
    parser.add_argument("--native-q-sha256", required=True)
    args = parser.parse_args()
    raw_dir = args.raw_directory.expanduser().resolve()
    result_path = raw_dir / "production_thermal_to_native_yee_pullback_result.json"
    result = json.loads(result_path.read_text())
    if not result.get("passed", False):
        raise RuntimeError("pullback gate did not pass")
    artifact = Path(result["raw_artifact"]["path"])
    if artifact.stat().st_size != result["raw_artifact"]["size_bytes"] or sha256(artifact) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("pullback NPZ provenance mismatch")
    native_path = args.native_q.expanduser().resolve()
    if sha256(native_path) != args.native_q_sha256:
        raise RuntimeError("native-Q SHA mismatch")
    pullback = np.load(artifact)
    native = np.load(native_path)

    contributions = {}
    for component in "xyz":
        sensitivity = np.asarray(pullback[f"native_Q{component}_density_sensitivity_A_m3_W"], float)
        q = np.asarray(native[f"Q{component}_W_m3"], float)
        if sensitivity.shape != q.shape:
            raise RuntimeError(f"pullback/native-Q shape mismatch for {component}")
        contributions[component] = np.sum(sensitivity * q, axis=2)

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "production_thermal_to_native_yee_pullback_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path = RESULTS / "PRODUCTION_THERMAL_TO_NATIVE_YEE_PULLBACK_REPORT.md"
    component_rows = "\n".join(
        f"| {component} | {result['component_records'][component]['transpose_dot_error']:.6e} | "
        f"{result['component_records'][component]['actual_Q_objective_contribution_A']:.12e} |"
        for component in "xyz"
    )
    report_path.write_text(
        f"""# Production thermal/PTE to native-Yee pullback

Status: `{result['status']}`

The uniform-rho production thermal adjoint was pulled through the exact
material-intersection deposition used by the forward source. The transpose is
applied as three memory-bounded 1-D overlap contractions; no full 3-D
Kronecker matrix, nearest-material relocation, Q rescaling, or index pairing
is used.

| native Q component | transpose dot error | actual-Q objective contribution (A) |
|---|---:|---:|
{component_rows}

- worst transpose dot error: `{result['worst_transpose_dot_error']:.6e}`;
- forward residual: `{result['forward']['residual']:.6e}`;
- adjoint residual: `{result['adjoint']['residual']:.6e}`;
- thermal energy balance: `{result['energy_balance_error']:.6e}`;
- Cauchy-normalized objective identity error:
  `{result['Cauchy_normalized_objective_identity_error']:.6e}`;
- Cauchy-normalized reciprocity error:
  `{result['Cauchy_normalized_reciprocity_error']:.6e}`.

The raw relative identity is cancellation-dominated because the centered
rho=0.5 PTE value is near zero. It remains in the JSON as a diagnostic and is
not used to rescale the gradient. The stored native absorption weights are now
the spatial weights needed to construct a thermal-weighted Maxwell adjoint
source. No Maxwell solve or optimization was run in this checkpoint.
"""
    )

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
    component_values = []
    for axis, component in zip(axes[:3], "xyz"):
        x = np.asarray(native[f"Q{component}_x_m"], float) * 1e6
        y = np.asarray(native[f"Q{component}_y_m"], float) * 1e6
        value = contributions[component]
        vmax = max(float(np.max(np.abs(value))), np.finfo(float).tiny)
        image = axis.pcolormesh(x, y, value.T, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        fig.colorbar(image, ax=axis, label="signed objective contribution per x-y sample (A)")
        axis.set(xlabel="x=b (um)", ylabel="y=a (um)", title=f"Q{component} pullback contribution", aspect="equal")
        component_values.append(result["component_records"][component]["actual_Q_objective_contribution_A"])
    axes[3].bar(["Qx", "Qy", "Qz"], component_values)
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set(ylabel="actual-Q objective contribution (A)", title="Component sum")
    fig.suptitle("Run 002 exact thermal/PTE -> native component-Yee Q pullback")
    plot_path = PLOTS / "production_thermal_to_native_yee_pullback.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    manifest = json.loads(MANIFEST.read_text())
    manifest["production_thermal_to_native_yee_pullback"] = {
        "status": result["status"],
        "raw_artifacts_committed_to_git": False,
        "raw_directory": str(raw_dir),
        "artifacts": [record(result_path), record(artifact)],
        "native_Q": record(native_path),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "summary": str(summary_path), "report": str(report_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
