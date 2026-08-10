#!/usr/bin/env python3
"""Publish the exact 3D thermal-grid Q deposition gate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, PLOTS = HERE / "results", HERE / "plots"
MANIFEST = HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", required=True, type=Path)
    raw = parser.parse_args().raw_directory.expanduser().resolve()
    result_path = raw / "production_thermal_q_deposition_result.json"
    result = json.loads(result_path.read_text())
    if not result.get("passed", False):
        raise RuntimeError("3D thermal-Q deposition did not pass")
    artifact = result["artifact"]
    npz_path = Path(artifact["path"])
    if npz_path.stat().st_size != artifact["size_bytes"] or sha256(npz_path) != artifact["sha256"]:
        raise RuntimeError("thermal-Q NPZ provenance mismatch")
    data = np.load(npz_path)
    x_edges, y_edges, z_edges = (np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
    x, y, z = (0.5 * (edge[:-1] + edge[1:]) * 1e6 for edge in (x_edges, y_edges, z_edges))
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    q = np.asarray(data["Q_total_W_m3"], float)
    qxy = np.tensordot(q, dz, axes=(2, 0))
    qz = np.einsum("ijk,i,j->k", q, dx, dy, optimize=True)
    RESULTS.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    summary = RESULTS / "production_thermal_q_deposition_summary.json"
    report = RESULTS / "PRODUCTION_THERMAL_Q_DEPOSITION_REPORT.md"
    plot = PLOTS / "production_thermal_q_deposition.png"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    image = axes[0].pcolormesh(x, y, qxy.T, shading="auto", cmap="inferno")
    fig.colorbar(image, ax=axes[0], label=r"$\int Q dz$ (W/m²)")
    axes[0].set(xlabel="x=b (µm)", ylabel="y=a (µm)", title="Mapped total Q", aspect="equal")
    axes[1].plot(z, qz)
    for value in (-0.385, -0.100, 0.0, 1.0): axes[1].axvline(value, color="0.5", linestyle=":", linewidth=1)
    axes[1].set(xlabel="z (µm)", ylabel=r"$\iint Q dxdy$ (W/m)", title="Mapped depth profile")
    names = ["Si", "bottom_SiO2", "physical_TaIrTe4", "design_effective_SiO2"]
    labels = ["Si", "bottom SiO₂", "TaIrTe₄", "design SiO₂"]
    axes[2].bar(labels, [result["power_W"][name] * 1e15 for name in names]); axes[2].tick_params(axis="x", rotation=25)
    axes[2].set(ylabel="mapped power (fW)", title="Material-resolved thermal RHS")
    fig.suptitle("Run 002 exact material-intersection Q deposition — 64 µm thermal domain")
    fig.savefig(plot, dpi=180); plt.close(fig)
    grid, gate = result["thermal_grid"], result["gates"]
    report.write_text(f"""# Production 3D thermal-grid Q deposition

Status: `{result['status']}`

Native component-Q was deposited on the actual 64×64 µm, 20 µm-Si-depth
thermal grid by exact Cartesian cell/material intersection. No full cut-cell
power was forced into a material, and no nearest-material relocation,
clipping, smoothing, gain, or rescaling was used.

- thermal grid shape: `{grid['shape_xyz']}`
- mapped physical source: `{result['power_W']['total_mapped']:.12e} W`
- expected attributed source: `{result['power_W']['expected_material_attributed']:.12e} W`
- total deposition error: `{gate['total_relative_to_attribution']:.6e}`
- worst component/material conservation error: `{gate['worst_internal_mapping_power_error']:.6e}`
- nonzero cells outside their material: `{gate['nonzero_cells_outside_own_material']}`

Temperature, PTE, adjoint, and optimization were not run in this checkpoint.
""")
    manifest = json.loads(MANIFEST.read_text())
    manifest["production_3d_thermal_q_deposition"] = {"status": result["status"], "raw_directory": str(raw), "artifacts": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(raw.iterdir()) if path.is_file()]}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report)}, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
