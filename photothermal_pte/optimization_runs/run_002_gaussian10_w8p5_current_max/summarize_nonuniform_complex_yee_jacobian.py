#!/usr/bin/env python3
"""Publish the nonuniform complex component-Yee Jacobian smoke control."""

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


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", required=True, type=Path)
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    result_path = raw / "component_yee_jacobian_result.json"
    result = json.loads(result_path.read_text())
    if not result.get("passed", False):
        raise RuntimeError("nonuniform complex component-Yee control did not pass")
    coordinates_path = Path(
        result["artifacts"]["coordinates_and_density"]["path"]
    )
    arrays = np.load(coordinates_path)
    rho = np.asarray(arrays["rho"], float)
    x = np.asarray(arrays["x_nodes_m"], float) * 1e6
    y = np.asarray(arrays["y_nodes_m"], float) * 1e6

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    published_json = RESULTS / "nonuniform_complex_yee_jacobian_summary.json"
    report_path = RESULTS / "NONUNIFORM_COMPLEX_YEE_JACOBIAN_REPORT.md"
    plot_path = PLOTS / "nonuniform_complex_yee_jacobian.png"
    published_json.write_text(json.dumps(result, indent=2) + "\n")

    directions = list(result["directions"])
    fd_error = [
        result["directions"][name]["mapping_only_FD_relative_error"]
        for name in directions
    ]
    dot_error = [
        result["directions"][name]["JVP_VJP_dot_relative_error"]
        for name in directions
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    image = axes[0].pcolormesh(x, y, rho.T, shading="auto", cmap="viridis")
    fig.colorbar(image, ax=axes[0], label=r"physical $\rho$")
    axes[0].set(
        xlabel="x (µm)",
        ylabel="y (µm)",
        title="Nonuniform mapping-control density",
        aspect="equal",
    )
    positions = np.arange(len(directions))
    axes[1].semilogy(positions, fd_error, "o-", label="mapping centered FD")
    axes[1].semilogy(positions, dot_error, "s-", label="JVP/VJP dot")
    axes[1].axhline(
        result["gates"]["mapping_only_FD_limit"],
        color="C0",
        linestyle="--",
        linewidth=1,
        label="FD gate",
    )
    axes[1].axhline(
        result["gates"]["dot_limit"],
        color="C1",
        linestyle="--",
        linewidth=1,
        label="dot gate",
    )
    axes[1].set_xticks(positions, [name.replace("_", "\n") for name in directions])
    axes[1].set(ylabel="relative error", title="Mapping and transpose tests")
    axes[1].legend(fontsize=8)
    components = "xyz"
    nnz = [
        result["coordinate_audit"]["components"][component]["J_nnz"]
        for component in components
    ]
    active = [
        result["coordinate_audit"]["components"][component]["active_J_row_count"]
        for component in components
    ]
    width = 0.36
    indices = np.arange(3)
    axes[2].bar(indices - width / 2, nnz, width, label="nonzeros")
    axes[2].bar(indices + width / 2, active, width, label="active Yee rows")
    axes[2].set_xticks(indices, [r"$J_x$", r"$J_y$", r"$J_z$"])
    axes[2].set(ylabel="count", title="Sparse component operators")
    axes[2].legend()
    fig.suptitle("10 µm nonuniform complex density → component-Yee mapping smoke")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    rows = []
    for name in directions:
        row = result["directions"][name]
        rows.append(
            f"| {name} | {row['mapping_only_FD_relative_error']:.6e} | "
            f"{row['JVP_VJP_dot_relative_error']:.6e} |"
        )
    component_rows = []
    for component in components:
        row = result["coordinate_audit"]["components"][component]
        component_rows.append(
            f"| {component} | {row['shape']} | {row['J_nnz']:,} | "
            f"{row['maximum_J_nonzeros_per_Yee_sample']} | "
            f"{row['maximum_field_index_coordinate_mismatch_m']:.6e} |"
        )
    report_path.write_text(
        f"""# Nonuniform complex component-Yee Jacobian smoke

Status: `{result['status']}`

This is a layout-only mapping control for an isolated 10×10×1 µm imported
complex-SiO2 block at 10 µm.  It uses 101×101 physical-density nodes and the
actual v261 component-specific `index_detail` coordinates.  It performed zero
Maxwell solves and no per-pixel solves.

The differentiated material chain is:

```text
rho -> epsilon=1+rho*(epsilon_SiO2-1) -> passive complex sqrt
    -> importnk2 -> index_detail_c -> epsilon_Yee,c=index_c^2
```

with `epsilon_SiO2={result['epsilon_SiO2'][0]:.16g} +
{result['epsilon_SiO2'][1]:.16g} i`.

| direction | mapping centered-FD relative error | JVP/VJP dot relative error |
|:--|--:|--:|
{chr(10).join(rows)}

| component | Yee shape | J nonzeros | max nonzeros/Yee sample | E/index coordinate mismatch (m) |
|:--:|:--|--:|--:|--:|
{chr(10).join(component_rows)}

Worst mapping FD error: `{result['gates']['worst_mapping_only_FD_relative_error']:.6e}`
(gate `{result['gates']['mapping_only_FD_limit']:.1e}`).

Worst transpose error: `{result['gates']['worst_JVP_VJP_dot_relative_error']:.6e}`
(gate `{result['gates']['dot_limit']:.1e}`).

Maximum component coordinate mismatch:
`{result['maximum_coordinate_mismatch_m']:.6e} m`.

## Scope boundary

This validates the complex interpolation and sparse-Jacobian construction
method on the isolated control only.  It is not the final production-geometry
Jacobian, a Maxwell adjoint certificate, a thermal/PTE gradient certificate,
or permission to start optimization.
"""
    )

    manifest = json.loads(MANIFEST.read_text())
    manifest["nonuniform_complex_component_yee_jacobian_smoke"] = {
        "status": result["status"],
        "raw_directory": str(raw),
        "Maxwell_solves": 0,
        "artifacts": [
            artifact(path) for path in sorted(raw.iterdir()) if path.is_file()
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
