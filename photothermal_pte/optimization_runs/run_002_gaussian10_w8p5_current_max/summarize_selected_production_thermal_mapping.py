#!/usr/bin/env python3
"""Publish selected-grid density mapping and conservative 3D Q deposition."""

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
    parser.add_argument("--density-mapping-directory", required=True, type=Path)
    parser.add_argument("--attribution-directory", required=True, type=Path)
    parser.add_argument("--deposition-directory", required=True, type=Path)
    args = parser.parse_args()
    density_path = args.density_mapping_directory.resolve() / "selected_thermal_density_mapping_result.json"
    attribution_path = args.attribution_directory.resolve() / "production_material_q_attribution.json"
    deposition_path = args.deposition_directory.resolve() / "production_thermal_q_deposition_result.json"
    density = json.loads(density_path.read_text())
    attribution = json.loads(attribution_path.read_text())
    deposition = json.loads(deposition_path.read_text())
    if density.get("status") != "VALIDATED_SELECTED_373_NODE_TO_186_THERMAL_CELL_MAPPING":
        raise RuntimeError("selected density mapping did not pass")
    if not attribution.get("passed") or attribution["geometry_m"]["design_xy"] != [-9.3e-6, 9.3e-6]:
        raise RuntimeError("selected material attribution did not pass exact support")
    if not deposition.get("passed") or not np.isclose(
        deposition["thermal_grid"]["design_span_m"], 18.6e-6, rtol=0.0, atol=2e-18
    ):
        raise RuntimeError("selected thermal deposition did not pass exact support")
    if deposition["power_W"]["expected_material_attributed"] != attribution["power_W"]["physical_thermal_source"]:
        raise RuntimeError("attribution/deposition power provenance mismatch")

    status = "VALIDATED_SELECTED_PRODUCTION_THERMAL_MAPPING"
    summary = {
        "status": status,
        "passed": True,
        "scope": "selected 373-node density to 186-cell thermal map plus native-Q material attribution and conservative 3D deposition",
        "density_mapping": density,
        "material_attribution": attribution,
        "thermal_Q_deposition": deposition,
        "thermal_solve": False,
        "adjoint_solve": False,
        "optimization_iterations": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "selected_production_thermal_mapping_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = RESULTS / "SELECTED_PRODUCTION_THERMAL_MAPPING_REPORT.md"
    report_path.write_text(
        f"""# Selected production thermal mapping

Status: `{status}`

The selected optical design uses 373×373 nodal physical-density values on
`[-9.3,9.3] µm` at 50 nm. The explicit thermal core has 186×186 cells over
the same support at 100 nm. Each thermal-cell density is the exact area
average of the bilinear nodal field, with one-dimensional weights
`[1,2,1]/4`; the transpose applies those weights in reverse.

- Constant-preservation error: `{density['constant_preservation_error']:.3e}`.
- Opposite-edge wrap error: `{density['opposite_edge_wrap_error']:.3e}`.
- Bilinear area-integral error: `{density['integral_relative_error']:.3e}`.
- Worst transpose error: `{density['worst_transpose_relative_error']:.3e}`.
- Worst mapping-only FD error: `{density['worst_mapping_FD_relative_error']:.3e}`.

The selected rho=0.5 GPU `Qx,Qy,Qz` artifact was partitioned by literal
native dual-cell/material intersection using the exact ±9.3 µm design
support. No full cut-cell power was forced into TaIrTe4, SiO2, or the design.
The physical thermal-source power is
`{attribution['power_W']['physical_thermal_source']:.12e} W`, or
`{100.0 * attribution['relative']['physical_thermal_source_fraction_of_full_P_Q']:.6f}%`
of full optical `P_Q`; the remaining air/interface and artificial-background
fractions are reported, not relocated.

Conservative deposition onto the 362×362×91 explicit 3D thermal grid has
relative total-power error `{deposition['gates']['total_relative_to_attribution']:.3e}`,
worst component/material error
`{deposition['gates']['worst_relative_to_attribution']:.3e}`, and zero nonzero
source cells outside their own material. There was no clipping, smoothing,
gain, rescaling, nearest-material relocation, Maxwell rerun, thermal solve,
adjoint solve, or optimization iteration.

This checkpoint does not certify a thermal gray-law exponent, selected-grid
combined AD-FD, exact-binary DRC, or optimization.
"""
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    row_names = [row["direction"].replace("_", "\n") for row in density["directions"]]
    axes[0].semilogy(
        row_names,
        [row["transpose_relative_error"] for row in density["directions"]],
        "o-",
        label="transpose",
    )
    axes[0].semilogy(
        row_names,
        [row["mapping_FD_relative_error"] for row in density["directions"]],
        "s-",
        label="mapping FD",
    )
    axes[0].axhline(1e-12, color="k", ls="--", lw=1)
    axes[0].set(title="373-node → 186-cell checks", ylabel="relative error")
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].legend()
    names = ["Si", "bottom_SiO2", "physical_TaIrTe4", "design_effective_SiO2"]
    axes[1].bar(
        ["Si", "bottom\nSiO2", "TaIrTe4", "design\nSiO2"],
        [attribution["power_W"][name] * 1e15 for name in names],
    )
    axes[1].set(title="Material-attributed thermal power", ylabel="power (fW)")
    gates = deposition["gates"]
    labels = ["total", "worst\ncomponent/material"]
    values = [gates["total_relative_to_attribution"], gates["worst_relative_to_attribution"]]
    axes[2].bar(labels, values)
    axes[2].set_yscale("log")
    axes[2].axhline(1e-12, color="k", ls="--", lw=1)
    axes[2].set(title="Conservative Q deposition", ylabel="relative power error")
    fig.suptitle("Run 002 selected-grid thermal mappings")
    plot_path = PLOTS / "selected_production_thermal_mapping.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    manifest = json.loads(MANIFEST.read_text())
    manifest["current_promoted_status"] = status
    manifest["selected_production_thermal_mapping"] = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "density_mapping": {
            "result": record(density_path),
            "NPZ": record(Path(density["artifact"]["path"])),
        },
        "material_attribution": {"result": record(attribution_path)},
        "thermal_Q_deposition": {
            "result": record(deposition_path),
            "NPZ": record(Path(deposition["artifact"]["path"])),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"status": status, "report": str(report_path), "summary": str(summary_path), "plot": str(plot_path)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
