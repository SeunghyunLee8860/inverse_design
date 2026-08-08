#!/usr/bin/env python3
"""Summarize uniform scalar-vs-imported complex material controls.

This certificate intentionally stops at uniform rho endpoints.  It does not
certify the nonuniform density-to-Yee Jacobian or any adjoint gradient.
"""

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


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def load_case(directory: Path) -> tuple[dict[str, object], np.lib.npyio.NpzFile]:
    result_path = directory / "case_result.json"
    npz_path = directory / "complex_material_control_q.npz"
    result = json.loads(result_path.read_text())
    if not result.get("passed", False):
        raise RuntimeError(f"case did not pass its forward gate: {directory}")
    return result, np.load(npz_path)


def compare_pair(rho: float, scalar_dir: Path, imported_dir: Path) -> dict[str, object]:
    scalar, scalar_q = load_case(scalar_dir)
    imported, imported_q = load_case(imported_dir)
    components: dict[str, object] = {}
    for component in "xyz":
        qs = np.asarray(scalar_q[f"Q{component}_W_m3"], float)
        qi = np.asarray(imported_q[f"Q{component}_W_m3"], float)
        coordinate_exact = all(
            np.array_equal(
                scalar_q[f"Q{component}_{axis}_m"],
                imported_q[f"Q{component}_{axis}_m"],
            )
            for axis in "xyz"
        )
        denominator = float(np.linalg.norm(qs))
        difference_norm = float(np.linalg.norm(qi - qs))
        nrmse = difference_norm / denominator if denominator else (
            0.0 if difference_norm == 0.0 else float("inf")
        )
        components[component] = {
            "coordinate_arrays_exact": coordinate_exact,
            "shape_scalar": list(qs.shape),
            "shape_imported": list(qi.shape),
            "spatial_Q_NRMSE": nrmse,
            "maximum_absolute_Q_difference_W_m3": float(np.max(np.abs(qi - qs))),
            "power_relative_difference": relative_difference(
                float(scalar["Q_component_power_W"][component]),
                float(imported["Q_component_power_W"][component]),
            ),
        }
    epsilon_error = 0.0
    for representation in (scalar, imported):
        requested = np.asarray(representation["material"]["requested_epsilon"], float)
        for component in "xyz":
            readback = np.asarray(
                representation["epsilon_component_readback"][component][
                    "epsilon_interior_median"
                ],
                float,
            )
            epsilon_error = max(epsilon_error, float(np.max(np.abs(readback - requested))))
    rho_zero = rho == 0.0
    endpoint = {
        "rho": rho,
        "scalar_directory": str(scalar_dir.resolve()),
        "imported_directory": str(imported_dir.resolve()),
        "requested_epsilon": scalar["material"]["requested_epsilon"],
        "requested_nk": scalar["material"]["requested_nk"],
        "scalar": {
            key: scalar[key]
            for key in (
                "P_Q_W",
                "P_six_W",
                "six_face_closure_relative",
                "Q_component_power_W",
                "solver_wall_time_s",
            )
        },
        "imported": {
            key: imported[key]
            for key in (
                "P_Q_W",
                "P_six_W",
                "six_face_closure_relative",
                "Q_component_power_W",
                "solver_wall_time_s",
            )
        },
        "P_Q_relative_difference": relative_difference(
            float(scalar["P_Q_W"]), float(imported["P_Q_W"])
        ),
        "P_six_relative_difference": relative_difference(
            float(scalar["P_six_W"]), float(imported["P_six_W"])
        ),
        "epsilon_readback_maximum_absolute_error": epsilon_error,
        "components": components,
        "rho_zero_interpretation": (
            "lossless air endpoint: use exact zero Q and epsilon readback; "
            "a relative absorption/closure comparison is ill-conditioned"
            if rho_zero
            else None
        ),
    }
    endpoint["passed"] = bool(
        epsilon_error < 1.0e-12
        and all(row["coordinate_arrays_exact"] for row in components.values())
        and all(row["spatial_Q_NRMSE"] < 0.005 for row in components.values())
        and endpoint["P_Q_relative_difference"] < 0.005
        and endpoint["P_six_relative_difference"] < 0.005
        and (
            (
                float(scalar["P_Q_W"]) == 0.0
                and float(imported["P_Q_W"]) == 0.0
            )
            if rho_zero
            else (
                float(scalar["six_face_closure_relative"]) < 0.005
                and float(imported["six_face_closure_relative"]) < 0.005
            )
        )
    )
    return endpoint


def make_plot(endpoints: list[dict[str, object]], output: Path) -> None:
    rhos = np.asarray([row["rho"] for row in endpoints], float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    width = 0.035
    for offset, representation, color in (
        (-width / 2, "scalar", "#2878B5"),
        (width / 2, "imported", "#F28E2B"),
    ):
        axes[0].bar(
            rhos + offset,
            [row[representation]["P_Q_W"] * 1e15 for row in endpoints],
            width=width,
            label=representation,
            color=color,
        )
        axes[1].bar(
            rhos + offset,
            [
                np.nan
                if row["rho"] == 0.0
                else row[representation]["six_face_closure_relative"] * 100
                for row in endpoints
            ],
            width=width,
            label=representation,
            color=color,
        )
    axes[0].set(xlabel=r"uniform $\rho$", ylabel=r"$P_Q$ (fW)", title="Absorbed power")
    axes[1].set(xlabel=r"uniform $\rho$", ylabel="closure (%)", title="Matched-volume closure")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5% gate")
    axes[1].annotate(
        "N/A\n(lossless)",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.18),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    for component, marker in zip("xyz", ("o", "s", "^")):
        axes[2].semilogy(
            rhos,
            [max(row["components"][component]["spatial_Q_NRMSE"], 1e-18) for row in endpoints],
            marker=marker,
            label=rf"$Q_{component}$",
        )
    axes[2].axhline(0.005, color="black", linestyle="--", linewidth=1)
    axes[2].set(
        xlabel=r"uniform $\rho$",
        ylabel="scalar/imported spatial NRMSE",
        title="Component-grid equivalence",
    )
    for axis in axes:
        axis.set_xlim(-0.1, 1.1)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.suptitle("10 µm uniform complex-material controls (GPU FDTD)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    for rho_label in ("0", "05", "1"):
        parser.add_argument(f"--rho{rho_label}-scalar", required=True, type=Path)
        parser.add_argument(f"--rho{rho_label}-imported", required=True, type=Path)
    parser.add_argument("--diagnostic-directory", action="append", default=[], type=Path)
    args = parser.parse_args()
    pairs = [
        (0.0, args.rho0_scalar, args.rho0_imported),
        (0.5, args.rho05_scalar, args.rho05_imported),
        (1.0, args.rho1_scalar, args.rho1_imported),
    ]
    endpoints = [compare_pair(*pair) for pair in pairs]
    passed = all(row["passed"] for row in endpoints)
    summary = {
        "status": (
            "VALIDATED_UNIFORM_COMPLEX_MATERIAL_REPRESENTATION_EQUIVALENCE"
            if passed
            else "FAILED_UNIFORM_COMPLEX_MATERIAL_REPRESENTATION_EQUIVALENCE"
        ),
        "scope": (
            "uniform rho=0,0.5,1 scalar (n,k) versus importnk2 GPU FDTD; "
            "not a nonuniform density-to-Yee Jacobian or AD-FD certificate"
        ),
        "wavelength_m": 10.0e-6,
        "gate": 0.005,
        "Q_clipping_smoothing_gain_or_rescaling": False,
        "endpoints": endpoints,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "complex_material_equivalence_summary.json"
    plot_path = PLOTS / "complex_material_equivalence.png"
    report_path = RESULTS / "COMPLEX_MATERIAL_EQUIVALENCE_REPORT.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    make_plot(endpoints, plot_path)

    rows = []
    for row in endpoints:
        closure_text = (
            "N/A (lossless)"
            if row["rho"] == 0.0
            else f"{max(row['scalar']['six_face_closure_relative'], row['imported']['six_face_closure_relative']) * 100:.5f}"
        )
        rows.append(
            "| {rho:g} | {pq:.3e} | {ps:.3e} | {cl} | {nrmse:.3e} | {passed} |".format(
                rho=row["rho"],
                pq=row["P_Q_relative_difference"],
                ps=row["P_six_relative_difference"],
                cl=closure_text,
                nrmse=max(v["spatial_Q_NRMSE"] for v in row["components"].values()),
                passed=row["passed"],
            )
        )
    report = f"""# Uniform complex-material representation equivalence

Status: `{summary['status']}`

At 10 µm the design law is

```text
epsilon(rho) = 1 + rho * (epsilon_SiO2 - 1)
epsilon_SiO2 = 7.3490019303043495 + 1.9899687286880576 i
```

The scalar `(n,k) Material` and uniform `importnk2` representations were run
with the same calibrated Gaussian source, local mesh, conformal variant 1,
and matched Q/six-face control volume.  No Q clipping, smoothing, gain, or
rescaling was applied.

| rho | rel. P_Q diff | rel. P_six diff | worst closure (%) | worst spatial component NRMSE | pass |
|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

For rho=0 both representations read back exactly epsilon=1+0i and all three
Q components are exactly zero.  Therefore a relative absorption closure is
ill-conditioned; this endpoint is judged from exact-zero loss and epsilon
readback instead.

## Scope boundary

This result validates only uniform rho=0, 0.5, and 1 representation
equivalence.  It does **not** validate nonuniform interpolation onto the
component-specific Yee grids, JVP/VJP transpose behavior, Maxwell adjoint
sources, combined PTE gradients, or optimization.  Those remain fail-closed.
"""
    report_path.write_text(report)

    manifest = json.loads(MANIFEST.read_text())
    successful = []
    for _, scalar_dir, imported_dir in pairs:
        for directory in (scalar_dir, imported_dir):
            successful.append(
                {
                    "directory": str(directory.resolve()),
                    "artifacts": [
                        artifact(path)
                        for path in sorted(directory.iterdir())
                        if path.is_file()
                    ],
                }
            )
    diagnostics = []
    for directory in args.diagnostic_directory:
        if directory.is_dir():
            diagnostics.append(
                {
                    "directory": str(directory.resolve()),
                    "artifacts": [
                        artifact(path)
                        for path in sorted(directory.iterdir())
                        if path.is_file()
                    ],
                }
            )
    manifest["uniform_complex_material_status"] = summary["status"]
    manifest["uniform_complex_material_raw_artifacts"] = successful
    manifest["uniform_complex_material_diagnostic_artifacts"] = diagnostics
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
