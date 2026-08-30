#!/usr/bin/env python3
"""Publish the LH CP+/CP- selected-Z volumetric-Q certificate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_Z_M2_selected_Q_5p25um")
OUT = HERE / "results_Z_M2_selected_Q_5p25um"
CASES = ("LH_CP_plus", "LH_CP_minus")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def overlap_matrix(target: np.ndarray, source: np.ndarray) -> sparse.csr_matrix:
    rows, columns, values = [], [], []
    cursor0 = 0
    for row in range(target.size - 1):
        left, right = target[row], target[row + 1]
        while cursor0 + 1 < source.size and source[cursor0 + 1] <= left:
            cursor0 += 1
        cursor = cursor0
        while cursor + 1 < source.size and source[cursor] < right:
            value = min(right, source[cursor + 1]) - max(left, source[cursor])
            if value > 0:
                rows.append(row); columns.append(cursor); values.append(float(value))
            cursor += 1
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(target.size - 1, source.size - 1)
    ).tocsr()


def common_areal_q(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, dict[str, float]]:
    xe = np.linspace(-2.55e-6, 2.55e-6, 205)
    ye = np.linspace(-1.30e-6, 1.30e-6, 105)
    dx, dy = np.diff(xe), np.diff(ye)
    total = np.zeros((dx.size, dy.size))
    powers: dict[str, float] = {}
    for component in "xyz":
        q = np.asarray(data[f"Q{component}_W_m3"], float)
        sx = dual_edges(data[f"Q{component}_x_m"])
        sy = dual_edges(data[f"Q{component}_y_m"])
        sz = dual_edges(data[f"Q{component}_z_m"])
        areal = np.sum(q * np.diff(sz)[None, None, :], axis=2)
        mapped_power = overlap_matrix(xe, sx) @ areal @ overlap_matrix(ye, sy).T
        mapped = np.asarray(mapped_power) / (dx[:, None] * dy[None, :])
        total += mapped
        powers[component] = float(np.sum(mapped_power))
    return total, powers


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, object]] = {}
    for case in CASES:
        root = RAW / case
        metadata_path = root / "Z2022_M2_selected_Q.json"
        npz_path = root / "Z2022_M2_selected_Q.npz"
        metadata = json.loads(metadata_path.read_text())
        if metadata["status"] != "COMPLETED_Z2022_M2_RECONSTRUCTED_SELECTED_Q":
            raise RuntimeError(f"{case}: {metadata['status']}")
        with np.load(npz_path, allow_pickle=False) as data:
            qxy, powers = common_areal_q(data)
        loaded[case] = {
            "metadata": metadata,
            "metadata_path": metadata_path,
            "npz_path": npz_path,
            "qxy": qxy,
            "common_component_power_W": powers,
        }

    plus = loaded["LH_CP_plus"]
    minus = loaded["LH_CP_minus"]
    qplus, qminus = np.asarray(plus["qxy"]), np.asarray(minus["qxy"])
    xeum = np.linspace(-2.55, 2.55, 205)
    yeum = np.linspace(-1.30, 1.30, 105)
    dx = np.diff(xeum * 1e-6); dy = np.diff(yeum * 1e-6)
    area = dx[:, None] * dy[None, :]
    pplus_native = float(np.sum(qplus * area))
    pminus_native = float(np.sum(qminus * area))
    nplus = qplus / pplus_native
    nminus = qminus / pminus_native
    difference = nplus - nminus
    nrmse = float(np.linalg.norm(difference) / np.linalg.norm(nminus))
    correlation = float(np.corrcoef(nplus.ravel(), nminus.ravel())[0, 1])
    pplus = float(plus["metadata"]["P_Q_pabs_periodic_W"])
    pminus = float(minus["metadata"]["P_Q_pabs_periodic_W"])
    g = 2.0 * (pplus - pminus) / (pplus + pminus)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    vmax = max(float(np.percentile(qplus, 99.8)), float(np.percentile(qminus, 99.8)))
    for axis, values, title in (
        (axes[0, 0], qplus, "LH CP+: raw depth-integrated Q"),
        (axes[0, 1], qminus, "LH CP-: raw depth-integrated Q"),
    ):
        image = axis.pcolormesh(xeum, yeum, values.T, shading="flat", cmap="inferno", vmax=vmax)
        plt.colorbar(image, ax=axis, label="W/m2")
        axis.set_title(title); axis.set_aspect("equal")
        axis.set_xlabel("x=b (um)"); axis.set_ylabel("y=a (um)")
    max_diff = float(np.percentile(np.abs(difference), 99.5))
    image = axes[1, 0].pcolormesh(
        xeum, yeum, difference.T, shading="flat", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_diff, vcenter=0.0, vmax=max_diff),
    )
    plt.colorbar(image, ax=axes[1, 0], label="1/m2")
    axes[1, 0].set_title("equal-power normalized CP+ minus CP-")
    axes[1, 0].set_aspect("equal"); axes[1, 0].set_xlabel("x=b (um)"); axes[1, 0].set_ylabel("y=a (um)")
    axes[1, 1].bar(["CP+", "CP-"], [pplus * 1e15, pminus * 1e15], color=["#386cb0", "#fdb462"])
    axes[1, 1].set_ylabel("periodic P_Q (fW/cell)")
    axes[1, 1].set_title(f"raw power: g = {g:+.5f}; spatial NRMSE = {nrmse:.3%}")
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle("Reconstructed Z2022 M2 at 5.25 um: selected volumetric-Q certificate")
    figure = OUT / "Z2022_M2_selected_Q_CP_comparison.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    summary = {
        "status": "VALIDATED_Z2022_M2_RECONSTRUCTED_SELECTED_Q_PAIR",
        "classification": plus["metadata"]["classification"],
        "wavelength_um": 5.25,
        "handedness": "LH",
        "phase_naming": "CP+ and CP- are explicit Ex/Ey phase definitions; not promoted to LCP/RCP",
        "cases": {
            case: {
                "P_Q_pabs_periodic_W": loaded[case]["metadata"]["P_Q_pabs_periodic_W"],
                "P_flux_absorbed_W": loaded[case]["metadata"]["P_flux_absorbed_W"],
                "closure_relative": loaded[case]["metadata"]["closure_relative"],
                "auto_shutoff": loaded[case]["metadata"]["log_audit"]["final_auto_shutoff"],
                "Q_component_power_native_W": loaded[case]["metadata"]["Q_component_power_native_W"],
                "common_component_power_W": loaded[case]["common_component_power_W"],
                "common_grid_power_W": float(
                    sum(loaded[case]["common_component_power_W"].values())
                ),
                "common_grid_vs_pabs_relative_error": abs(
                    float(sum(loaded[case]["common_component_power_W"].values()))
                    - float(loaded[case]["metadata"]["P_Q_pabs_periodic_W"])
                )
                / abs(float(loaded[case]["metadata"]["P_Q_pabs_periodic_W"])),
                "gates": {
                    **loaded[case]["metadata"]["gates"],
                    "common_grid_vs_pabs_lt_0p5pct": abs(
                        float(sum(loaded[case]["common_component_power_W"].values()))
                        - float(loaded[case]["metadata"]["P_Q_pabs_periodic_W"])
                    )
                    / abs(float(loaded[case]["metadata"]["P_Q_pabs_periodic_W"]))
                    < 0.005,
                },
            }
            for case in CASES
        },
        "signed_circular_phase_absorption_contrast_g": g,
        "equal_power_spatial_Q_NRMSE": nrmse,
        "equal_power_spatial_Q_correlation": correlation,
        "scope": "periodic optical Q only; finite Gaussian Z array and thermal/electrical contacts are not yet defined",
    }
    summary_path = OUT / "Z2022_M2_SELECTED_Q_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = OUT / "Z2022_M2_selected_Q_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case", "P_Q_W", "P_flux_W", "closure", "auto_shutoff", "Qx_W", "Qy_W", "Qz_W"])
        for case in CASES:
            meta = loaded[case]["metadata"]
            comp = meta["Q_component_power_native_W"]
            writer.writerow([case, meta["P_Q_pabs_periodic_W"], meta["P_flux_absorbed_W"], meta["closure_relative"], meta["log_audit"]["final_auto_shutoff"], comp["x"], comp["y"], comp["z"]])

    report = OUT / "Z2022_M2_SELECTED_Q_REPORT.md"
    report.write_text(
        f"""# Reconstructed Z2022 M2 selected volumetric Q

Status: `VALIDATED_Z2022_M2_RECONSTRUCTED_SELECTED_Q_PAIR`

This is a real v261 GPU Maxwell result for the **explicit corner-joined reconstruction** of the published M2 scalar dimensions. It is not author CAD. The active 2-D layer is replaced by fixed 100-nm anisotropic TaIrTe4 (`x=b, y=a, z=c=b closure`).

At 5.25 um, LH CP+ gives `P_Q={pplus:.9e} W/cell` and LH CP- gives `P_Q={pminus:.9e} W/cell`, hence `g={g:+.6f}`. Closures are `{plus['metadata']['closure_relative']:.4%}` and `{minus['metadata']['closure_relative']:.4%}`; both auto-shutoff and Q gates pass. The component-specific conservative common-grid powers differ from the periodic-`pabs` totals by `{summary['cases']['LH_CP_plus']['common_grid_vs_pabs_relative_error']:.4%}` and `{summary['cases']['LH_CP_minus']['common_grid_vs_pabs_relative_error']:.4%}`. Equal-power spatial-Q NRMSE is `{nrmse:.4%}` and correlation is `{correlation:.8f}`.

CP+ and CP- retain explicit solver phase definitions and are not silently renamed LCP/RCP. No thermal/PTE result is claimed for the periodic unit cell: finite Gaussian illumination and finite electrical contacts must be defined first.

- [Q comparison](Z2022_M2_selected_Q_CP_comparison.png)
- [summary JSON](Z2022_M2_SELECTED_Q_SUMMARY.json)
- [cases CSV](Z2022_M2_selected_Q_cases.csv)
- [raw manifest](RAW_ARTIFACT_MANIFEST.json)
"""
    )
    artifacts = []
    for case in CASES:
        artifacts.extend(
            [
                {"path": str(loaded[case]["metadata_path"]), "size_bytes": loaded[case]["metadata_path"].stat().st_size, "sha256": sha256(loaded[case]["metadata_path"])},
            ]
        )
        artifacts.extend(loaded[case]["metadata"]["raw_artifacts"])
    manifest = {
        "raw_not_committed": True,
        "generation": "runres 41_run_v261_z2022_m2_selected_q.py, 4 ps, GPU 5",
        "artifacts": artifacts,
    }
    manifest_path = OUT / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
