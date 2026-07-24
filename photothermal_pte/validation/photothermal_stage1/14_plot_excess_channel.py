#!/usr/bin/env python3
"""Compare the measured DeltaQ residual with Qy and hypothetical false Ez."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def norm(values: np.ndarray) -> TwoSlopeNorm:
    limit = float(np.max(np.abs(values)))
    return TwoSlopeNorm(vmin=-max(limit, 1e-300), vcenter=0, vmax=max(limit, 1e-300))


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def plot(path: Path, x: np.ndarray, y: np.ndarray, panels: list[tuple[str, np.ndarray]], ylabel: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for ax, (title, values) in zip(axes.flat, panels):
        image = ax.pcolormesh(x, y, values.T, shading="auto", cmap="coolwarm", norm=norm(values))
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x [um]")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--disk-fdtd-dir", required=True)
    parser.add_argument("--flat-fdtd-dir", required=True)
    args = parser.parse_args()
    output = Path(args.component_dir).expanduser().resolve()
    disk = np.load(output / "disk_component_absorption_common_grid.npz")
    flat = np.load(output / "flat_component_absorption_common_grid.npz")
    disk_pabs = np.load(Path(args.disk_fdtd_dir).resolve() / "q_on_raw.npz")
    flat_pabs = np.load(Path(args.flat_fdtd_dir).resolve() / "q_on_raw.npz")

    x, y, z = disk["x_m"], disk["y_m"], disk["z_m"]
    source_power = float(disk["source_power_W"])
    delta_q = disk_pabs["Q_on_W_m3"] - flat_pabs["Q_on_W_m3"]
    delta_qx = disk["Qx_W_m3"] - flat["Qx_W_m3"]
    delta_qy = disk["Qy_W_m3"] - flat["Qy_W_m3"]
    residual = delta_q - delta_qx
    false_ez = (
        disk["false_Qz_scalar_eps_x_W_m3"]
        - flat["false_Qz_scalar_eps_x_W_m3"]
    )
    delta_ez2 = disk["Ez2_common"] - flat["Ez2_common"]
    residual_minus_qy = residual - delta_qy
    mask = np.broadcast_to(
        ((z >= -100e-9) & (z <= 0))[None, None, :], delta_q.shape
    )
    integrate = lambda q: float(
        np.trapezoid(
            np.trapezoid(np.trapezoid(q, z, axis=2), y, axis=1),
            x,
            axis=0,
        )
        / source_power
    )
    summary = {
        "integrals": {
            "DeltaA_Q": integrate(delta_q),
            "DeltaA_Qx": integrate(delta_qx),
            "DeltaA_Q_minus_DeltaA_Qx": integrate(residual),
            "DeltaA_Qy": integrate(delta_qy),
            "residual_minus_Qy": integrate(residual_minus_qy),
            "false_DeltaA_Qz_scalar_eps_x": integrate(false_ez),
        },
        "correlations_in_TaIrTe4": {
            "residual_vs_DeltaQy": corr(residual, delta_qy, mask),
            "residual_vs_false_DeltaQz": corr(residual, false_ez, mask),
            "residual_vs_Delta_Ez2": corr(residual, delta_ez2, mask),
        },
        "definition": "residual = DeltaQ_pabs_adv - conservative_DeltaQx",
        "clipping": False,
        "gain": False,
    }
    (output / "disk_excess_channel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    zint = lambda q: np.trapezoid(q, z, axis=2) / source_power
    plot(
        output / "disk_excess_channel_comparison_xy.png",
        x * 1e6,
        y * 1e6,
        [
            ("DeltaQ total", zint(delta_q)),
            ("conservative DeltaQx", zint(delta_qx)),
            ("residual: DeltaQ-DeltaQx", zint(residual)),
            ("conservative DeltaQy", zint(delta_qy)),
            ("hypothetical false DeltaQz", zint(false_ez)),
            ("Delta |Ez|^2", np.trapezoid(delta_ez2, z, axis=2)),
        ],
        "y [um]",
    )
    iy = int(np.argmin(np.abs(y)))
    plot(
        output / "disk_excess_channel_comparison_xz.png",
        x * 1e6,
        z * 1e9,
        [
            ("DeltaQ total", delta_q[:, iy, :] / source_power),
            ("conservative DeltaQx", delta_qx[:, iy, :] / source_power),
            ("residual: DeltaQ-DeltaQx", residual[:, iy, :] / source_power),
            ("conservative DeltaQy", delta_qy[:, iy, :] / source_power),
            ("hypothetical false DeltaQz", false_ez[:, iy, :] / source_power),
            ("Delta |Ez|^2", delta_ez2[:, iy, :]),
        ],
        "z [nm]",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
