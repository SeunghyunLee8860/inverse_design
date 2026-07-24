#!/usr/bin/env python3
"""Component-resolved absorption from non-interpolated native Yee data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


EPS0 = 8.8541878128e-12
TAIRTE4_Z_MIN_M = -100e-9
TAIRTE4_Z_MAX_M = 0.0
EPS_A_IMAG = 50.848086107787424
EPS_B_IMAG = 9.289194887622557


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("disk", "square", "flat"):
        parser.add_argument(f"--{name}-native-dir", required=True)
        parser.add_argument(f"--{name}-fdtd-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    x = np.asarray(coordinates, float).reshape(-1)
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("quadrature coordinates must be strictly increasing")
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def integrate_xyz(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    return float(
        np.einsum(
            "i,j,k,ijk->",
            trapezoid_weights(x),
            trapezoid_weights(y),
            trapezoid_weights(z),
            np.asarray(values, float),
            optimize=True,
        )
    )


def conservative_remap_axis(
    values: np.ndarray,
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    axis: int,
    periodic: bool,
) -> np.ndarray:
    """Conservatively move nodal density between grids using quadrature mass.

    Each source nodal mass q_i*w_i is linearly deposited on the two adjacent
    target nodes, then divided by target trapezoidal weights. The resulting
    target-grid density preserves the source integral to roundoff.
    """
    source = np.asarray(source_coordinates, float).reshape(-1)
    target = np.asarray(target_coordinates, float).reshape(-1)
    source_weights = trapezoid_weights(source)
    target_weights = trapezoid_weights(target)
    moved = np.moveaxis(np.asarray(values, float), axis, 0)
    if moved.shape[0] != source.size:
        raise ValueError("source coordinate length does not match data axis")

    deposited_mass = np.zeros((target.size, *moved.shape[1:]), float)
    locations = source.copy()
    if periodic:
        period = target[-1] - target[0]
        locations = (locations - target[0]) % period + target[0]
    else:
        locations = np.clip(locations, target[0], target[-1])

    upper = np.searchsorted(target, locations, side="right")
    upper = np.clip(upper, 1, target.size - 1)
    lower = upper - 1
    denominator = target[upper] - target[lower]
    fraction_upper = (locations - target[lower]) / denominator
    mass = moved * source_weights.reshape(
        (source.size,) + (1,) * (moved.ndim - 1)
    )
    for index in range(source.size):
        deposited_mass[lower[index]] += mass[index] * (
            1.0 - fraction_upper[index]
        )
        deposited_mass[upper[index]] += mass[index] * fraction_upper[index]
    density = deposited_mass / target_weights.reshape(
        (target.size,) + (1,) * (moved.ndim - 1)
    )
    return np.moveaxis(density, 0, axis)


def load_case(native_dir: Path, fdtd_dir: Path) -> dict[str, Any]:
    data = np.load(native_dir / "native_yee_fields_and_index.npz")
    summary = json.loads(
        (fdtd_dir / "fdtd_absorption_summary.json").read_text()
    )
    result = {key: np.asarray(data[key]) for key in data.files}
    result["summary"] = summary
    result["native_dir"] = str(native_dir)
    result["fdtd_dir"] = str(fdtd_dir)
    return result


def component_absorption(case: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray(case["x_m"], float)
    y = np.asarray(case["y_m"], float)
    z = np.asarray(case["z_m"], float)
    delta = {
        "x": np.asarray(case["delta_x_m"], float),
        "y": np.asarray(case["delta_y_m"], float),
        "z": np.asarray(case["delta_z_m"], float),
    }
    frequency = float(np.asarray(case["frequency_Hz"]).reshape(-1)[0])
    source_power = float(np.asarray(case["source_power_W"]).reshape(-1)[0])
    omega = 2.0 * np.pi * frequency
    common: dict[str, np.ndarray] = {}
    native: dict[str, np.ndarray] = {}
    native_integrals: dict[str, float] = {}
    mapped_integrals: dict[str, float] = {}
    native_coordinates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for axis_index, component in enumerate("xyz"):
        electric = np.asarray(case[f"E{component}"])
        epsilon = np.asarray(case[f"index_{component}"]) ** 2
        q_native = (
            0.5
            * EPS0
            * omega
            * np.abs(electric) ** 2
            * np.imag(epsilon)
        )
        coordinates = [x, y, z]
        coordinates[axis_index] = coordinates[axis_index] + delta[component]
        native_coordinates[component] = tuple(coordinates)
        q_common = conservative_remap_axis(
            q_native,
            coordinates[axis_index],
            (x, y, z)[axis_index],
            axis=axis_index,
            periodic=component in "xy",
        )
        native[component] = q_native
        common[component] = q_common
        native_integrals[component] = (
            integrate_xyz(q_native, *coordinates) / source_power
        )
        mapped_integrals[component] = (
            integrate_xyz(q_common, x, y, z) / source_power
        )

    ez2_native = np.abs(np.asarray(case["Ez"])) ** 2
    ez_coordinates = native_coordinates["z"]
    native_z = ez_coordinates[2]
    flake_mask = (
        (native_z >= TAIRTE4_Z_MIN_M - 1e-15)
        & (native_z <= TAIRTE4_Z_MAX_M + 1e-15)
    )
    false_ez: dict[str, np.ndarray] = {}
    false_ez_integrals: dict[str, float] = {}
    for label, epsilon_imaginary in (
        ("scalar_eps_x_on_Ez", EPS_A_IMAG),
        ("scalar_eps_y_on_Ez", EPS_B_IMAG),
    ):
        q_native = (
            0.5
            * EPS0
            * omega
            * epsilon_imaginary
            * ez2_native
            * flake_mask[None, None, :]
        )
        q_common = conservative_remap_axis(
            q_native,
            native_z,
            z,
            axis=2,
            periodic=False,
        )
        false_ez[label] = q_common
        false_ez_integrals[label] = (
            integrate_xyz(q_native, *ez_coordinates) / source_power
        )

    ez2_common = conservative_remap_axis(
        ez2_native, native_z, z, axis=2, periodic=False
    )
    ey2_common = conservative_remap_axis(
        np.abs(np.asarray(case["Ey"])) ** 2,
        y + delta["y"],
        y,
        axis=1,
        periodic=True,
    )
    absorption = case["summary"]["absorption"]
    total_native = sum(native_integrals.values())
    total_mapped = sum(mapped_integrals.values())
    local_flux = float(absorption["A_local_flux"])
    return {
        "x": x,
        "y": y,
        "z": z,
        "native_q": native,
        "common_q": common,
        "native_integrals": native_integrals,
        "mapped_integrals": mapped_integrals,
        "total_native": total_native,
        "total_mapped": total_mapped,
        "pabs_adv_internal": float(
            absorption["Pabs_total_normalized_from_lumerical"]
        ),
        "local_flux": local_flux,
        "relative_mismatch": (total_native - local_flux) / local_flux,
        "false_ez": false_ez,
        "false_ez_integrals": false_ez_integrals,
        "ez2_common": ez2_common,
        "ey2_common": ey2_common,
        "source_power": source_power,
    }


def signed_norm(values: np.ndarray) -> TwoSlopeNorm:
    limit = float(np.nanmax(np.abs(values)))
    if limit == 0.0:
        limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = np.asarray(a)[mask].reshape(-1)
    bb = np.asarray(b)[mask].reshape(-1)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def plot_comparison(
    path: Path,
    x: np.ndarray,
    second_axis: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    *,
    second_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.1))
    for ax, (title, values) in zip(axes.flat, panels):
        values = np.asarray(values, float)
        image = ax.pcolormesh(
            x,
            second_axis,
            values.T,
            shading="auto",
            cmap="coolwarm",
            norm=signed_norm(values),
        )
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x [um]")
        ax.set_ylabel(second_label)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def jsonable_case(result: dict[str, Any]) -> dict[str, Any]:
    native = result["native_integrals"]
    mapped = result["mapped_integrals"]
    false = result["false_ez_integrals"]
    return {
        "A_Qx_native": native["x"],
        "A_Qy_native": native["y"],
        "A_Qz_native": native["z"],
        "A_Q_total_native": result["total_native"],
        "A_Qx_conservative_common": mapped["x"],
        "A_Qy_conservative_common": mapped["y"],
        "A_Qz_conservative_common": mapped["z"],
        "A_Q_total_conservative_common": result["total_mapped"],
        "A_Q_pabs_adv_internal": result["pabs_adv_internal"],
        "A_local_flux": result["local_flux"],
        "relative_mismatch": result["relative_mismatch"],
        "native_minus_conservative_common": (
            result["total_native"] - result["total_mapped"]
        ),
        "native_minus_pabs_adv": (
            result["total_native"] - result["pabs_adv_internal"]
        ),
        "false_Ez_scalar_eps_x": false["scalar_eps_x_on_Ez"],
        "false_Ez_scalar_eps_y": false["scalar_eps_y_on_Ez"],
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name in ("disk", "square", "flat"):
        native_dir = Path(getattr(args, f"{name}_native_dir")).resolve()
        fdtd_dir = Path(getattr(args, f"{name}_fdtd_dir")).resolve()
        cases[name] = component_absorption(load_case(native_dir, fdtd_dir))

    for name, result in cases.items():
        np.savez_compressed(
            output / f"{name}_component_absorption_common_grid.npz",
            x_m=result["x"],
            y_m=result["y"],
            z_m=result["z"],
            Qx_W_m3=result["common_q"]["x"],
            Qy_W_m3=result["common_q"]["y"],
            Qz_W_m3=result["common_q"]["z"],
            false_Qz_scalar_eps_x_W_m3=result["false_ez"][
                "scalar_eps_x_on_Ez"
            ],
            false_Qz_scalar_eps_y_W_m3=result["false_ez"][
                "scalar_eps_y_on_Ez"
            ],
            Ez2_common=result["ez2_common"],
            Ey2_common=result["ey2_common"],
            source_power_W=result["source_power"],
        )

    disk = cases["disk"]
    flat = cases["flat"]
    for axis in ("x", "y", "z"):
        if not np.allclose(
            disk[axis], flat[axis], rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(f"disk and flat {axis} grids differ")
    pabs_disk = np.load(
        Path(args.disk_fdtd_dir) / "q_on_raw.npz"
    )
    pabs_flat = np.load(
        Path(args.flat_fdtd_dir) / "q_on_raw.npz"
    )
    delta_pabs = (
        np.asarray(pabs_disk["Q_on_W_m3"])
        - np.asarray(pabs_flat["Q_on_W_m3"])
    )
    delta_components = {
        component: (
            disk["common_q"][component] - flat["common_q"][component]
        )
        for component in "xyz"
    }
    delta_false_ez = (
        disk["false_ez"]["scalar_eps_x_on_Ez"]
        - flat["false_ez"]["scalar_eps_x_on_Ez"]
    )
    delta_ez2 = disk["ez2_common"] - flat["ez2_common"]
    delta_ey2 = disk["ey2_common"] - flat["ey2_common"]
    x, y, z = disk["x"], disk["y"], disk["z"]
    source_power = disk["source_power"]
    flake_mask = (
        (z >= TAIRTE4_Z_MIN_M - 1e-15)
        & (z <= TAIRTE4_Z_MAX_M + 1e-15)
    )[None, None, :]
    full_mask = np.broadcast_to(flake_mask, delta_pabs.shape)
    correlations = {
        "DeltaQ_vs_DeltaQx": correlation(
            delta_pabs, delta_components["x"], full_mask
        ),
        "DeltaQ_vs_DeltaQy": correlation(
            delta_pabs, delta_components["y"], full_mask
        ),
        "DeltaQ_vs_false_DeltaQz_scalar_eps_x": correlation(
            delta_pabs, delta_false_ez, full_mask
        ),
        "DeltaQ_vs_Delta_Ez2": correlation(
            delta_pabs, delta_ez2, full_mask
        ),
        "DeltaQ_vs_Delta_Ey2": correlation(
            delta_pabs, delta_ey2, full_mask
        ),
    }

    z_integral = lambda values: np.trapezoid(
        values, z, axis=2
    ) / source_power
    xy_panels = [
        ("DeltaQ pabs_adv", z_integral(delta_pabs)),
        ("conservative DeltaQx", z_integral(delta_components["x"])),
        ("conservative DeltaQy", z_integral(delta_components["y"])),
        ("hypothetical false DeltaQz (eps_x)", z_integral(delta_false_ez)),
        ("Delta |Ez|^2", np.trapezoid(delta_ez2, z, axis=2)),
        ("Delta |Ey|^2", np.trapezoid(delta_ey2, z, axis=2)),
    ]
    plot_comparison(
        output / "disk_deltaq_component_comparison_xy.png",
        x * 1e6,
        y * 1e6,
        xy_panels,
        second_label="y [um]",
    )
    iy = int(np.argmin(np.abs(y)))
    xz_panels = [
        ("DeltaQ pabs_adv", delta_pabs[:, iy, :] / source_power),
        (
            "conservative DeltaQx",
            delta_components["x"][:, iy, :] / source_power,
        ),
        (
            "conservative DeltaQy",
            delta_components["y"][:, iy, :] / source_power,
        ),
        (
            "hypothetical false DeltaQz (eps_x)",
            delta_false_ez[:, iy, :] / source_power,
        ),
        ("Delta |Ez|^2", delta_ez2[:, iy, :]),
        ("Delta |Ey|^2", delta_ey2[:, iy, :]),
    ]
    plot_comparison(
        output / "disk_deltaq_component_comparison_xz.png",
        x * 1e6,
        z * 1e9,
        xz_panels,
        second_label="z [nm]",
    )

    delta_a_components = {
        component: (
            disk["mapped_integrals"][component]
            - flat["mapped_integrals"][component]
        )
        for component in "xyz"
    }
    disk_excess = disk["total_native"] - disk["local_flux"]
    delta_flux = disk["local_flux"] - flat["local_flux"]
    delta_a_q = disk["total_native"] - flat["total_native"]
    delta_excess = delta_a_q - delta_flux
    report = {
        "method": {
            "raw_getdata_option": 1,
            "fields_interpolated_before_Q": False,
            "epsilon_component_specific": True,
            "formula": (
                "Qi=+0.5*omega*eps0*imag(eps_i)*abs(Ei)^2 "
                "at native Yee positions"
            ),
            "mapping": (
                "quadrature-mass-conservative linear deposition of scalar Qi "
                "onto the common x/y/z grid"
            ),
            "periodic_mapping_axes": ["x", "y"],
            "clipping": False,
            "flux_gain": False,
            "heat_run": False,
        },
        "disk": jsonable_case(disk),
        "square": jsonable_case(cases["square"]),
        "flat": jsonable_case(flat),
        "disk_minus_flat": {
            "DeltaA_Qx": delta_a_components["x"],
            "DeltaA_Qy": delta_a_components["y"],
            "DeltaA_Qz": delta_a_components["z"],
            "DeltaA_Q_total": delta_a_q,
            "DeltaA_flux": delta_flux,
            "excess": delta_excess,
        },
        "disk_false_Ez_comparison": {
            "actual_disk_Q_minus_flux": disk_excess,
            "false_Ez_scalar_eps_x": disk["false_ez_integrals"][
                "scalar_eps_x_on_Ez"
            ],
            "false_Ez_scalar_eps_y": disk["false_ez_integrals"][
                "scalar_eps_y_on_Ez"
            ],
            "false_eps_x_over_actual_excess": (
                disk["false_ez_integrals"]["scalar_eps_x_on_Ez"]
                / disk_excess
            ),
            "false_eps_y_over_actual_excess": (
                disk["false_ez_integrals"]["scalar_eps_y_on_Ez"]
                / disk_excess
            ),
        },
        "spatial_correlations_in_TaIrTe4": correlations,
    }
    (output / "native_yee_component_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    rows = []
    for name in ("flat", "disk", "square"):
        row = {"case": name, **jsonable_case(cases[name])}
        rows.append(row)
    with (output / "native_yee_component_table.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
