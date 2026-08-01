#!/usr/bin/env python3
"""Mesh-convergence analysis of the straight-45-edge optical Q.

Consumes the six ``w2edge_conv_*`` edge-isolation-smoke artifacts
(polarization a/b x uniform local mesh 50/25/12.5 nm on a 12-um domain)
and quantifies whether the edge-localized absorption that drives the
Device-A |Ia|/|Ib| disagreement is mesh-converged or a staircase artifact.

Pure post-processing: no Lumerical session is opened.

Observables per run (flake occupies y <= x; edge line y = x; inward
normal n = (1, -1)/sqrt(2); signed inward distance d_n = (x - y)/sqrt(2)):

* areal absorption profile lambda(d_n): Q integrated over z and along the
  edge tangent inside |d_t| <= tangent-window, per unit edge length and
  per unit d_n, normalized by the realized source power;
* edge-band absorbed fraction P(|d_n| <= band)/P_flake for bands
  0.25 / 0.5 um;
* edge enhancement factor: peak lambda near the edge over the interior
  plateau (d_n in [1.5, 2.5] um);
* Cartesian component split (Qx, Qy, Qz) of the same profile;
* cross-mesh convergence of all of the above (50 -> 25 -> 12.5 nm),
  including the a/b edge contrast per mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BANDS_UM = (0.25, 0.5)
PLATEAU_UM = (1.5, 2.5)
PROFILE_RANGE_UM = (-1.0, 3.0)
PROFILE_BIN_UM = 0.05
TANGENT_WINDOW_UM = 2.0


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_run(directory: Path) -> dict[str, Any]:
    """Load a run, accepting the documented auto-shutoff-floor failure.

    The 12-um edge-isolation-smoke geometry has a case-dependent
    auto-shutoff floor above the requested 1e-5 (the same floor the
    accepted w2 planar/edge diagnostics recorded), so a run is usable
    when the ONLY failed acceptance gate is
    ``auto_shutoff_reached_requested_threshold`` and the solver
    completed; the six-face closure gates must have passed.
    """
    case = json.loads((directory / "case_result.json").read_text())
    acceptance = (case.get("run_result") or {}).get("acceptance") or {}
    failed = [key for key, value in acceptance.items() if not value]
    completed = ((case.get("run_result") or {}).get("auto_shutoff") or {}).get(
        "simulation_completed_successfully", False
    )
    if not acceptance or not completed or not (
        set(failed) <= {"auto_shutoff_reached_requested_threshold"}
    ):
        raise RuntimeError(
            f"unusable run {directory}: failed gates {failed}"
        )
    data = np.load(directory / "diagnostic_q_common_grid_artifact.npz")
    return {"dir": directory, "case": case, "npz": data}


def cell_sizes(coordinates: np.ndarray) -> np.ndarray:
    edges = np.empty(coordinates.size + 1)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
    edges[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    return np.diff(edges)


def analyze_run(run: dict[str, Any]) -> dict[str, Any]:
    npz = run["npz"]
    x = np.asarray(npz["x_m"], float)
    y = np.asarray(npz["y_m"], float)
    z = np.asarray(npz["z_m"], float)
    mask = np.asarray(npz["exact_flake_mask"], bool)
    dx, dy, dz = cell_sizes(x), cell_sizes(y), cell_sizes(z)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    source_power = float(np.asarray(npz["source_power_native_W"]).ravel()[0])

    components = {
        "total": np.asarray(npz["Q_common_grid_W_m3"], float),
        "Qx": np.asarray(npz["Qx_common_grid_W_m3"], float),
        "Qy": np.asarray(npz["Qy_common_grid_W_m3"], float),
        "Qz": np.asarray(npz["Qz_common_grid_W_m3"], float),
    }

    grid_x = x[:, None]
    grid_y = y[None, :]
    distance_normal = (grid_x - grid_y) / np.sqrt(2.0)
    distance_tangent = (grid_x + grid_y) / np.sqrt(2.0)
    inside_window = np.abs(distance_tangent) <= TANGENT_WINDOW_UM * 1.0e-6

    flake_any = np.any(mask, axis=2)
    flake_side = distance_normal[flake_any]
    if flake_side.size and np.min(flake_side) < -1.0e-7:
        raise RuntimeError(
            f"{run['dir'].name}: flake mask extends beyond y <= x by "
            f"{-np.min(flake_side):.3e} m"
        )

    power = {
        name: float(np.sum(field[mask] * volume[mask]))
        for name, field in components.items()
    }

    bins = np.arange(
        PROFILE_RANGE_UM[0], PROFILE_RANGE_UM[1] + 1e-12, PROFILE_BIN_UM
    ) * 1.0e-6
    centers = 0.5 * (bins[:-1] + bins[1:])
    profiles: dict[str, np.ndarray] = {}
    masked_volume = np.where(mask, volume, 0.0)
    window_3d = inside_window[:, :, None]
    weight = masked_volume * window_3d
    flat_dn = np.broadcast_to(
        distance_normal[:, :, None], mask.shape
    ).ravel()
    tangent_length = 2.0 * TANGENT_WINDOW_UM * 1.0e-6
    for name, field in components.items():
        deposition = (field * weight).ravel()
        histogram, _ = np.histogram(flat_dn, bins=bins, weights=deposition)
        profiles[name] = histogram / (
            PROFILE_BIN_UM * 1.0e-6 * tangent_length * source_power
        )

    band_fractions = {}
    for band_um in BANDS_UM:
        selected = (
            mask
            & (np.abs(distance_normal[:, :, None]) <= band_um * 1.0e-6)
        )
        band_fractions[f"{band_um:g}um"] = float(
            np.sum(components["total"][selected] * volume[selected])
            / power["total"]
        )

    plateau = (centers >= PLATEAU_UM[0] * 1e-6) & (
        centers <= PLATEAU_UM[1] * 1e-6
    )
    near_edge = (centers >= 0.0) & (centers <= 0.5e-6)
    plateau_level = float(np.mean(profiles["total"][plateau]))
    peak_level = float(np.max(profiles["total"][near_edge]))
    result = {
        "directory": run["dir"].name,
        "mesh_nm": float(run["case"]["run_result"]["requested_local_xy_mesh_nm"])
        if "requested_local_xy_mesh_nm" in run["case"].get("run_result", {})
        else None,
        "source_power_W": source_power,
        "P_Q_W": power["total"],
        "absorbed_fraction_of_source": power["total"] / source_power,
        "component_power_W": {k: v for k, v in power.items() if k != "total"},
        "six_face_relative_closure": run["case"]
        .get("run_result", {})
        .get(
            "common_grid_six_face_relative_closure",
            run["case"].get("run_result", {}).get(
                "six_face_relative_closure"
            ),
        ),
        "auto_shutoff_final_value": run["case"]
        .get("run_result", {})
        .get("auto_shutoff", {})
        .get("final_value"),
        "band_fraction_of_flake_absorption": band_fractions,
        "edge_peak_over_plateau": peak_level / plateau_level,
        "plateau_level_per_W_m2": plateau_level,
        "edge_peak_per_W_m2": peak_level,
        "profile_bin_centers_um": centers * 1.0e6,
        "profiles_per_W_m2": profiles,
    }
    return result


def relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(old), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity"
        ),
    )
    parser.add_argument("--stamp", default="20260801")
    parser.add_argument("--gpu-map", default="a=4,b=5")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_of = dict(
        pair.split("=") for pair in args.gpu_map.split(",")
    )
    meshes = ("50", "25", "12p5")
    mesh_nm = {"50": 50.0, "25": 25.0, "12p5": 12.5}
    runs: dict[str, dict[str, Any]] = {}
    for pol in ("a", "b"):
        for mesh in meshes:
            name = (
                f"w2edge_conv_{pol}_xy{mesh}_dz5_t4_"
                f"gpu{gpu_of[pol]}_{args.stamp}"
            )
            directory = args.artifact_root / name
            key = f"{pol}/{mesh}"
            runs[key] = analyze_run(load_run(directory))
            runs[key]["mesh_nm"] = mesh_nm[mesh]

    convergence: dict[str, Any] = {}
    for pol in ("a", "b"):
        seq = [runs[f"{pol}/{mesh}"] for mesh in meshes]
        entry: dict[str, Any] = {}
        for metric, getter in (
            (
                "band_fraction_0.25um",
                lambda r: r["band_fraction_of_flake_absorption"]["0.25um"],
            ),
            (
                "band_fraction_0.5um",
                lambda r: r["band_fraction_of_flake_absorption"]["0.5um"],
            ),
            ("edge_peak_over_plateau", lambda r: r["edge_peak_over_plateau"]),
            (
                "absorbed_fraction_of_source",
                lambda r: r["absorbed_fraction_of_source"],
            ),
            ("edge_peak_per_W_m2", lambda r: r["edge_peak_per_W_m2"]),
        ):
            values = [getter(r) for r in seq]
            step_50_25 = relative_change(values[1], values[0])
            step_25_12 = relative_change(values[2], values[1])
            entry[metric] = {
                "values_50_25_12p5": values,
                "rel_change_50_to_25": step_50_25,
                "rel_change_25_to_12p5": step_25_12,
                "convergence_ratio": (
                    step_25_12 / step_50_25 if step_50_25 > 0 else None
                ),
            }
        convergence[pol] = entry

    contrast = {}
    for mesh in meshes:
        run_a = runs[f"a/{mesh}"]
        run_b = runs[f"b/{mesh}"]
        contrast[mesh] = {
            "edge_band_0.5um_fraction_a_over_b": (
                run_a["band_fraction_of_flake_absorption"]["0.5um"]
                / run_b["band_fraction_of_flake_absorption"]["0.5um"]
            ),
            "edge_peak_over_plateau_a_over_b": (
                run_a["edge_peak_over_plateau"]
                / run_b["edge_peak_over_plateau"]
            ),
            "absorbed_fraction_a_over_b": (
                run_a["absorbed_fraction_of_source"]
                / run_b["absorbed_fraction_of_source"]
            ),
        }

    summary = {
        "status": "COMPLETED_EDGE_Q_MESH_CONVERGENCE_ANALYSIS",
        "contract": {
            "geometry": "straight-45-edge (flake y <= x), 12 um domain",
            "source": "edge-isolation-smoke Gaussian, nominal w0 = 2 um",
            "meshes_nm": [50.0, 25.0, 12.5],
            "flake_dz_nm": 5.0,
            "tangent_window_um": TANGENT_WINDOW_UM,
            "profile_bin_um": PROFILE_BIN_UM,
        },
        "runs": {
            key: {k: v for k, v in run.items() if not k.startswith("profile")}
            for key, run in runs.items()
        },
        "convergence": convergence,
        "a_over_b_contrast_by_mesh": contrast,
    }
    (args.report_dir / "edge_q_mesh_convergence_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n"
    )
    np.savez_compressed(
        args.report_dir / "edge_q_mesh_convergence_profiles.npz",
        **{
            f"{key.replace('/', '_')}_{name}": run["profiles_per_W_m2"][name]
            for key, run in runs.items()
            for name in ("total", "Qx", "Qy", "Qz")
        },
        bin_centers_um=runs["a/50"]["profile_bin_centers_um"],
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    colors = {"50": "#66ccee", "25": "#4477aa", "12p5": "#cc3311"}
    for axis, pol in zip(axes, ("a", "b")):
        for mesh in meshes:
            run = runs[f"{pol}/{mesh}"]
            axis.plot(
                run["profile_bin_centers_um"],
                run["profiles_per_W_m2"]["total"],
                color=colors[mesh],
                label=f"{mesh_nm[mesh]:g} nm",
            )
        axis.set_title(f"E||{pol}: areal Q profile vs edge distance")
        axis.set_xlabel("inward distance from edge d_n [um]")
        axis.grid(alpha=0.25)
        axis.legend(title="local mesh")
    axes[0].set_ylabel("z,t-integrated Q / P_source [1/m^2]")
    figure.tight_layout()
    figure.savefig(args.report_dir / "EDGEQ_PROFILE_CONVERGENCE.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for axis, pol in zip(axes, ("a", "b")):
        run = runs[f"{pol}/12p5"]
        for name, color in (
            ("Qx", "#4477aa"),
            ("Qy", "#cc3311"),
            ("Qz", "#228833"),
        ):
            axis.plot(
                run["profile_bin_centers_um"],
                run["profiles_per_W_m2"][name],
                color=color,
                label=name,
            )
        axis.set_title(f"E||{pol}: component split at 12.5 nm")
        axis.set_xlabel("inward distance from edge d_n [um]")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("z,t-integrated Q / P_source [1/m^2]")
    figure.tight_layout()
    figure.savefig(args.report_dir / "EDGEQ_COMPONENT_SPLIT.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    mesh_values = [50.0, 25.0, 12.5]
    for pol, marker in (("a", "o"), ("b", "s")):
        fractions = [
            runs[f"{pol}/{mesh}"]["band_fraction_of_flake_absorption"][
                "0.5um"
            ]
            for mesh in meshes
        ]
        axis.plot(
            mesh_values, fractions, marker=marker, label=f"E||{pol}"
        )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("local x/y mesh [nm]")
    axis.set_ylabel("edge-band (|d_n| <= 0.5 um) fraction of P_Q")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.report_dir / "EDGEQ_BAND_FRACTION_VS_MESH.png", dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            jsonable(
                {
                    "convergence": convergence,
                    "a_over_b_contrast_by_mesh": contrast,
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
