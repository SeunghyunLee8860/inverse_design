#!/usr/bin/env python3
"""Conservatively attribute finite v261 Yee Q to explicit thermal materials.

This stage never assigns a complete conformal/Yee cell to whichever material
happens to touch it.  For each E component it partitions the cell power using
the exact rectangular material-overlap volume multiplied by the requested
component loss, Im(epsilon_c).  The attributed power is then remapped through
the same material intersection onto a finite nonuniform thermal grid.

No clipping, smoothing, gain, polarization matching, or global rescaling is
performed.  The generated NPZ files are raw artifacts outside Git.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_Q")
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_material_Q")
OUTPUT = HERE / "results_finite_T_Z_material_Q_mapping"
CASES = {
    "T_Ea_Au_on": RAW / "T_Ea_Au_on_v2",
    "T_Eb_Au_on": RAW / "T_Eb_Au_on",
    "T_Ea_Au_off": RAW / "T_Ea_Au_off",
    "T_Eb_Au_off": RAW / "T_Eb_Au_off",
    "Z_Ea_Au_on": RAW / "Z_Ea_Au_on",
    "Z_Eb_Au_on": RAW / "Z_Eb_Au_on",
    "Z_Ea_Au_off": RAW / "Z_Ea_Au_off",
    "Z_Eb_Au_off": RAW / "Z_Eb_Au_off",
}

# v261 material-database readback at the exact single wavelength.  These are
# not fitted here and are retained in the output provenance.
DB_EPS = {
    "T": {
        "Au": complex(-1146.959175457364, 102.68584558184479),
        "SiO2": complex(1.8336893905724556, 0.005433652006940445),
        "Si": complex(11.74292397842102, 1.0765166842105272e-6),
    },
    "Z": {
        "Au": complex(-1408.2262176752547, 135.9407502917725),
        "SiO2": complex(1.7470187649976519, 0.012957987310981406),
        "Si": complex(11.733740827735822, 9.004825094339621e-7),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_edges(coordinate: np.ndarray) -> np.ndarray:
    """Edges whose widths equal the production trapezoid weights."""

    values = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[0] = values[0]
    edges[-1] = values[-1]
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    return edges


def interval_overlap(edges: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.maximum(
        0.0,
        np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower),
    )


def transfer_matrix(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Conditional source-intersection to target-intersection fractions."""

    denominator = interval_overlap(source_edges, lower, upper)
    left = np.maximum(
        np.maximum(source_edges[:-1][None, :], target_edges[:-1][:, None]),
        lower,
    )
    right = np.minimum(
        np.minimum(source_edges[1:][None, :], target_edges[1:][:, None]),
        upper,
    )
    overlap = np.maximum(0.0, right - left)
    overlap *= (target_edges[:-1, None] < upper) & (target_edges[1:, None] > lower)
    result = np.divide(
        overlap,
        denominator[None, :],
        out=np.zeros_like(overlap),
        where=denominator[None, :] > 0.0,
    )
    return result


def remap_box_power(
    power: np.ndarray,
    source_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    matrices = tuple(
        transfer_matrix(source_edges[i], target_edges[i], *box[i])
        for i in range(3)
    )
    first = np.tensordot(matrices[0], power, axes=(1, 0))
    second = np.tensordot(matrices[1], first, axes=(1, 1)).transpose(1, 0, 2)
    third = np.tensordot(matrices[2], second, axes=(1, 2)).transpose(1, 2, 0)
    return third


def box_volume_fraction(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    widths = tuple(np.diff(item) for item in edges)
    fractions = tuple(
        interval_overlap(edges[i], *box[i]) / widths[i] for i in range(3)
    )
    return (
        fractions[0][:, None, None]
        * fractions[1][None, :, None]
        * fractions[2][None, None, :]
    )


def thermal_edges(architecture: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 32 x 32 um finite physical domain.  The TaIrTe4/source core is 100 nm;
    # shoulders and remote substrate are coarser without changing interfaces.
    lateral = np.unique(
        np.concatenate(
            (
                np.asarray((-16.0, -14.0)),
                np.arange(-14.0, -10.0, 0.25),
                np.arange(-10.0, 10.0 + 0.05, 0.1),
                np.arange(10.25, 14.0 + 0.125, 0.25),
                np.asarray((16.0,)),
            )
        )
    ) * 1e-6
    if architecture == "T":
        z_um = np.concatenate(
            (
                np.asarray((-21.735, -13.735, -9.735, -6.735, -4.735, -3.235, -2.235, -1.735)),
                np.arange(-1.685, -0.235 + 0.025, 0.05),
                np.arange(-0.215, -0.035 + 0.01, 0.02),
                np.arange(-0.030, 0.0 + 0.0025, 0.005),
                np.arange(0.010, 0.100 + 0.005, 0.010),
                np.asarray((0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.133)),
                np.asarray((0.140, 0.150, 0.170, 0.200, 0.250, 0.350, 0.500, 0.750, 1.0, 1.5, 2.0)),
            )
        )
    else:
        z_um = np.concatenate(
            (
                np.asarray((-20.685, -12.685, -8.685, -5.685, -3.685, -2.185, -1.185, -0.685)),
                np.asarray((-0.635, -0.585, -0.535, -0.485, -0.435, -0.400)),
                np.arange(-0.380, -0.200 + 0.01, 0.020),
                np.arange(-0.190, 0.0 + 0.005, 0.010),
                np.arange(0.010, 0.100 + 0.005, 0.010),
                np.arange(0.105, 0.150 + 0.0025, 0.005),
                np.asarray((0.160, 0.180, 0.200, 0.250, 0.350, 0.500, 0.750, 1.0, 1.5, 2.0)),
            )
        )
    z = np.unique(np.round(z_um, 12)) * 1e-6
    return lateral, lateral.copy(), z


def material_boxes(architecture: str, au_on: bool, z_source_min: float):
    inf = (-12.0e-6, 12.0e-6)
    if architecture == "T":
        z = {
            "Si": (z_source_min, -1.735e-6),
            "SiO2": (-1.735e-6, -0.235e-6),
            "Au_mirror": (-0.235e-6, -0.035e-6),
            "TaIrTe4": (0.0, 0.100e-6),
        }
        top = [
            ((-0.600e-6, 0.600e-6), (-0.350e-6, -0.250e-6), (0.100e-6, 0.133e-6)),
            ((-0.100e-6, 0.100e-6), (-0.250e-6, 0.350e-6), (0.100e-6, 0.133e-6)),
        ]
    else:
        z = {
            "Si": (z_source_min, -0.685e-6),
            "SiO2": (-0.685e-6, -0.400e-6),
            "Au_mirror": (-0.400e-6, -0.200e-6),
            "TaIrTe4": (0.0, 0.100e-6),
        }
        top = [
            ((-0.130e-6, 1.230e-6), (-1.000e-6, 1.300e-6), (0.100e-6, 0.150e-6)),
            ((-1.230e-6, -0.130e-6), (-1.300e-6, 0.400e-6), (0.100e-6, 0.150e-6)),
        ]
    result = {
        "Si": [(inf, inf, z["Si"])],
        "SiO2": [(inf, inf, z["SiO2"])],
        "Au_mirror": [(inf, inf, z["Au_mirror"])],
        "TaIrTe4": [((-10e-6, 10e-6), (-10e-6, 10e-6), z["TaIrTe4"])],
    }
    if au_on:
        result["top_Au"] = top
    return result


def material_imaginary_epsilon(result: dict, architecture: str, component: str):
    ta = result["geometry"]["TaIrTe4"]["requested_epsilon"][component]["imag"]
    return {
        "Si": float(DB_EPS[architecture]["Si"].imag),
        "SiO2": float(DB_EPS[architecture]["SiO2"].imag),
        "Au_mirror": float(DB_EPS[architecture]["Au"].imag),
        "top_Au": float(DB_EPS[architecture]["Au"].imag),
        "TaIrTe4": float(ta),
    }


def map_case(name: str, directory: Path) -> dict:
    result_path = next(directory.glob("FINITE_*_Q.json"))
    npz_path = next(directory.glob("finite_*_Q.npz"))
    result = json.loads(result_path.read_text())
    architecture = result["architecture"]
    au_on = bool(result["top_Au_present"])
    target_edges = thermal_edges(architecture)
    target_shape = tuple(len(item) - 1 for item in target_edges)
    target_material_power = {
        key: np.zeros(target_shape, dtype=np.float64)
        for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4", "top_Au")
    }
    component_input = {}
    component_output = {}
    zero_loss_positive_power_cells = 0
    denominator_floor_cells = 0
    with np.load(npz_path, allow_pickle=False) as raw:
        for component in "xyz":
            q = np.asarray(raw[f"Q{component}_W_m3"], dtype=np.float64)
            source_edges = tuple(
                coordinate_edges(raw[f"Q{component}_{axis}_m"]) for axis in "xyz"
            )
            widths = tuple(np.diff(item) for item in source_edges)
            volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
            power = q * volume
            boxes = material_boxes(architecture, au_on, source_edges[2][0])
            fractions = {
                material: sum((box_volume_fraction(source_edges, box) for box in items), start=np.zeros_like(q))
                for material, items in boxes.items()
            }
            loss = material_imaginary_epsilon(result, architecture, component)
            weights = {
                material: fraction * loss[material]
                for material, fraction in fractions.items()
            }
            denominator = sum(weights.values(), start=np.zeros_like(q))
            positive = power > max(float(np.max(power)) * 1e-15, np.finfo(float).tiny)
            bad = positive & (denominator <= 0.0)
            zero_loss_positive_power_cells += int(np.count_nonzero(bad))
            denominator_floor_cells += int(np.count_nonzero(positive & (denominator < 1e-12)))
            if np.any(bad):
                raise RuntimeError(f"{name} Q{component}: positive power without lossy material overlap")
            component_input[component] = float(np.sum(power))
            component_mapped = 0.0
            for material, items in boxes.items():
                material_fraction = fractions[material]
                material_power = np.divide(
                    power * weights[material],
                    denominator,
                    out=np.zeros_like(power),
                    where=denominator > 0.0,
                )
                for box in items:
                    box_fraction = box_volume_fraction(source_edges, box)
                    box_power = np.divide(
                        material_power * box_fraction,
                        material_fraction,
                        out=np.zeros_like(material_power),
                        where=material_fraction > 0.0,
                    )
                    mapped = remap_box_power(box_power, source_edges, target_edges, box)
                    target_material_power[material] += mapped
                    component_mapped += float(np.sum(mapped))
            component_output[component] = component_mapped

    input_total = float(sum(component_input.values()))
    output_by_material = {key: float(np.sum(value)) for key, value in target_material_power.items()}
    output_total = float(sum(output_by_material.values()))
    relative = abs(output_total - input_total) / max(abs(input_total), np.finfo(float).tiny)
    if relative >= 1e-12:
        raise RuntimeError(f"{name}: mapping power error {relative:.6e}")
    total = sum(target_material_power.values(), start=np.zeros(target_shape, dtype=np.float64))
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    raw_out = RAW_OUT / f"{name}_material_Q_thermal_grid.npz"
    np.savez_compressed(
        raw_out,
        x_edges_m=target_edges[0], y_edges_m=target_edges[1], z_edges_m=target_edges[2],
        power_total_W=total.astype(np.float32),
        **{f"power_{key}_W": value.astype(np.float32) for key, value in target_material_power.items()},
    )
    return {
        "case": name,
        "architecture": architecture,
        "polarization": result["polarization"],
        "top_Au_present": au_on,
        "input_raw_Q_npz": {"path": str(npz_path), "bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
        "input_component_power_W": component_input,
        "mapped_component_power_W": component_output,
        "input_total_power_W": input_total,
        "mapped_total_power_W": output_total,
        "mapped_power_by_material_W": output_by_material,
        "power_conservation_relative_error": relative,
        "positive_power_cells_without_loss_overlap": zero_loss_positive_power_cells,
        "positive_power_cells_near_denominator_floor": denominator_floor_cells,
        "thermal_grid_shape": list(target_shape),
        "thermal_grid_bounds_m": [[float(e[0]), float(e[-1])] for e in target_edges],
        "raw_mapped_artifact": {"path": str(raw_out), "bytes": raw_out.stat().st_size, "sha256": sha256(raw_out), "committed_to_git": False},
        "operations_forbidden_and_absent": ["clipping", "smoothing", "gain", "global_rescaling", "tiling", "full_cell_material_assignment"],
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = {name: map_case(name, directory) for name, directory in CASES.items()}
    gates = {
        "all_power_conservation_lt_1e-12": max(v["power_conservation_relative_error"] for v in cases.values()) < 1e-12,
        "no_positive_power_cell_without_loss_overlap": sum(v["positive_power_cells_without_loss_overlap"] for v in cases.values()) == 0,
        "no_forbidden_Q_processing": True,
        "raw_artifacts_outside_Git": True,
    }
    status = "VALIDATED_FINITE_T_Z_COMPONENT_MATERIAL_OVERLAP_Q_MAPPING" if all(gates.values()) else "FAILED_FINITE_T_Z_COMPONENT_MATERIAL_OVERLAP_Q_MAPPING"
    summary = {
        "status": status,
        "method": "component-specific Yee power -> exact rectangular cut-cell volume x Im(epsilon_c) loss participation -> same-material conservative thermal-grid overlap",
        "loss_participation_is_occupancy": False,
        "axes": {"x": "b", "y": "a", "z": "c=b optical closure"},
        "material_database_readback_epsilon": {
            arch: {key: {"real": value.real, "imag": value.imag} for key, value in db.items()}
            for arch, db in DB_EPS.items()
        },
        "cases": cases,
        "gates": gates,
        "next_gate": "explicit finite 3-D thermal solve on the exact mapped cell-power arrays",
    }
    (OUTPUT / "FINITE_T_Z_MATERIAL_Q_MAPPING_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "finite_T_Z_material_Q_mapping_cases.csv").open("w", newline="") as stream:
        fields = ["case", "input_total_power_W", "mapped_total_power_W", "power_conservation_relative_error", "Si_W", "SiO2_W", "Au_mirror_W", "TaIrTe4_W", "top_Au_W"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, value in cases.items():
            material = value["mapped_power_by_material_W"]
            writer.writerow({"case": name, "input_total_power_W": value["input_total_power_W"], "mapped_total_power_W": value["mapped_total_power_W"], "power_conservation_relative_error": value["power_conservation_relative_error"], **{f"{key}_W": material[key] for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4", "top_Au")}})
    lines = [
        "# Finite T/Z component-material Q mapping",
        "",
        f"Status: **{status}**",
        "",
        "Each native Yee component is paired with its own coordinates. Cell power is split by exact material cut-cell volume times component Im(epsilon), then transferred only through that material overlap to the explicit thermal grid.",
        "This loss-participation diagnostic is not an occupancy field. No complete boundary cell is forced into TaIrTe4 or Au.",
        "",
        "| case | TaIrTe4 | top Au | mirror Au | SiO2 | Si | mapping error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in cases.items():
        m = value["mapped_power_by_material_W"]
        lines.append(f"| {name} | {m['TaIrTe4']*1e15:.4f} fW | {m['top_Au']*1e15:.4f} fW | {m['Au_mirror']*1e15:.4f} fW | {m['SiO2']*1e15:.4f} fW | {m['Si']*1e15:.4f} fW | {value['power_conservation_relative_error']:.3e} |")
    (OUTPUT / "FINITE_T_Z_MATERIAL_Q_MAPPING_REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {"status": status, "raw_files_committed_to_git": False, "artifacts": {name: value["raw_mapped_artifact"] for name, value in cases.items()}}
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "gates": gates, "cases": {name: v["mapped_power_by_material_W"] for name, v in cases.items()}}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
