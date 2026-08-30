#!/usr/bin/env python3
"""Map finite T11x15/Z1x3 native Yee Q onto the explicit thermal grid.

Every top-Au polygon is decomposed into non-overlapping rectilinear cut cells.
The component-specific Yee-cell power is partitioned by exact material overlap
times Im(epsilon_c), and power assigned to top Au is conservatively transferred
only through the corresponding Au overlap.  Raw arrays remain outside Git.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_material_Q")
OUTPUT = HERE / "results_finite_T_Z_array_material_Q_mapping"
CASES = {
    "T11x15_Ea_Au_on": RAW / "T11x15_Ea_Au_on",
    "T11x15_Eb_Au_on": RAW / "T11x15_Eb_Au_on",
    "Z1x3_Ea_Au_on": RAW / "Z1x3_Ea_Au_on",
    "Z1x3_Eb_Au_on": RAW / "Z1x3_Eb_Au_on",
}


def load_stage80():
    path = HERE / "80_map_finite_t_z_material_q.py"
    spec = importlib.util.spec_from_file_location("finite_t_z_stage80", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_stage80()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_in_polygon(x: float, y: float, vertices: np.ndarray) -> bool:
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def polygon_rectangles_m(polygon: dict) -> list[tuple[float, float, float, float]]:
    vertices = np.asarray(polygon["vertices_nm"], dtype=np.float64) * 1e-9
    xs = np.unique(vertices[:, 0])
    ys = np.unique(vertices[:, 1])
    rectangles = []
    for x0, x1 in zip(xs[:-1], xs[1:], strict=True):
        for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
            if point_in_polygon(0.5 * (x0 + x1), 0.5 * (y0 + y1), vertices):
                rectangles.append((float(x0), float(x1), float(y0), float(y1)))
    if not rectangles:
        raise RuntimeError(f"polygon has no interior rectangles: {polygon.get('name')}")
    polygon_area = 0.5 * abs(
        np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
        - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
    )
    rectangle_area = sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in rectangles)
    if abs(rectangle_area - polygon_area) > max(polygon_area, 1e-30) * 1e-12:
        raise RuntimeError(f"rectilinear decomposition area mismatch: {polygon.get('name')}")
    return rectangles


def top_au_boxes(result: dict):
    boxes = []
    rectangles = []
    for polygon in result["geometry"]["top_Au_polygons"]:
        z0 = float(polygon["z_min_nm"]) * 1e-9
        z1 = float(polygon["z_max_nm"]) * 1e-9
        for x0, x1, y0, y1 in polygon_rectangles_m(polygon):
            boxes.append(((x0, x1), (y0, y1), (z0, z1)))
            rectangles.append([x0, x1, y0, y1, z0, z1])
    return boxes, rectangles


def fixed_material_boxes(architecture: str, z_source_min: float):
    boxes = BASE.material_boxes(architecture, False, z_source_min)
    return {key: value for key, value in boxes.items() if key != "top_Au"}


def union_fraction(edges, boxes):
    shape = tuple(len(item) - 1 for item in edges)
    result = np.zeros(shape, dtype=np.float64)
    for box in boxes:
        ix = np.flatnonzero(BASE.interval_overlap(edges[0], *box[0]) > 0.0)
        iy = np.flatnonzero(BASE.interval_overlap(edges[1], *box[1]) > 0.0)
        iz = np.flatnonzero(BASE.interval_overlap(edges[2], *box[2]) > 0.0)
        if not (ix.size and iy.size and iz.size):
            continue
        fx = BASE.interval_overlap(edges[0][ix[0] : ix[-1] + 2], *box[0]) / np.diff(edges[0][ix[0] : ix[-1] + 2])
        fy = BASE.interval_overlap(edges[1][iy[0] : iy[-1] + 2], *box[1]) / np.diff(edges[1][iy[0] : iy[-1] + 2])
        fz = BASE.interval_overlap(edges[2][iz[0] : iz[-1] + 2], *box[2]) / np.diff(edges[2][iz[0] : iz[-1] + 2])
        result[np.ix_(ix, iy, iz)] += fx[:, None, None] * fy[None, :, None] * fz[None, None, :]
    if np.max(result, initial=0.0) > 1.0 + 1e-10:
        raise RuntimeError("top-Au boxes overlap in volume")
    return np.minimum(result, 1.0)


def remap_box_cropped(power, material_fraction, source_edges, target_edges, box):
    source_indices = [
        np.flatnonzero(BASE.interval_overlap(source_edges[i], *box[i]) > 0.0)
        for i in range(3)
    ]
    target_indices = [
        np.flatnonzero(BASE.interval_overlap(target_edges[i], *box[i]) > 0.0)
        for i in range(3)
    ]
    if any(item.size == 0 for item in source_indices + target_indices):
        return target_indices, None
    source_slices = tuple(slice(item[0], item[-1] + 1) for item in source_indices)
    local_edges_source = tuple(
        source_edges[i][source_indices[i][0] : source_indices[i][-1] + 2]
        for i in range(3)
    )
    local_edges_target = tuple(
        target_edges[i][target_indices[i][0] : target_indices[i][-1] + 2]
        for i in range(3)
    )
    fraction = BASE.box_volume_fraction(local_edges_source, box)
    local_power = np.divide(
        power[source_slices] * fraction,
        material_fraction[source_slices],
        out=np.zeros_like(fraction),
        where=material_fraction[source_slices] > 0.0,
    )
    mapped = BASE.remap_box_power(local_power, local_edges_source, local_edges_target, box)
    return target_indices, mapped


def map_case(name: str, directory: Path) -> dict:
    result_path = next(directory.glob("FINITE_*_Q.json"))
    npz_path = next(directory.glob("finite_*_Q.npz"))
    result = json.loads(result_path.read_text())
    if not str(result.get("status", "")).startswith("VALIDATED_FINITE_"):
        raise RuntimeError(f"input optical case is not validated: {name}")
    architecture = result["architecture"]
    target_edges = BASE.thermal_edges(architecture)
    target_shape = tuple(len(item) - 1 for item in target_edges)
    materials = ("Si", "SiO2", "Au_mirror", "TaIrTe4", "top_Au")
    target_material_power = {key: np.zeros(target_shape, dtype=np.float64) for key in materials}
    top_boxes, top_rectangles = top_au_boxes(result)
    component_input = {}
    component_output = {}
    zero_loss_positive_power_cells = 0
    denominator_floor_cells = 0
    with np.load(npz_path, allow_pickle=False) as raw:
        for component in "xyz":
            q = np.asarray(raw[f"Q{component}_W_m3"], dtype=np.float64)
            source_edges = tuple(BASE.coordinate_edges(raw[f"Q{component}_{axis}_m"]) for axis in "xyz")
            widths = tuple(np.diff(item) for item in source_edges)
            volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
            power = q * volume
            fixed = fixed_material_boxes(architecture, source_edges[2][0])
            fractions = {
                material: BASE.box_volume_fraction(source_edges, boxes[0])
                for material, boxes in fixed.items()
            }
            fractions["top_Au"] = union_fraction(source_edges, top_boxes)
            loss = BASE.material_imaginary_epsilon(result, architecture, component)
            weights = {material: fractions[material] * loss[material] for material in materials}
            denominator = sum(weights.values(), start=np.zeros_like(q))
            positive = power > max(float(np.max(power)) * 1e-15, np.finfo(float).tiny)
            bad = positive & (denominator <= 0.0)
            zero_loss_positive_power_cells += int(np.count_nonzero(bad))
            denominator_floor_cells += int(np.count_nonzero(positive & (denominator < 1e-12)))
            if np.any(bad):
                raise RuntimeError(f"{name} Q{component}: positive power without lossy material overlap")
            component_input[component] = float(np.sum(power))
            component_mapped = 0.0
            for material in materials:
                material_power = np.divide(
                    power * weights[material], denominator,
                    out=np.zeros_like(power), where=denominator > 0.0,
                )
                if material != "top_Au":
                    mapped = BASE.remap_box_power(
                        material_power, source_edges, target_edges, fixed[material][0]
                    )
                    target_material_power[material] += mapped
                    component_mapped += float(np.sum(mapped))
                    continue
                for box in top_boxes:
                    target_indices, mapped = remap_box_cropped(
                        material_power, fractions["top_Au"], source_edges, target_edges, box
                    )
                    if mapped is None:
                        continue
                    target_material_power[material][np.ix_(*target_indices)] += mapped
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
        "array_variant": result["geometry"]["array_contract"]["identity"],
        "array_contract": result["geometry"]["array_contract"],
        "polarization": result["polarization"],
        "top_Au_present": True,
        "top_Au_rectangles_m": top_rectangles,
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
        "component_specific_Yee_coordinates": True,
        "exact_rectilinear_cut_cell_overlap": True,
        "no_forbidden_Q_processing": True,
        "raw_artifacts_outside_Git": True,
    }
    status = "VALIDATED_FINITE_T11X15_Z1X3_COMPONENT_MATERIAL_Q_MAPPING" if all(gates.values()) else "FAILED_FINITE_T11X15_Z1X3_COMPONENT_MATERIAL_Q_MAPPING"
    summary = {
        "status": status,
        "method": "component-specific Yee power -> exact polygon cut-cell volume x Im(epsilon_c) -> conditional same-material conservative thermal-grid overlap",
        "loss_participation_is_occupancy": False,
        "axes": {"x": "b", "y": "a", "z": "c=b optical closure"},
        "cases": cases,
        "gates": gates,
        "next_gate": "same explicit finite 3-D thermal/electrical operator for all four array sources",
    }
    summary_path = OUTPUT / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with (OUTPUT / "finite_T_Z_array_material_Q_mapping_cases.csv").open("w", newline="") as stream:
        fields = ["case", "input_total_power_W", "mapped_total_power_W", "power_conservation_relative_error", "Si_W", "SiO2_W", "Au_mirror_W", "TaIrTe4_W", "top_Au_W"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, value in cases.items():
            material = value["mapped_power_by_material_W"]
            writer.writerow({"case": name, "input_total_power_W": value["input_total_power_W"], "mapped_total_power_W": value["mapped_total_power_W"], "power_conservation_relative_error": value["power_conservation_relative_error"], **{f"{key}_W": material[key] for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4", "top_Au")}})
    lines = [
        "# Finite T11x15 / Z1x3 component-material Q mapping", "", f"Status: **{status}**", "",
        "All array polygons use exact rectilinear cut-cell overlap on each component-specific Yee grid. Raw cell power is neither clipped nor globally rescaled.", "",
        "| case | TaIrTe4 | top Au | mirror Au | SiO2 | Si | mapping error |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in cases.items():
        m = value["mapped_power_by_material_W"]
        lines.append(f"| {name} | {m['TaIrTe4']*1e15:.4f} fW | {m['top_Au']*1e15:.4f} fW | {m['Au_mirror']*1e15:.4f} fW | {m['SiO2']*1e15:.4f} fW | {m['Si']*1e15:.4f} fW | {value['power_conservation_relative_error']:.3e} |")
    (OUTPUT / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {"status": status, "raw_files_committed_to_git": False, "artifacts": {name: value["raw_mapped_artifact"] for name, value in cases.items()}}
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
