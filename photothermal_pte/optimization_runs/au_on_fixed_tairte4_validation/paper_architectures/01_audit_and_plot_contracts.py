#!/usr/bin/env python3
"""Audit and visualize the paper-derived TaIrTe4 architecture contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))

from contracts import (  # noqa: E402
    EXPECTED_PDF_SHA256,
    Z_PUBLISHED_DIMENSIONS_NM,
    architectures,
    optical_backplane_attenuation,
    proposed_10um_seed_sweeps,
    reduced_substrate_impedance,
)


COLORS = {
    "air": "#dbefff",
    "Au": "#e0ac22",
    "Ti/Au": "#d9981e",
    "Cr/Au": "#d9981e",
    "TaIrTe4": "#e85d5d",
    "Al2O3": "#b9dbe8",
    "SiO2": "#9fd7d0",
    "Si": "#65788f",
    "intrinsic Si": "#65788f",
    "heavily p-doped Si": "#52677f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("CSV requires rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_stack(ax, architecture) -> None:
    layers = list(architecture.layers)
    heights = []
    for layer in layers:
        if layer.material == "air":
            heights.append(0.55)
        elif layer.thickness_nm is None:
            heights.append(0.6)
        else:
            heights.append(max(0.16, min(0.75, layer.thickness_nm / 500.0)))
    total = sum(heights)
    y = total
    for layer, height in zip(layers, heights):
        y -= height
        color = COLORS.get(layer.material, "#cccccc")
        ax.add_patch(Rectangle((0.08, y), 0.84, height, facecolor=color, edgecolor="black", linewidth=0.8))
        thickness = "" if layer.thickness_nm is None else f" ({layer.thickness_nm:g} nm)"
        ax.text(0.5, y + height / 2, f"{layer.name}: {layer.material}{thickness}", ha="center", va="center", fontsize=8)
        if layer.name == "back_reflector":
            ax.axhline(y, color="red", linestyle="--", linewidth=1.1)
            ax.text(0.93, y - 0.03, "optical truncation below opaque Au", color="red", fontsize=7, ha="right", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, total + 0.08)
    ax.set_title(architecture.key, fontsize=9)
    ax.axis("off")


def _plot_t(ax) -> None:
    ax.add_patch(Rectangle((-0.8, -0.15), 1.6, 0.3, color="#d9981e"))
    ax.add_patch(Rectangle((-0.15, -0.15), 0.3, 1.1, color="#d9981e"))
    ax.set_title("2024 inverse-T schematic\n(not dimensioned at 10 um)")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.55, 1.1)
    ax.set_xlabel("baseline")
    ax.set_ylabel("crossbar")
    ax.grid(alpha=0.2)


def _plot_z(ax, handedness: str) -> None:
    sign = 1.0 if handedness == "LH" else -1.0
    vertices = [
        (-0.85, 0.75),
        (0.35, 0.75),
        (0.35, 0.45),
        (-0.25, 0.45),
        (0.65 * sign, -0.45),
        (0.85 * sign, -0.25),
        (0.85 * sign, -0.75),
        (-0.35, -0.75),
        (-0.35, -0.45),
        (0.25, -0.45),
        (-0.65 * sign, 0.45),
        (-0.85, 0.25),
    ]
    ax.add_patch(Polygon(vertices, closed=True, facecolor="#d9981e", edgecolor="black"))
    ax.set_title(f"2022 {handedness} Z schematic\nperiodic meta-molecule")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")


def _make_plot(contracts: dict, output: Path) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    for column, architecture in enumerate(contracts.values()):
        _plot_stack(fig.add_subplot(grid[0, column]), architecture)
    _plot_t(fig.add_subplot(grid[1, 0]))
    _plot_z(fig.add_subplot(grid[1, 1]), "LH")
    _plot_z(fig.add_subplot(grid[1, 2]), "RH")
    fig.suptitle(
        "Paper-derived architecture audit: only the active 2-D material is replaced by 100-nm TaIrTe4",
        fontsize=13,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    contracts = architectures()

    pdfs = {}
    for raw_path, expected in EXPECTED_PDF_SHA256.items():
        path = Path(raw_path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Supplement SHA mismatch: {path}: {actual} != {expected}")
        pdfs[str(path)] = {"bytes": path.stat().st_size, "sha256": actual}

    attenuation = [optical_backplane_attenuation(value) for value in (50, 100, 150, 200, 300)]
    reduced = {
        "2024_1p5um_thermal_SiO2": reduced_substrate_impedance(1500.0),
        "2022_285nm_thermal_SiO2": reduced_substrate_impedance(285.0),
    }
    gates = {
        "supplement_SHA256_verified": True,
        "all_active_layers_are_TaIrTe4": all(
            any(layer.name == "active_2d" and layer.material == "TaIrTe4" for layer in item.layers)
            for item in contracts.values()
        ),
        "direct_A_substrate_not_silently_reduced": not contracts["A_DIRECT_AU_TAIRTE4"].optical_substrate_reduction_allowed,
        "2024_T_top_Au_contacts_active_layer": [layer.name for layer in contracts["B_T_2024_TAIRTE4_SUBSTITUTION"].layers].index("inverse_T_resonator")
        < [layer.name for layer in contracts["B_T_2024_TAIRTE4_SUBSTITUTION"].layers].index("active_2d"),
        "2024_spacer_is_below_active_layer": [layer.name for layer in contracts["B_T_2024_TAIRTE4_SUBSTITUTION"].layers].index("active_2d")
        < [layer.name for layer in contracts["B_T_2024_TAIRTE4_SUBSTITUTION"].layers].index("cavity_spacer"),
        "2022_TaIrTe4_is_transferred_above_Z": [layer.name for layer in contracts["B_Z_2022_TAIRTE4_SUBSTITUTION"].layers].index("active_2d")
        < [layer.name for layer in contracts["B_Z_2022_TAIRTE4_SUBSTITUTION"].layers].index("chiral_Z_resonator"),
        "200nm_Au_bulk_propagation_factor_lt_1e_7": attenuation[3]["bulk_intensity_propagation_factor"] < 1.0e-7,
        "reduced_thermal_substrates_remain_unvalidated": all(
            item["status"] == "UNVALIDATED_REDUCED_THERMAL_SUBSTRATE_CANDIDATE"
            for item in reduced.values()
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Contract audit failed: {gates}")

    layer_rows = []
    for key, item in contracts.items():
        for layer in item.layers:
            row = {"architecture": key, **layer.__dict__}
            layer_rows.append(row)
    _write_csv(RESULTS / "paper_architecture_layers.csv", layer_rows)
    _write_csv(RESULTS / "z_metamaterial_published_dimensions.csv", list(Z_PUBLISHED_DIMENSIONS_NM))
    _write_csv(RESULTS / "optical_backplane_attenuation.csv", attenuation)
    _make_plot(contracts, RESULTS / "paper_architecture_geometry.png")

    payload = {
        "status": "VALIDATED_PAPER_ARCHITECTURE_CONTRACT_OFFLINE",
        "scope": "offline paper/geometry/provenance audit; no Maxwell, thermal, PTE, adjoint, or optimization solve",
        "contracts": {key: value.as_dict() for key, value in contracts.items()},
        "published_Z_dimensions_nm": list(Z_PUBLISHED_DIMENSIONS_NM),
        "proposed_10um_seed_sweeps": proposed_10um_seed_sweeps(),
        "optical_backplane_bulk_propagation_diagnostic": attenuation,
        "thermal_reduction_candidates": reduced,
        "paper_2022_thermal_note": {
            "recorded_statement": "10 W/m2 heat flux across several solid interfaces",
            "interpretation": "reported flux boundary, not an interfacial conductance G",
            "promoted_as_G": False,
        },
        "gates": gates,
        "pdfs": pdfs,
    }
    summary_path = RESULTS / "paper_architecture_contract.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = f"""# Paper-derived Au/TaIrTe4 architecture contract

Status: `VALIDATED_PAPER_ARCHITECTURE_CONTRACT_OFFLINE`

This checkpoint reads the official 2024 and 2022 supplements and changes only
the active 2-D thermoelectric material to the project's fixed 100-nm TaIrTe4
flake. It does **not** claim a Maxwell, thermal, PTE, adjoint, or optimization
result.

## Substrate decision

- `A_DIRECT_AU_TAIRTE4`: SiO2/Si cannot be removed optically without an
  endpoint-equivalence test because there is no opaque Au mirror.
- `B_T_2024_TAIRTE4_SUBSTITUTION`: SiO2/Si below the Au mirror may be omitted
  from the optical domain after a thickness/PML convergence test.
- `B_Z_2022_TAIRTE4_SUBSTITUTION`: the published 200-nm Au backplate likewise
  permits optical truncation below the metal.
- Thermal SiO2/Si is retained in the explicit reference. The reported reduced
  Robin values are screening candidates only and omit semi-infinite lateral
  spreading.

At 10 um, Ordal Au with k=69.2 has an intensity skin depth of
`{attenuation[3]['intensity_skin_depth_nm']:.6f} nm`. The 200-nm bulk
propagation factor is `{attenuation[3]['bulk_intensity_propagation_factor']:.6e}`.
This is not a replacement for the pending numerical backplane convergence.

## Important geometry corrections

- In the 2024 T architecture, the Ti/Au resonator touches the active 2-D layer;
  the Al2O3 cavity spacer is below that layer and above the Au mirror.
- In the 2022 Z architecture, the Au/Cr antenna chip is fabricated first and
  the active 2-D material is dry-transferred over it.
- T and Z are not interchangeable plan-view masks in a common stack.
- Published Z dimensions end at 8 um. The stored 10-um sweep is explicitly a
  numerical initialization, not a paper value.
- The 2022 paper's `10 W/m2` interface statement is a heat flux, not `G=10
  W/(m2 K)`; it is not promoted as an interface conductance.
"""
    (RESULTS / "PAPER_ARCHITECTURE_CONTRACT_REPORT.md").write_text(report, encoding="utf-8")

    artifacts = {}
    for path in sorted(RESULTS.iterdir()):
        if path.name == "RAW_ARTIFACT_MANIFEST.json" or not path.is_file():
            continue
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "status": payload["status"],
        "raw_FSP_or_NPZ_committed": False,
        "inputs": pdfs,
        "artifacts": artifacts,
    }
    (RESULTS / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

