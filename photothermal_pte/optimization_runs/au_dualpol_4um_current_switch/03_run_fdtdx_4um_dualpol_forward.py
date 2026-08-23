#!/usr/bin/env python3
"""Run the first real 4 um Ea/Eb optical gate at uniform rho=0.5.

This is an optical-only checkpoint.  It neither solves the thermal/electrical
systems nor evaluates an adjoint or changes the design.  Raw spatial arrays
are kept outside Git and are referenced by SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    ABSORPTION_LOSS_BASIS,
    build_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
    au_material_fraction,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import (
    raw_root,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_fdtdx_4um_dualpol_forward"
RAW = raw_root()
EPS0_F_PER_M = 8.8541878128e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _electric_yee_dual_volumes(grid, grid_slice: tuple[slice, slice, slice]):
    """Component-specific Ex/Ey/Ez physical dual volumes."""

    widths = [np.asarray(grid.cell_widths(axis), dtype=np.float64) for axis in range(3)]
    edge_dual = [
        0.5 * (np.concatenate((value[:1], value[:-1])) + value)
        for value in widths
    ]
    bounds = tuple((int(part.start), int(part.stop)) for part in grid_slice)
    volumes = []
    for component in range(3):
        selected = []
        for axis, (lower, upper) in enumerate(bounds):
            metric = widths[axis] if axis == component else edge_dual[axis]
            selected.append(metric[lower:upper])
        volumes.append(
            selected[0][:, None, None]
            * selected[1][None, :, None]
            * selected[2][None, None, :]
        )
    return np.stack(volumes)


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


def _field_change(late: np.ndarray, previous: np.ndarray) -> float:
    return float(
        np.linalg.norm(late.ravel() - previous.ravel())
        / max(np.linalg.norm(late.ravel()), np.finfo(float).tiny)
    )


def _centroid_and_waist(intensity: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    weights = np.maximum(np.asarray(intensity, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if not total > 0.0:
        raise RuntimeError("non-positive target-plane field intensity")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cx = float(np.sum(weights * xx) / total)
    cy = float(np.sum(weights * yy) / total)
    var_x = float(np.sum(weights * (xx - cx) ** 2) / total)
    var_y = float(np.sum(weights * (yy - cy) ** 2) / total)
    # I=I0 exp[-2 x^2/w^2] has variance w^2/4.
    return {
        "center_x_m": cx,
        "center_y_m": cy,
        "second_moment_waist_x_m": 2.0 * math.sqrt(max(var_x, 0.0)),
        "second_moment_waist_y_m": 2.0 * math.sqrt(max(var_y, 0.0)),
    }


def run_case(polarization: str) -> tuple[dict[str, object], Path]:
    model = build_model(polarization, include_adjoint_source=False)
    jnp = model["jnp"]
    fdtdx = model["fdtdx"]
    rho = jnp.full(CONTRACT.design_shape, 0.5, dtype=jnp.float32)
    strength = au_material_fraction(rho)
    c3 = model["fixed_c3"]
    au_slice = model["slices"]["au_design"]
    au_c3 = float(model["coefficients"]["au"][2])
    for component in range(3):
        c3 = c3.at[(0, component, *au_slice)].set(
            au_c3 * strength[:, :, None]
        )
    arrays = (
        model["base"]
        .reset()
        .aset("dispersive_c1", model["fixed_c1"])
        .aset("dispersive_c2", model["fixed_c2"])
        .aset("dispersive_c3", c3)
    )
    start = time.perf_counter()
    _, output = fdtdx.run_fdtd(
        arrays, model["placed"], model["config"], model["key"], show_progress=False
    )
    runtime_s = time.perf_counter() - start

    eta0 = float(fdtdx.constants.eta0)
    physical_prefactor = (
        0.5 * model["omega_rad_s"] * EPS0_F_PER_M * eta0**2
    )
    volumes = {
        material: _electric_yee_dual_volumes(
            model["grid"], model["slices"][material]
        )
        for material in ("au_design", "fixed_tairte4")
    }
    au_imag = float(model["discrete_susceptibility"]["au"].imag)
    ta_imag = np.asarray(
        [
            model["discrete_susceptibility"][axis].imag
            for axis in ("b", "a", "c")
        ],
        dtype=np.float64,
    )[:, None, None, None]
    material_fields: dict[str, dict[str, np.ndarray]] = {}
    component_power: dict[str, dict[str, list[float]]] = {}
    for window in ("previous", "late"):
        e_au = np.asarray(output.detector_states[f"au_{window}"]["phasor"][0, 0])
        e_ta = np.asarray(output.detector_states[f"tairte4_{window}"]["phasor"][0, 0])
        q_au = (
            physical_prefactor
            * au_imag
            * np.asarray(strength)[None, :, :, None]
            * np.abs(e_au) ** 2
        )
        q_ta = physical_prefactor * ta_imag * np.abs(e_ta) ** 2
        material_fields[window] = {
            "e_au": e_au,
            "e_ta": e_ta,
            "q_au": q_au,
            "q_ta": q_ta,
        }
        component_power[window] = {
            "au_W": [
                float(np.sum(q_au[c] * volumes["au_design"][c])) for c in range(3)
            ],
            "tairte4_W": [
                float(np.sum(q_ta[c] * volumes["fixed_tairte4"][c]))
                for c in range(3)
            ],
        }
    for window in component_power:
        values = component_power[window]
        values["total_W"] = float(sum(values["au_W"]) + sum(values["tairte4_W"]))

    p_closed_phasor = float(
        eta0
        * np.asarray(
            model["placed"]["material_flux"].compute_net_flux(
                output.detector_states["material_flux"]
            )
        )[0]
    )
    p_closed_td = float(
        eta0
        * np.mean(
            np.asarray(
                output.detector_states["material_flux_td"]["poynting_flux"]
            )[:, 0]
        )
    )
    p_incident = float(
        eta0
        * np.asarray(
            model["placed"]["incident_plane"].compute_poynting_flux(
                output.detector_states["incident_plane"]
            )
        )[0]
    )
    target = np.asarray(output.detector_states["target_field"]["phasor"][0, 0])
    target_intensity = np.sum(np.abs(target[:, :, :, 0]) ** 2, axis=0)
    target_slice = model["slices"]["target_field"]
    x_edges = np.asarray(model["grid"].edges(0), dtype=float)
    y_edges = np.asarray(model["grid"].edges(1), dtype=float)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])[target_slice[0]]
    y = 0.5 * (y_edges[:-1] + y_edges[1:])[target_slice[1]]
    beam = _centroid_and_waist(target_intensity, x, y)

    late = component_power["late"]
    previous = component_power["previous"]
    closure = _relative(late["total_W"], p_closed_td)
    closure_phasor = _relative(late["total_W"], p_closed_phasor)
    q_power_change = _relative(late["total_W"], previous["total_W"])
    q_field_change = float(
        math.sqrt(
            np.linalg.norm(
                material_fields["late"]["q_au"].ravel()
                - material_fields["previous"]["q_au"].ravel()
            ) ** 2
            + np.linalg.norm(
                material_fields["late"]["q_ta"].ravel()
                - material_fields["previous"]["q_ta"].ravel()
            ) ** 2
        )
        / max(
            math.sqrt(
                np.linalg.norm(material_fields["late"]["q_au"].ravel()) ** 2
                + np.linalg.norm(material_fields["late"]["q_ta"].ravel()) ** 2
            ),
            np.finfo(float).tiny,
        )
    )
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            material_fields["late"]["q_au"],
            material_fields["late"]["q_ta"],
            target,
        )
    )
    nonnegative = bool(
        np.min(material_fields["late"]["q_au"]) >= 0.0
        and np.min(material_fields["late"]["q_ta"]) >= 0.0
    )
    gates = {
        "six_face_closure_lt_0p5pct": closure < 0.005,
        "phasor_Q_power_change_lt_0p5pct": q_power_change < 0.005,
        "phasor_spatial_Q_change_lt_0p5pct": q_field_change < 0.005,
        "finite_fields": finite,
        "nonnegative_Q": nonnegative,
    }

    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"fdtdx_4um_rho0p5_{polarization}.npz"
    np.savez_compressed(
        raw_path,
        rho=np.asarray(rho),
        q_au_W_m3=material_fields["late"]["q_au"],
        q_tairte4_W_m3=material_fields["late"]["q_ta"],
        q_au_previous_W_m3=material_fields["previous"]["q_au"],
        q_tairte4_previous_W_m3=material_fields["previous"]["q_ta"],
        volume_au_m3=volumes["au_design"],
        volume_tairte4_m3=volumes["fixed_tairte4"],
        target_field=target,
        target_x_m=x,
        target_y_m=y,
    )
    case = {
        "polarization": polarization,
        "status": (
            "VALIDATED_FDTDX_4UM_RHO0P5_FORWARD"
            if all(gates.values())
            else "FAILED_FDTDX_4UM_RHO0P5_FORWARD"
        ),
        "rho": 0.5,
        "au_material_fraction": material_fraction_audit(),
        "absorption_loss_basis": model["absorption_loss_basis"],
        "realized_discrete_susceptibility": {
            name: [value.real, value.imag]
            for name, value in model["discrete_susceptibility"].items()
        },
        "runtime_s": runtime_s,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "power": {
            "previous": previous,
            "late": late,
            "incident_plane_signed_W": p_incident,
            "six_face_inward_td_mean_signed_W": p_closed_td,
            "six_face_inward_phasor_signed_W": p_closed_phasor,
            "six_face_closure_relative": closure,
            "six_face_phasor_closure_relative": closure_phasor,
            "phasor_Q_power_change_relative": q_power_change,
            "phasor_spatial_Q_change_relative": q_field_change,
        },
        "target_total_field_diagnostic": beam,
        "gates": gates,
        "raw": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": _sha256(raw_path),
        },
        "prohibitions": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "rescaling": False,
        },
    }
    return case, raw_path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for polarization in ("Ea", "Eb"):
        print(f"[{polarization}] starting real FDTDX forward", flush=True)
        case, _ = run_case(polarization)
        cases.append(case)
        print(
            f"[{polarization}] status={case['status']} runtime={case['runtime_s']:.2f}s "
            f"closure={100*case['power']['six_face_closure_relative']:.4f}%",
            flush=True,
        )

    summary_status = (
        "VALIDATED_FDTDX_4UM_DUALPOL_RHO0P5_FORWARD"
        if all(case["status"].startswith("VALIDATED_") for case in cases)
        else "FAILED_FDTDX_4UM_DUALPOL_RHO0P5_FORWARD"
    )
    summary = {
        "status": summary_status,
        "scope": "rho=0.5 optical forward only; no thermal/electrical/adjoint/optimization",
        "axis_mapping": {"x": "b", "y": "a"},
        "au_material_fraction": material_fraction_audit(),
        "absorption_loss_basis": ABSORPTION_LOSS_BASIS,
        "cases": cases,
    }
    (OUT / "fdtdx_4um_dualpol_forward.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "fdtdx_4um_dualpol_forward.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "polarization", "status", "runtime_s", "P_Q_W", "P_closed_W",
                "closure_relative", "Q_window_change_relative", "Q_field_change_relative",
                "P_Au_W", "P_TaIrTe4_W", "raw_sha256",
            ]
        )
        for case in cases:
            late = case["power"]["late"]
            writer.writerow(
                [
                    case["polarization"], case["status"], case["runtime_s"],
                late["total_W"], case["power"]["six_face_inward_td_mean_signed_W"],
                    case["power"]["six_face_closure_relative"],
                    case["power"]["phasor_Q_power_change_relative"],
                    case["power"]["phasor_spatial_Q_change_relative"],
                    sum(late["au_W"]), sum(late["tairte4_W"]), case["raw"]["sha256"],
                ]
            )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row, case in enumerate(cases):
        with np.load(case["raw"]["path"], allow_pickle=False) as raw:
            q_au = np.asarray(raw["q_au_W_m3"]).sum(axis=(0, 3))
            q_ta = np.asarray(raw["q_tairte4_W_m3"]).sum(axis=(0, 3))
            target = np.sum(np.abs(np.asarray(raw["target_field"])[:, :, :, 0]) ** 2, axis=0)
        for ax, field, title in zip(
            axes[row],
            (q_au, q_ta, target),
            ("Au component-sum Q", "TaIrTe4 component-sum Q", "target total |E|^2"),
            strict=True,
        ):
            image = ax.imshow(field.T, origin="lower", cmap="inferno", extent=(-4, 4, -4, 4) if field.shape[0] == 80 else (-8, 8, -8, 8))
            ax.set(title=f"{case['polarization']}: {title}", xlabel="x=b (um)", ylabel="y=a (um)", aspect="equal")
            fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle("4 um uniform-rho=0.5 dual-polarization optical forward gate")
    fig.savefig(OUT / "FDTDX_4UM_DUALPOL_RHO0P5_FIELDS.png", dpi=180)
    plt.close(fig)

    lines = [
        "# FDTDX 4 um dual-polarization rho=0.5 forward gate",
        "",
        f"Status: **{summary_status}**",
        "",
        "This checkpoint ran two real Maxwell forward solves on the identical finite six-PML grid.",
        "It did not solve thermal, weighting, current, adjoint, or optimization problems.",
        "The target-plane field is a total-field diagnostic and is not labelled a pure incident beam.",
        "",
        "| polarization | P_Q (W) | P_Au (W) | P_TaIrTe4 (W) | closure | Q window change | runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        late = case["power"]["late"]
        lines.append(
            f"| {case['polarization']} | {late['total_W']:.8e} | "
            f"{sum(late['au_W']):.8e} | {sum(late['tairte4_W']):.8e} | "
            f"{100*case['power']['six_face_closure_relative']:.5f}% | "
            f"{100*case['power']['phasor_Q_power_change_relative']:.5f}% | "
            f"{case['runtime_s']:.2f} s |"
        )
    (OUT / "FDTDX_4UM_DUALPOL_RHO0P5_FORWARD.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary_status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
