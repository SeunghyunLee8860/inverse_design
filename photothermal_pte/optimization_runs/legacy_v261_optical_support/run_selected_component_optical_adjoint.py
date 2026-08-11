#!/usr/bin/env python3
"""Run one exact component-Yee weighted optical adjoint on GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
for path in (HERE, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import PABS_FIELD  # noqa: E402
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    component_volumes,
    fieldregion_profile,
    import_named_fieldregion_profile,
    monitor_electric,
)
from build_nonuniform_complex_yee_jacobian import index_detail  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    FIELD_REGION,
    FREQUENCY_HZ,
    load_operator,
    open_fdtd,
    run_adjoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=tuple("xyz"), required=True)
    parser.add_argument("--combined-directory", type=Path, required=True)
    parser.add_argument("--jacobian-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--gpu-device", required=True)
    args = parser.parse_args()
    component = args.component
    component_index = "xyz".index(component)
    source_root = args.combined_directory.resolve()
    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"selected_component_{component}_adjoint_result.json"
    result: dict[str, object] = {
        "status": f"FAILED_SELECTED_COMPONENT_{component.upper()}_OPTICAL_ADJOINT",
        "passed": False,
        "Maxwell_forward_solves": 0,
        "Maxwell_adjoint_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
    }
    fdtd = None
    try:
        failed = json.loads(
            (source_root / "production_combined_adfd_smoke_result.json").read_text()
        )
        raw_path = Path(failed["raw_artifact"]["path"])
        if sha256(raw_path) != failed["raw_artifact"]["sha256"]:
            raise RuntimeError("combined raw SHA mismatch")
        raw = np.load(raw_path)
        direction = np.asarray(raw["direction"], float)
        operator, rho, operator_meta = load_operator(
            args.jacobian_directory,
            "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
        )
        if not np.array_equal(rho, np.asarray(raw["rho"], float)):
            raise RuntimeError("density mismatch")
        base_project = Path(failed["base_forward"]["project"]["path"])
        if sha256(base_project) != failed["base_forward"]["project"]["sha256"]:
            raise RuntimeError("base FSP SHA mismatch")
        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        fdtd.load(str(base_project))
        forward, grid = monitor_electric(fdtd, PABS_FIELD)
        detail = index_detail(fdtd)
        epsilon = np.stack(
            [detail[f"epsilon_{axis}"] for axis in "xyz"], axis=-1
        )[..., None, :]
        pulled = np.asarray(
            raw[f"native_Q{component}_density_sensitivity_A_m3_W"], float
        )
        native_source = np.zeros_like(forward, complex)
        native_source[..., 0, component_index] = (
            0.5
            * EPS0
            * (2.0 * np.pi * FREQUENCY_HZ)
            * np.imag(epsilon[..., 0, component_index])
            * pulled
            * forward[..., 0, component_index]
        )
        profile, profile_scale = fieldregion_profile(native_source)
        source_grid = {key: np.array(value, copy=True) for key, value in grid.items()}
        source_grid[component] = (
            np.asarray(grid[component], float)
            + np.asarray(grid[f"delta_{component}"], float)
        )
        raw_mesh = fdtd.getresult("FDTD", component)
        if isinstance(raw_mesh, dict):
            raw_mesh = raw_mesh[component]
        solver_mesh = np.asarray(raw_mesh, float).reshape(-1)
        common_axis = np.asarray(grid[component], float)
        after = solver_mesh[solver_mesh > common_axis[-1] + 2.0e-18]
        if after.size == 0:
            raise RuntimeError(f"no solver mesh plane after {component} source support")
        fieldregion_bounds = {
            axis: [float(source_grid[axis][0]), float(source_grid[axis][-1])]
            for axis in "xyz"
        }
        # The nonzero vector component is sampled at cell centers along its
        # staggered axis. FieldRegion geometry must therefore use the mesh-node
        # faces enclosing those centers, while the imported dataset retains
        # the exact native Yee center coordinates.
        fieldregion_bounds[component] = [
            float(common_axis[0]),
            float(after[0]),
        ]
        fdtd.switchtolayout()
        original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
        fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
        if not bool(fdtd.getnamed(audit.SOURCE_NAME, "enabled")):
            raise RuntimeError("forward Gaussian mesh anchor is disabled")
        for axis in "xyz":
            fdtd.setnamed(FIELD_REGION, f"{axis} min", fieldregion_bounds[axis][0])
            fdtd.setnamed(FIELD_REGION, f"{axis} max", fieldregion_bounds[axis][1])
        fdtd.setnamed(FIELD_REGION, "source mode", True)
        try:
            fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
        except Exception:
            pass
        roundtrip = import_named_fieldregion_profile(
            fdtd, FIELD_REGION, source_grid, profile
        )
        base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
        template = output / f"selected_component_{component}_adjoint_template.fsp"
        fdtd.save(str(template))
        adjoint = run_adjoint(
            fdtd,
            audit,
            runtime,
            template=template,
            project=output / f"selected_component_{component}_adjoint_gpu.fsp",
        )
        mismatch = max(
            float(
                np.max(
                    np.abs(np.asarray(grid[key]) - np.asarray(adjoint["grid"][key]))
                )
            )
            for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
        )
        volumes = component_volumes(grid)
        cotangent = {}
        component_terms = {}
        jvp = operator.jvp(direction)
        for index, material_component in enumerate("xyz"):
            cotangent[material_component] = (
                2.0
                * EPS0
                * volumes[index]
                * forward[..., 0, index]
                * (
                    adjoint["electric"][..., 0, index]
                    * profile_scale
                    / base_amplitude
                )
            )
            component_terms[material_component] = float(
                np.real(
                    np.sum(
                        cotangent[material_component]
                        * jvp[material_component]
                    )
                )
            )
        gradient = operator.vjp(cotangent)
        directional = float(np.sum(gradient * direction))
        explicit = float(sum(component_terms.values()))
        raw_out = output / f"selected_component_{component}_indirect_gradient.npz"
        np.savez_compressed(raw_out, gradient_A=gradient)
        passed = bool(
            mismatch < 2.0e-18
            and roundtrip == 0.0
            and abs(directional - explicit)
            / max(abs(directional), abs(explicit), np.finfo(float).tiny)
            < 1.0e-12
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
        )
        result = {
            "status": (
                f"COMPLETED_SELECTED_COMPONENT_{component.upper()}_OPTICAL_ADJOINT"
                if passed
                else f"FAILED_SELECTED_COMPONENT_{component.upper()}_OPTICAL_ADJOINT"
            ),
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_component": component,
            "source": {
                "method": "single nonzero FieldRegion vector component on its native Yee coordinates",
                "coordinate_bounds_m": {
                    axis: [float(source_grid[axis][0]), float(source_grid[axis][-1])]
                    for axis in "xyz"
                },
                "fieldregion_geometry_bounds_m": fieldregion_bounds,
                "profile_scale": profile_scale,
                "base_amplitude": base_amplitude,
                "roundtrip_max_abs_error": roundtrip,
                "forward_Gaussian_preserved_as_zero_amplitude_mesh_anchor": True,
                "forward_Gaussian_original_amplitude": original_amplitude,
                "empirical_normalization": False,
                "gradient_rescaling": False,
            },
            "operator": operator_meta,
            "indirect_directional_A": directional,
            "indirect_gradient_L2_A": float(np.linalg.norm(gradient)),
            "material_component_directional_terms_A": component_terms,
            "forward_adjoint_coordinate_mismatch_m": mismatch,
            "adjoint": {
                key: value
                for key, value in adjoint.items()
                if key not in {"electric", "grid"}
            },
            "artifacts": {
                "template": {"path": str(template), "sha256": sha256(template)},
                "gradient_NPZ": {
                    "path": str(raw_out),
                    "size_bytes": raw_out.stat().st_size,
                    "sha256": sha256(raw_out),
                },
            },
            "Maxwell_forward_solves": 0,
            "Maxwell_adjoint_solves": 1,
            "thermal_solves": 0,
            "optimizer_started": False,
            "CPU_FDTD_fallback": False,
        }
    except Exception as exc:
        result.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
