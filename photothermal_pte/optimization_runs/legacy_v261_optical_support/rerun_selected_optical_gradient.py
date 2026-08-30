#!/usr/bin/env python3
"""Rerun only the corrected selected-grid weighted Maxwell adjoint."""

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import EPS0  # noqa: E402
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    component_volumes,
    fieldregion_profile,
    import_named_fieldregion_profile,
    monitor_electric,
    weighted_fieldregion_source_from_native_multiplier,
)
from build_nonuniform_complex_yee_jacobian import index_detail  # noqa: E402
from run_production_combined_adfd_smoke import (  # noqa: E402
    FIELD_REGION,
    FREQUENCY_HZ,
    load_operator,
    open_fdtd,
    prepare_solver_aligned_source,
    relative,
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
    parser.add_argument("--combined-directory", type=Path, required=True)
    parser.add_argument("--jacobian-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--gpu-device", default="GPU 0")
    parser.add_argument(
        "--source-mode",
        choices=("exact_inverse", "multiplier_common"),
        default="exact_inverse",
    )
    args = parser.parse_args()
    source_root = args.combined_directory.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "selected_corrected_optical_gradient_result.json"
    result: dict[str, object] = {
        "status": "FAILED_SELECTED_CORRECTED_OPTICAL_GRADIENT",
        "passed": False,
        "Maxwell_forward_solves": 0,
        "Maxwell_adjoint_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
    }
    fdtd = None
    try:
        failed_path = source_root / "production_combined_adfd_smoke_result.json"
        failed = json.loads(failed_path.read_text())
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
            raise RuntimeError("operator/combined density mismatch")
        decomposition_path = source_root / "selected_combined_failure_decomposition.json"
        decomposition = json.loads(decomposition_path.read_text())
        optical_fd = float(
            decomposition["directional_derivatives_A"][
                "FD_optical_fixed_thermal_material"
            ]
        )
        base_project = Path(failed["base_forward"]["project"]["path"])
        if sha256(base_project) != failed["base_forward"]["project"]["sha256"]:
            raise RuntimeError("base FSP SHA mismatch")

        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        fdtd.load(str(base_project))
        forward_electric, forward_grid = monitor_electric(fdtd, PABS_FIELD)
        detail = index_detail(fdtd)
        epsilon = np.stack(
            [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
        )[..., None, :]
        if epsilon.shape != forward_electric.shape:
            raise RuntimeError("forward E/index shape mismatch")
        pulled = {
            component: np.asarray(
                raw[f"native_Q{component}_density_sensitivity_A_m3_W"], float
            )
            for component in "xyz"
        }
        coefficient = np.zeros((*forward_electric.shape[:3], 3), float)
        native_source = np.zeros_like(forward_electric, complex)
        for index, component in enumerate("xyz"):
            coefficient[..., index] = pulled[component]
            native_source[..., 0, index] = (
                0.5
                * EPS0
                * (2.0 * np.pi * FREQUENCY_HZ)
                * np.imag(epsilon[..., 0, index])
                * pulled[component]
                * forward_electric[..., 0, index]
            )
        template = output / "selected_corrected_optical_adjoint_template.fsp"
        if args.source_mode == "exact_inverse":
            profile_scale, base_amplitude, source_meta = prepare_solver_aligned_source(
                fdtd,
                audit,
                base_project=base_project,
                grid=forward_grid,
                native_source=native_source,
                template=template,
            )
        else:
            common_source, multiplier_meta = (
                weighted_fieldregion_source_from_native_multiplier(
                    electric=forward_electric,
                    epsilon=epsilon,
                    coefficient=coefficient,
                    grid=forward_grid,
                    frequency_hz=FREQUENCY_HZ,
                )
            )
            profile, profile_scale = fieldregion_profile(common_source)
            fdtd.load(str(base_project))
            fdtd.switchtolayout()
            original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
            fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
            if not bool(fdtd.getnamed(audit.SOURCE_NAME, "enabled")):
                raise RuntimeError("forward Gaussian mesh anchor is disabled")
            for axis in "xyz":
                fdtd.setnamed(FIELD_REGION, f"{axis} min", float(forward_grid[axis][0]))
                fdtd.setnamed(FIELD_REGION, f"{axis} max", float(forward_grid[axis][-1]))
            fdtd.setnamed(FIELD_REGION, "source mode", True)
            try:
                fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
            except Exception:
                pass
            roundtrip = import_named_fieldregion_profile(
                fdtd, FIELD_REGION, forward_grid, profile
            )
            base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
            fdtd.save(str(template))
            source_meta = {
                "method": "official FieldRegion E profile with native-Yee scalar multiplier interpolated to common coordinates",
                "component_collocation": multiplier_meta,
                "source_profile_roundtrip_max_abs_error": roundtrip,
                "profile_scale": profile_scale,
                "template": {
                    "path": str(template),
                    "size_bytes": template.stat().st_size,
                    "sha256": sha256(template),
                },
                "forward_Gaussian_source_object_preserved_as_mesh_anchor": True,
                "forward_Gaussian_source_original_amplitude": original_amplitude,
                "forward_Gaussian_source_adjoint_amplitude": 0.0,
                "empirical_normalization": False,
                "gradient_rescaling": False,
            }
        adjoint = run_adjoint(
            fdtd,
            audit,
            runtime,
            template=template,
            project=output / "selected_corrected_optical_adjoint_gpu.fsp",
        )
        mismatch = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(forward_grid[key])
                        - np.asarray(adjoint["grid"][key])
                    )
                )
            )
            for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
        )
        jvp = operator.jvp(direction)
        volumes = component_volumes(forward_grid)
        indirect_cotangent = {}
        direct_cotangent = {}
        components = {}
        for index, component in enumerate("xyz"):
            field = forward_electric[..., 0, index]
            adjoint_field = adjoint["electric"][..., 0, index]
            indirect_cotangent[component] = (
                (2.0 * EPS0 / base_amplitude)
                * volumes[index]
                * field
                * (adjoint_field * profile_scale)
            )
            direct_cotangent[component] = (
                -1j
                * 0.5
                * EPS0
                * (2.0 * np.pi * FREQUENCY_HZ)
                * pulled[component]
                * np.abs(field) ** 2
            )
            components[component] = {
                "indirect_directional_A": float(
                    np.real(np.sum(indirect_cotangent[component] * jvp[component]))
                ),
                "direct_directional_A": float(
                    np.real(np.sum(direct_cotangent[component] * jvp[component]))
                ),
            }
        indirect = operator.vjp(indirect_cotangent)
        direct = operator.vjp(direct_cotangent)
        total = indirect + direct
        indirect_directional = float(np.sum(indirect * direction))
        direct_directional = float(np.sum(direct * direction))
        optical_ad = indirect_directional + direct_directional
        optical_error = relative(optical_ad, optical_fd)
        reconstruction_values = [
            float(record["reconstruction_max_abs_error"])
            for record in source_meta["component_collocation"].get("components", {}).values()
            if "reconstruction_max_abs_error" in record
        ]
        reconstruction = max(reconstruction_values, default=0.0)
        passed = bool(
            optical_error < 0.01
            and mismatch < 2.0e-18
            and reconstruction < 1.0e-12
            and source_meta["source_profile_roundtrip_max_abs_error"] == 0.0
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
        )
        raw_out = output / "selected_corrected_optical_gradient.npz"
        np.savez_compressed(
            raw_out,
            rho=rho,
            direction=direction,
            gradient_indirect_A=indirect,
            gradient_direct_A=direct,
            gradient_total_optical_A=total,
        )
        result = {
            "status": (
                "VALIDATED_SELECTED_CORRECTED_OPTICAL_GRADIENT"
                if passed
                else "FAILED_SELECTED_CORRECTED_OPTICAL_GRADIENT"
            ),
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_method": source_meta,
            "source_mode": args.source_mode,
            "operator": operator_meta,
            "directional_derivatives_A": {
                "indirect_field_mediated": indirect_directional,
                "direct_material_loss": direct_directional,
                "total_optical_AD": optical_ad,
                "total_optical_FD_reused": optical_fd,
            },
            "optical_AD_FD_relative_error": optical_error,
            "component_directional_terms": components,
            "gradient_norms_A": {
                "indirect": float(np.linalg.norm(indirect)),
                "direct": float(np.linalg.norm(direct)),
                "total": float(np.linalg.norm(total)),
            },
            "gates": {
                "optical_AD_FD_relative_error": optical_error,
                "optical_AD_FD_limit": 0.01,
                "collocation_reconstruction_max_abs_error": reconstruction,
                "source_profile_roundtrip_max_abs_error": source_meta[
                    "source_profile_roundtrip_max_abs_error"
                ],
                "forward_adjoint_coordinate_mismatch_m": mismatch,
                "adjoint_auto_shutoff": adjoint["log_audit"]["final_auto_shutoff"],
            },
            "adjoint": {
                key: value
                for key, value in adjoint.items()
                if key not in {"electric", "grid"}
            },
            "artifacts": {
                "failed_checkpoint": {"path": str(failed_path), "sha256": sha256(failed_path)},
                "decomposition": {"path": str(decomposition_path), "sha256": sha256(decomposition_path)},
                "base_FSP": {"path": str(base_project), "sha256": sha256(base_project)},
                "corrected_gradient_NPZ": {
                    "path": str(raw_out),
                    "size_bytes": raw_out.stat().st_size,
                    "sha256": sha256(raw_out),
                },
            },
            "Maxwell_forward_solves": 0,
            "Maxwell_adjoint_solves": 1,
            "thermal_solves": 0,
            "optimizer_started": False,
            "empirical_normalization": False,
            "gradient_rescaling": False,
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
