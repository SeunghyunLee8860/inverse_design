#!/usr/bin/env python3
"""Selected-grid scalar absorbed-power AD--FD control.

This reuses the completed baseline and +/- forward solves.  It runs exactly
one GPU Maxwell adjoint and no thermal solve, so that weighted-source errors
can be separated from the underlying Maxwell/material adjoint convention.
"""

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
    absorption_objective_and_source,
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
        choices=("exact_inverse", "common_grid"),
        default="common_grid",
    )
    parser.add_argument(
        "--forward-source-state",
        choices=("disabled", "zero_amplitude_enabled"),
        default="disabled",
    )
    args = parser.parse_args()

    root = args.combined_directory.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "selected_scalar_pq_adjoint_result.json"
    result: dict[str, object] = {
        "status": "FAILED_SELECTED_SCALAR_PQ_ADJOINT_CONTROL",
        "passed": False,
        "Maxwell_forward_solves": 0,
        "Maxwell_adjoint_solves": 0,
        "thermal_solves": 0,
        "optimizer_started": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "CPU_FDTD_fallback": False,
    }
    fdtd = None
    try:
        combined_path = root / "production_combined_adfd_smoke_result.json"
        combined = json.loads(combined_path.read_text())
        raw_path = Path(combined["raw_artifact"]["path"])
        if sha256(raw_path) != combined["raw_artifact"]["sha256"]:
            raise RuntimeError("combined raw SHA mismatch")
        raw = np.load(raw_path)
        direction = np.asarray(raw["direction"], float)
        operator, rho, operator_meta = load_operator(
            args.jacobian_directory,
            "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN",
        )
        if not np.array_equal(rho, np.asarray(raw["rho"], float)):
            raise RuntimeError("operator/combined density mismatch")

        step = float(combined["step"])
        plus = float(combined["FD_pair"]["plus"]["forward"]["P_Q_W"])
        minus = float(combined["FD_pair"]["minus"]["forward"]["P_Q_W"])
        finite_difference = (plus - minus) / (2.0 * step)
        base_project = Path(combined["base_forward"]["project"]["path"])
        if sha256(base_project) != combined["base_forward"]["project"]["sha256"]:
            raise RuntimeError("base FSP SHA mismatch")

        fdtd, audit, runtime = open_fdtd(args.gpu_device)
        fdtd.load(str(base_project))
        forward, grid = monitor_electric(fdtd, PABS_FIELD)
        detail = index_detail(fdtd)
        epsilon = np.stack(
            [detail[f"epsilon_{component}"] for component in "xyz"], axis=-1
        )[..., None, :]
        if epsilon.shape != forward.shape:
            raise RuntimeError("forward E/index shape mismatch")
        objective, native_source, component_power = absorption_objective_and_source(
            forward, epsilon, grid
        )

        template = output / "selected_scalar_pq_adjoint_template.fsp"
        if args.source_mode == "exact_inverse":
            profile_scale, base_amplitude, source_meta = prepare_solver_aligned_source(
                fdtd,
                audit,
                base_project=base_project,
                grid=grid,
                native_source=native_source,
                template=template,
            )
        else:
            profile, profile_scale = fieldregion_profile(native_source)
            fdtd.load(str(base_project))
            fdtd.switchtolayout()
            original_amplitude = float(fdtd.getnamed(audit.SOURCE_NAME, "amplitude"))
            if args.forward_source_state == "disabled":
                fdtd.setnamed(audit.SOURCE_NAME, "enabled", False)
            else:
                fdtd.setnamed(audit.SOURCE_NAME, "amplitude", 0.0)
                fdtd.setnamed(audit.SOURCE_NAME, "enabled", True)
            for axis in "xyz":
                fdtd.setnamed(FIELD_REGION, f"{axis} min", float(grid[axis][0]))
                fdtd.setnamed(FIELD_REGION, f"{axis} max", float(grid[axis][-1]))
            fdtd.setnamed(FIELD_REGION, "source mode", True)
            try:
                fdtd.setnamed(FIELD_REGION, "nuttall window pulse", False)
            except Exception:
                pass
            roundtrip = import_named_fieldregion_profile(
                fdtd, FIELD_REGION, grid, profile
            )
            base_amplitude = float(fdtd.getnamed(FIELD_REGION, "base amplitude"))
            fdtd.save(str(template))
            source_meta = {
                "method": (
                    "official common-grid FieldRegion vector profile; native component "
                    "arrays are supplied on the monitor common coordinates without "
                    "inverse deconvolution"
                ),
                "component_collocation": {
                    "method": "official common-grid FieldRegion placement",
                    "components": {},
                },
                "source_profile_roundtrip_max_abs_error": roundtrip,
                "profile_scale": profile_scale,
                "fieldregion_base_amplitude": base_amplitude,
                "template": {
                    "path": str(template),
                    "size_bytes": template.stat().st_size,
                    "sha256": sha256(template),
                },
                "forward_Gaussian_source_object_preserved_as_mesh_anchor": True,
                "forward_Gaussian_source_original_amplitude": original_amplitude,
                "forward_Gaussian_source_state": args.forward_source_state,
                "forward_Gaussian_source_enabled_in_adjoint": bool(
                    fdtd.getnamed(audit.SOURCE_NAME, "enabled")
                ),
                "forward_Gaussian_source_adjoint_amplitude": float(
                    fdtd.getnamed(audit.SOURCE_NAME, "amplitude")
                ),
                "empirical_normalization": False,
                "gradient_rescaling": False,
            }
        adjoint = run_adjoint(
            fdtd,
            audit,
            runtime,
            template=template,
            project=output / "selected_scalar_pq_adjoint_gpu.fsp",
        )
        result["Maxwell_adjoint_solves"] = 1
        mismatch = max(
            float(
                np.max(
                    np.abs(np.asarray(grid[key]) - np.asarray(adjoint["grid"][key]))
                )
            )
            for key in ("x", "y", "z", "delta_x", "delta_y", "delta_z")
        )
        volumes = component_volumes(grid)
        indirect_cotangent = {}
        direct_cotangent = {}
        omega = 2.0 * np.pi * FREQUENCY_HZ
        for index, component in enumerate("xyz"):
            field = forward[..., 0, index]
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
                * omega
                * volumes[index]
                * np.abs(field) ** 2
            )
        indirect = operator.vjp(indirect_cotangent)
        direct = operator.vjp(direct_cotangent)
        total = indirect + direct
        indirect_directional = float(np.sum(indirect * direction))
        direct_directional = float(np.sum(direct * direction))
        adjoint_directional = indirect_directional + direct_directional
        error = relative(adjoint_directional, finite_difference)
        reconstruction = max(
            (
                float(row["reconstruction_max_abs_error"])
                for row in source_meta["component_collocation"]["components"].values()
            ),
            default=0.0,
        )
        passed = bool(
            error < 0.01
            and mismatch < 2.0e-18
            and reconstruction < 1.0e-12
            and source_meta["source_profile_roundtrip_max_abs_error"] == 0.0
            and float(adjoint["log_audit"]["final_auto_shutoff"]) < 1.0e-5
        )
        gradient_path = output / "selected_scalar_pq_gradient.npz"
        np.savez_compressed(
            gradient_path,
            rho=rho,
            direction=direction,
            gradient_indirect_W=indirect,
            gradient_direct_W=direct,
            gradient_total_W=total,
        )
        result.update(
            {
                "status": (
                    "VALIDATED_SELECTED_SCALAR_PQ_ADJOINT_CONTROL"
                    if passed
                    else "FAILED_SELECTED_SCALAR_PQ_ADJOINT_CONTROL"
                ),
                "passed": passed,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "P_Q_objective_W": objective,
                "P_Q_reported_W": combined["base_forward"]["P_Q_W"],
                "P_Q_objective_relative_difference": relative(
                    objective, float(combined["base_forward"]["P_Q_W"])
                ),
                "component_power_W": component_power,
                "operator": operator_meta,
                "source_method": source_meta,
                "source_mode": args.source_mode,
                "forward_source_state": args.forward_source_state,
                "directional_derivatives_W": {
                    "indirect_field_mediated": indirect_directional,
                    "direct_material_loss": direct_directional,
                    "total_AD": adjoint_directional,
                    "total_FD_reused": finite_difference,
                    "P_Q_plus": plus,
                    "P_Q_minus": minus,
                    "step": step,
                },
                "AD_FD_relative_error": error,
                "gates": {
                    "AD_FD_relative_error": error,
                    "AD_FD_limit": 0.01,
                    "collocation_reconstruction_max_abs_error": reconstruction,
                    "source_profile_roundtrip_max_abs_error": source_meta[
                        "source_profile_roundtrip_max_abs_error"
                    ],
                    "forward_adjoint_coordinate_mismatch_m": mismatch,
                    "adjoint_auto_shutoff": adjoint["log_audit"][
                        "final_auto_shutoff"
                    ],
                },
                "adjoint": {
                    key: value
                    for key, value in adjoint.items()
                    if key not in {"electric", "grid"}
                },
                "artifacts": {
                    "combined_result": {
                        "path": str(combined_path),
                        "sha256": sha256(combined_path),
                    },
                    "base_FSP": {
                        "path": str(base_project),
                        "sha256": sha256(base_project),
                    },
                    "gradient_NPZ": {
                        "path": str(gradient_path),
                        "size_bytes": gradient_path.stat().st_size,
                        "sha256": sha256(gradient_path),
                    },
                },
            }
        )
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
