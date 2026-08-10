#!/usr/bin/env python3
"""GPU-only source certificate for the compact TaIrTe4 optimization domain."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (  # noqa: E402
    CONTRACT,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as source_audit,
)


BASE_SETUP = source_audit.setup


def configure() -> None:
    """Replace only scenario constants in the already-audited source runner."""
    source_audit.contract = SimpleNamespace(
        WAVELENGTH_M=CONTRACT.wavelength_m,
        SELECTED_W0_M=CONTRACT.target_waist_m,
        SOURCE_SPAN_M=CONTRACT.source_span_m,
        LATERAL_DOMAIN_M=CONTRACT.optical_lateral_span_m,
        SOURCE_Z_M=CONTRACT.source_z_m,
        FOCUS_Z_M=CONTRACT.focus_z_m,
        FDTD_Z_MIN_M=CONTRACT.optical_z_min_m,
        FDTD_Z_MAX_M=CONTRACT.optical_z_max_m,
    )
    source_audit.TARGET_FREQUENCY_HZ = source_audit.C0 / CONTRACT.wavelength_m
    source_audit.SOURCE_START_M = 8.5e-6
    source_audit.SOURCE_STOP_M = 12.142857142857142e-6
    source_audit.PML_LAYERS = CONTRACT.pml_layers
    source_audit.MESH_ACCURACY = CONTRACT.mesh_accuracy
    source_audit.SOURCE_NAME = "run010_gaussian10_w8p5_source"
    source_audit.MONITORS = {
        "source_plane": CONTRACT.source_z_m - 0.5e-6,
        "flake_target_plane": CONTRACT.focus_z_m,
        "downstream_plane": -2.5e-6,
    }

    def compact_setup(*args, **kwargs):
        built = BASE_SETUP(*args, **kwargs)
        built["source"]["source_object_waist_calibration"] = {
            "method": "Run002 calibrated scalar-Gaussian source-object waist",
            "input_source_object_waist_m": CONTRACT.calibrated_source_object_waist_m,
            "target_plane_realized_waist": "MEASURED_BY_THIS_RUN",
            "legacy_11um_12um_calibration_reused": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
        }
        built["scope_note"] = (
            "homogeneous-air source only; no TaIrTe4, substrate, Q, thermal, "
            "electrical, adjoint, or optimization solve"
        )
        return built

    source_audit.setup = compact_setup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-device", default="GPU 5")
    parser.add_argument("--duration-ps", type=float, default=4.0)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-5)
    parser.add_argument("--threads", default="8")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    configure()
    forwarded = [
        sys.argv[0],
        "--output-dir", args.output_dir,
        "--duration-ps", str(args.duration_ps),
        "--auto-shutoff-min", str(args.auto_shutoff_min),
        "--gpu-device", args.gpu_device,
        "--threads", args.threads,
        "--mesh-accuracy", str(CONTRACT.mesh_accuracy),
        "--target-waist-um", str(CONTRACT.target_waist_m * 1e6),
        "--source-object-waist-um", str(CONTRACT.calibrated_source_object_waist_m * 1e6),
    ]
    if args.contract_only:
        forwarded.append("--contract-only")
    sys.argv = forwarded
    return source_audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
