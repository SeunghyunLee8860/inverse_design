#!/usr/bin/env python3
"""Run the existing v261 GPU source audit under the Run-002 contract.

The validated 11 um paper-sanity implementation is imported, not copied or
edited. Only its scenario constants are replaced before ``main`` is called.
This run contains homogeneous air: no material, Q, thermal, PTE, adjoint, or
optimization object is present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    validate_paper_ir_source_only_gpu as source_audit,
)

BASE_SETUP = source_audit.setup


WAVELENGTH_M = 10.0e-6
TARGET_W0_M = 8.5e-6
SOURCE_SPAN_M = 40.0e-6
LATERAL_DOMAIN_M = 48.0e-6
SOURCE_Z_M = 5.0e-6
FOCUS_Z_M = 0.0
FDTD_Z_MIN_M = -8.0e-6
FDTD_Z_MAX_M = 8.0e-6
SOURCE_START_M = 8.5e-6
SOURCE_STOP_M = 12.142857142857142e-6


def configure_source_audit() -> None:
    """Install the immutable Run-002 constants in the generic audit module."""
    source_audit.contract = SimpleNamespace(
        WAVELENGTH_M=WAVELENGTH_M,
        SELECTED_W0_M=TARGET_W0_M,
        SOURCE_SPAN_M=SOURCE_SPAN_M,
        LATERAL_DOMAIN_M=LATERAL_DOMAIN_M,
        SOURCE_Z_M=SOURCE_Z_M,
        FOCUS_Z_M=FOCUS_Z_M,
        FDTD_Z_MIN_M=FDTD_Z_MIN_M,
        FDTD_Z_MAX_M=FDTD_Z_MAX_M,
    )
    source_audit.TARGET_FREQUENCY_HZ = source_audit.C0 / WAVELENGTH_M
    source_audit.SOURCE_START_M = SOURCE_START_M
    source_audit.SOURCE_STOP_M = SOURCE_STOP_M
    source_audit.PML_LAYERS = 24
    source_audit.MESH_ACCURACY = 3
    source_audit.SOURCE_NAME = "run002_gaussian10_w8p5_source"
    source_audit.MONITORS = {
        "source_plane": SOURCE_Z_M - 0.5e-6,
        "flake_target_plane": FOCUS_Z_M,
        "downstream_plane": FOCUS_Z_M - 5.0e-6,
    }

    def run002_setup(*args, **kwargs):
        built = BASE_SETUP(*args, **kwargs)
        built["source"]["source_object_waist_calibration"] = {
            "method": "none_before_first_run002_source_only_measurement",
            "input_source_object_waist_m": float(
                built["source"]["Lumerical_source_object_waist_radius_m"]
            ),
            "target_plane_realized_waist": "TO_BE_MEASURED_BY_THIS_RUN",
            "legacy_11um_12um_calibration_reused": False,
            "Q_clipping_smoothing_gain_or_rescaling": False,
        }
        return built

    source_audit.setup = run002_setup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-ps", type=float, default=4.0)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-5)
    parser.add_argument("--gpu-device", default="GPU 4")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument(
        "--source-object-waist-um",
        type=float,
        default=8.5,
        help=(
            "Lumerical source-object waist. The target-plane realized waist "
            "is measured and must pass the 0.5%% gate before material runs."
        ),
    )
    args = parser.parse_args()
    if args.source_object_waist_um <= 0.0:
        parser.error("--source-object-waist-um must be positive")
    configure_source_audit()
    forwarded = [
        sys.argv[0],
        "--output-dir",
        args.output_dir,
        "--duration-ps",
        str(args.duration_ps),
        "--auto-shutoff-min",
        str(args.auto_shutoff_min),
        "--gpu-device",
        args.gpu_device,
        "--threads",
        args.threads,
        "--mesh-accuracy",
        "3",
        "--target-waist-um",
        "8.5",
        "--source-object-waist-um",
        str(args.source_object_waist_um),
    ]
    if args.contract_only:
        forwarded.append("--contract-only")
    sys.argv = forwarded
    return source_audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
