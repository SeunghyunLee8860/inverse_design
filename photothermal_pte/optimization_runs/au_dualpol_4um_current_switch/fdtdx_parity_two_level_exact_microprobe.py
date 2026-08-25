#!/usr/bin/env python3
"""Bounded GPU probe adapter for the offline two-level exact sparse VJP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_parity_blockwise_exact_microprobe as base,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ad_contract import (
    latent_directions,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_optical_controls import (
    _validate_new_external_path,
    file_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_two_level_exact_sparse_vjp import (
    two_level_exact_sparse_ade_cpml_phasor_design_fdtd,
    two_level_exact_sparse_checkpoint_audit,
)


SCHEMA = "fdtdx_4um_parity_two_level_exact_microprobe_v1"
DEFAULT_STEPS = 4096
DEFAULT_OUTER_BLOCK_STEPS = 4096
DEFAULT_SEGMENT_STEPS = 64
MAX_STEPS = 4096
HERE = Path(__file__).resolve().parent
TWO_LEVEL_SOURCE_FILES = (
    HERE / "fdtdx_parity_two_level_exact_sparse_vjp.py",
    HERE / "fdtdx_parity_sparse_ade_checkpoint.py",
)


def two_level_exact_source_audit() -> dict[str, Any]:
    """Hash and fail-close the offline direct-segment implementation."""

    missing = [str(path) for path in TWO_LEVEL_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing two-level source files: {missing}")
    implementation_path = TWO_LEVEL_SOURCE_FILES[0]
    implementation = implementation_path.read_text(encoding="utf-8")
    checks = {
        "standard_pinned_forward_step_used": (
            "from fdtdx.fdtd.forward import forward" in implementation
            and "_, output = forward(" in implementation
        ),
        "exact_outer_starts_retained": (
            "exact_outer_starts" in implementation
            and "return final_state, state" in implementation
        ),
        "exact_inner_starts_recomputed": (
            "exact_segment_starts = run_outer(" in implementation
            and "exact_segment_start" in implementation
        ),
        "short_direct_segment_VJP_used": (
            "jax.vjp(" in implementation
            and "run_segment(" in implementation
        ),
        "online_checkpointed_loop_absent": ('kind="checkpointed"' not in implementation),
        "algebraic_reverse_calls_absent": (
            "update_H_reverse" not in implementation
            and "update_E_reverse" not in implementation
            and "reverse_cpml_auxiliary" not in implementation
        ),
        "only_design_c3_is_differentiable": (
            "primitive(initial_state, design_c3)" in implementation
        ),
        "sparse_regional_P_is_used": (
            "extract(container.fields.dispersive_P_curr)" in implementation
            and "dispersive_P_curr=expand(P_curr)" in implementation
        ),
    }
    return {
        "schema": "fdtdx_4um_two_level_exact_source_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in TWO_LEVEL_SOURCE_FILES
        },
    }


def _checkpoint_audit_adapter(
    arrays: Any,
    *,
    regions: Any,
    jax_module: Any,
    total_steps: int,
    steps_per_block: int,
    inner_checkpoints: int,
) -> dict[str, Any]:
    return two_level_exact_sparse_checkpoint_audit(
        arrays,
        regions=regions,
        jax_module=jax_module,
        total_steps=total_steps,
        outer_block_steps=steps_per_block,
        segment_steps=inner_checkpoints,
    )


def _vjp_adapter(
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
    steps_per_block: int,
    inner_checkpoints: int,
    regions: Any,
    design_region: Any,
    support_audit: dict[str, Any],
):
    return two_level_exact_sparse_ade_cpml_phasor_design_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        outer_block_steps=steps_per_block,
        segment_steps=inner_checkpoints,
        regions=regions,
        design_region=design_region,
        support_audit=support_audit,
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run through the hash/GPU-safe base harness and relabel its adapter fields."""

    output_json = _validate_new_external_path(args.output_json)
    output_npz = _validate_new_external_path(args.output_npz)
    intermediate_json = _validate_new_external_path(
        output_json.with_name(output_json.name + ".adapter.json")
    )

    base.blockwise_exact_source_audit = two_level_exact_source_audit
    base.blockwise_exact_sparse_checkpoint_audit = _checkpoint_audit_adapter
    base.blockwise_exact_sparse_ade_cpml_phasor_design_fdtd = _vjp_adapter
    base.SCHEMA = SCHEMA
    adapted_args = argparse.Namespace(
        gpu_uuid=args.gpu_uuid,
        output_json=intermediate_json,
        output_npz=output_npz,
        polarization=args.polarization,
        steps=args.steps,
        block_steps=args.outer_block_steps,
        inner_checkpoints=args.segment_steps,
        fd_step=args.fd_step,
        direction=args.direction,
    )
    report = dict(base.run_probe(adapted_args))
    report["schema"] = SCHEMA
    if report["status"] == "PASS_BOUNDED_BLOCKWISE_EXACT_AD_CONNECTIVITY_ONLY":
        report["status"] = "PASS_BOUNDED_TWO_LEVEL_EXACT_AD_CONNECTIVITY_ONLY"
    report["scope"] = (
        "short_exact_grid_offline_two_level_exact_AD_connectivity_and_resource_probe_only"
    )
    report["two_level_exact_source_audit"] = report.pop(
        "blockwise_exact_source_audit"
    )
    report["outer_block_steps"] = report.pop("block_steps")
    report["num_outer_blocks"] = report.pop("num_blocks")
    report["segment_steps"] = report.pop("inner_checkpoints")
    report["online_checkpointed_loop_used"] = False
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["base_harness_sha256"] = hashlib.sha256(
        Path(base.__file__).read_bytes()
    ).hexdigest()
    report["adapter_intermediate_json_path"] = str(intermediate_json)
    report["adapter_intermediate_json_sha256"] = file_sha256(intermediate_json)
    report["report_sha256"] = ""
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    base._write_new_external_json(output_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--outer-block-steps",
        type=int,
        default=DEFAULT_OUTER_BLOCK_STEPS,
    )
    parser.add_argument("--segment-steps", type=int, default=DEFAULT_SEGMENT_STEPS)
    parser.add_argument("--fd-step", type=float, default=base.DEFAULT_FD_STEP)
    parser.add_argument(
        "--direction",
        choices=tuple(latent_directions()),
        default=base.DEFAULT_DIRECTION,
    )
    args = parser.parse_args()
    if not 2 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [2,{MAX_STEPS}]")
    if not 1 <= args.outer_block_steps <= args.steps:
        parser.error("--outer-block-steps must be in [1,steps]")
    if (
        not 1 <= args.segment_steps <= args.outer_block_steps
        or args.outer_block_steps % args.segment_steps
    ):
        parser.error("--segment-steps must divide --outer-block-steps")
    if not math.isfinite(args.fd_step) or not 0.0 < args.fd_step <= 2.0e-2:
        parser.error("--fd-step must be finite and in (0,0.02]")
    return args


def main() -> int:
    report = run_probe(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
