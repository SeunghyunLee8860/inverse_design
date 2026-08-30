#!/usr/bin/env python3
"""Publish the fail-closed Au sharp-interface P_Q adjoint diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path("/home/seunghyun/tairte4/raw_artifacts/au_topology_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_entry(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    case = args.raw_root / "sharp_width_8p0_edge25_pq_adjoint"
    source = json.loads(
        (case / "au_sharp_interface_pq_adjoint_result.json").read_text()
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if source.get("passed", False):
        raise RuntimeError("this fail-closed summarizer received a passing result")
    if "CONTINUOUS_TRACE_INCOMPATIBLE" not in str(source.get("status", "")):
        raise RuntimeError(f"unexpected source status: {source.get('status')}")

    decomposition = source["derivative_decomposition"]
    fd = source["finite_difference"]
    fd_strong = float(fd["h_0.05_um"]["derivative_W_per_um"])
    ad_total = float(decomposition["total_AD_W_per_um"])
    magnitude_ratio = abs(ad_total) / abs(fd_strong)
    sign_agrees = ad_total * fd_strong > 0.0
    status = "BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_UNVALIDATED"

    summary = {
        "status": status,
        "diagnostic_status": source["status"],
        "scope": source["scope"],
        "baseline": source["baseline"],
        "finite_difference": fd,
        "candidate_shape_adjoint": decomposition,
        "AD_FD_comparison": source["AD_FD_comparison"],
        "AD_to_strong_FD_magnitude_ratio": magnitude_ratio,
        "AD_FD_sign_agrees": sign_agrees,
        "surface_quadrature": source["surface_quadrature"],
        "surface_quadrature_final_refinement_relative_change": source[
            "surface_quadrature_final_refinement_relative_change"
        ],
        "source_and_coordinate_gates": source["gates"],
        "interpretation": {
            "accepted": [
                "25 nm exact-binary Au baseline forward optical gates",
                "GPU FieldRegion source round trip and forward/adjoint coordinate match",
                "candidate surface quadrature convergence",
            ],
            "rejected": [
                "continuous pointwise inside-Au Q trace as the direct derivative of the discrete conformal-Yee P_Q objective",
                "production Au P_Q shape gradient",
                "coupled Au thermal/electrical/PTE optimization",
            ],
            "reason": (
                "the candidate AD has the wrong sign and differs from the "
                "strong central FD by more than five orders of magnitude; "
                "the sharp metal-edge trace is not a solver-consistent "
                "derivative of the discrete conformal-Yee loss integral"
            ),
        },
        "density_route_status": (
            "FAILED_DENSITY_ROUTE_UNIFORM_AU_IMPORTNK2_DIVERGENCE_"
            "FALLBACK_SHARP_INTERFACE"
        ),
        "production_Au_optimization_permitted": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "finite_difference_used_to_fit_AD": False,
    }
    (output / "au_sharp_interface_pq_adjoint_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    rows = [
        {
            "kind": "FD",
            "term": name,
            "derivative_W_per_um": item["derivative_W_per_um"],
            "relative_to_abs_strong_FD": float(item["derivative_W_per_um"])
            / abs(fd_strong),
        }
        for name, item in fd.items()
    ]
    for term, key in (
        ("field_mediated_boundary", "field_mediated_boundary_W_per_um"),
        ("explicit_moving_domain_candidate", "explicit_moving_absorption_domain_W_per_um"),
        ("candidate_total", "total_AD_W_per_um"),
    ):
        value = float(decomposition[key])
        rows.append(
            {
                "kind": "AD_candidate",
                "term": term,
                "derivative_W_per_um": value,
                "relative_to_abs_strong_FD": value / abs(fd_strong),
            }
        )
    with (output / "au_sharp_interface_pq_adjoint_terms.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    labels = [row["term"] for row in rows]
    values = [row["relative_to_abs_strong_FD"] for row in rows]
    colors = ["#2878B5" if row["kind"] == "FD" else "#C82423" for row in rows]
    axes[0].bar(range(len(rows)), values, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xticks(range(len(rows)), labels, rotation=28, ha="right")
    axes[0].set_ylabel("derivative / |strong FD|")
    axes[0].set_title("Candidate continuous trace is rejected")
    axes[0].grid(axis="y", alpha=0.25)
    quadrature = source["surface_quadrature"]
    resolution = [1e9 * float(row["dy_m"]) for row in quadrature]
    total = [1e6 * float(row["total_W_per_m"]) for row in quadrature]
    direct = [1e6 * float(row["direct_W_per_m"]) for row in quadrature]
    indirect = [1e6 * float(row["indirect_W_per_m"]) for row in quadrature]
    axes[1].plot(resolution, total, "o-", label="candidate total")
    axes[1].plot(resolution, direct, "o--", label="direct trace")
    axes[1].plot(resolution, indirect, "o--", label="field-mediated")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("surface quadrature dy (nm)")
    axes[1].set_ylabel("candidate derivative (W/um)")
    axes[1].set_title("Quadrature converges to the wrong derivative")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "au_sharp_interface_pq_adjoint_failure.png", dpi=180)
    plt.close(figure)

    report = f"""# Au sharp-interface P_Q adjoint diagnostic

Status: `{status}`

The 25 nm exact-binary Au baseline forward solve passed: `P_Q =
{float(source['baseline']['P_Q_W']):.12e} W`, six-face closure is
`{100.0*float(source['baseline']['closure_relative']):.6f}%`, and the adjoint
auto-shutoff is below `1e-5`. The FieldRegion source round trip is exact and
the forward/adjoint coordinate mismatch is zero.

The numerical AD--FD gate nevertheless fails decisively. The strong central
FD is `{fd_strong:.12e} W/um`, while the candidate boundary result is
`{ad_total:.12e} W/um`. Their signs {'agree' if sign_agrees else 'do not agree'}
and the magnitude ratio is `{magnitude_ratio:.6e}`.

The candidate explicitly included both the bundled v261 tangential-E/normal-D
field-mediated boundary kernel and a moving-domain absorption trace. The
surface quadrature itself converges (`{100.0*float(source['surface_quadrature_final_refinement_relative_change']):.6f}%`
on its final refinement), so quadrature resolution is not the explanation.
Instead, the continuous pointwise inside-Au loss trace is incompatible with
the discrete conformal-Yee `P_Q` objective at a sharp lossy-metal edge. It is
therefore rejected; no fit, normalization, sign change, or gradient rescaling
is applied.

The alternative density route is also not available: the uniform rho=1 Au
`importnk2` endpoint diverged. Consequently neither current Au representation
has a certified optical gradient, and no Au/TaIrTe4 thermal, electrical, PTE,
or optimization run is permitted from this checkpoint.

The next isolated diagnostic should use a fixed external field observable to
certify the boundary kernel independently of an explicit moving material-loss
integral. A production PTE formulation then needs a solver-consistent
conformal material derivative or a boundary-fitted discretization; the rejected
continuous trace must not be reused.
"""
    (output / "AU_SHARP_INTERFACE_PQ_ADJOINT_REPORT.md").write_text(report)

    raw_files = []
    for role, path in (
        (
            "25nm_baseline_forward_fsp",
            args.raw_root / "sharp_width_8p0_edge25_forward" / "complex_material_control.fsp",
        ),
        (
            "25nm_baseline_forward_npz",
            args.raw_root / "sharp_width_8p0_edge25_forward" / "complex_material_control_q.npz",
        ),
        ("adjoint_template_fsp", case / "au_sharp_interface_pq_adjoint_template.fsp"),
        ("adjoint_gpu_fsp", case / "au_sharp_interface_pq_adjoint_gpu.fsp"),
        (
            "adjoint_gpu_h5",
            case
            / "au_sharp_interface_pq_adjoint_gpu"
            / "au_sharp_interface_pq_adjoint_gpu_output.h5",
        ),
        ("adjoint_log", case / "au_sharp_interface_pq_adjoint_gpu_p0.log"),
        ("raw_result", case / "au_sharp_interface_pq_adjoint_result.json"),
    ):
        raw_files.append(raw_entry(role, path))
    manifest = {
        "status": status,
        "raw_files_committed": False,
        "generation_command": (
            "python 10_run_au_sharp_interface_pq_adjoint.py "
            "--output-dir <raw_case> --gpu-device 'GPU 6'"
        ),
        "resume_command_used_for_vectorized_postprocess": (
            "python 10_run_au_sharp_interface_pq_adjoint.py "
            "--output-dir <raw_case> --gpu-device 'GPU 6' "
            "--resume-completed-adjoint"
        ),
        "raw_files": raw_files,
    }
    (output / "AU_SHARP_INTERFACE_PQ_ADJOINT_RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "raw_artifacts": len(raw_files)}, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
