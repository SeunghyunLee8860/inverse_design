#!/usr/bin/env python3
"""Publish the paper-stack substrate decision without weakening Q gates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAPERS = Path("/home/seunghyun/tairte4/papers")

CASES = {
    "2022_Z_285nm_SiO2": RESULTS
    / "z2022_backplane_truncation"
    / "backplane_truncation_summary.json",
    "2024_T_main_1500nm_SiO2": RESULTS
    / "t2024_main_backplane_truncation"
    / "backplane_truncation_summary.json",
}

PAPER_FILES = {
    "2022_supplement": PAPERS / "41467_2022_32309_MOESM1_ESM.pdf",
    "2024_supplement": PAPERS / "41467_2024_51599_MOESM1_ESM.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    summaries = {name: json.loads(path.read_text()) for name, path in CASES.items()}
    accepted = {
        "VALIDATED_OPTICAL_SUBSTRATE_TRUNCATION_BELOW_AU_BACKPLANE",
        (
            "VALIDATED_OPTICAL_SUBSTRATE_INSENSITIVITY_BELOW_AU_BACKPLANE_"
            "WITH_Q_CLOSURE_UNRESOLVED"
        ),
    }
    for name, summary in summaries.items():
        if summary["status"] not in accepted:
            raise RuntimeError(f"substrate discriminator did not pass: {name}")

    rows = []
    for name, summary in summaries.items():
        metrics = summary["metrics"]
        rows.append(
            {
                "case": name,
                "flux_absorbed_power_difference_pct": 100.0
                * metrics["flux_absorbed_power_relative_difference"],
                "P_Q_difference_pct": 100.0 * metrics["P_Q_relative_difference"],
                "reflectance_absolute_difference_pct": 100.0
                * metrics["reflectance_absolute_difference"],
                "top_field_NRMSE_pct": 100.0 * metrics["top_field_vector_NRMSE"],
                "full_stack_transmission": metrics["full_transmission"],
                "full_Q_flux_closure_pct": 100.0 * metrics["full_closure_relative"],
            }
        )

    decision = {
        "status": "PUBLISHED_PAPER_STACK_SUBSTRATE_DECISION",
        "active_2D_material_rule": (
            "Only the paper active 2-D material is replaced by fixed 100-nm "
            "TaIrTe4 in the architecture contracts."
        ),
        "Maxwell": {
            "A_direct_without_Au_backplane": (
                "KEEP_EXPLICIT_SIO2_SI_UNTIL_SEPARATE_ENDPOINT_EQUIVALENCE"
            ),
            "B_T_2024_above_opaque_Au": (
                "AU_TRUNCATION_ALLOWED_FOR_OPTICAL_ACCELERATION_UNDER_"
                "VALIDATED_200NM_NUMERICAL_CLOSURE"
            ),
            "B_Z_2022_above_published_200nm_Au": (
                "AU_TRUNCATION_ALLOWED_FOR_OPTICAL_ACCELERATION"
            ),
        },
        "thermal": {
            "all_architectures": (
                "KEEP_EXPLICIT_SIO2_SI; OPTICAL OPACITY DOES NOT REMOVE THE "
                "THERMAL HEAT PATH"
            ),
            "reduced_boundary_option": (
                "A single effective substrate/Robin impedance is a future "
                "candidate only after explicit-3D equivalence."
            ),
        },
        "paper_provenance": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in PAPER_FILES.items()
        },
        "important_limits": [
            (
                "2024 main Methods reports 1.5 um thermal SiO2, while "
                "Supplementary Fig. 17 RF cross-section reports 1.0 um; these "
                "remain separate scenarios."
            ),
            (
                "The 2024 publication does not provide a complete 10-um T geometry "
                "or an exact MIR back-reflector thickness. The 200-nm Au control is "
                "an explicit numerical closure, not a claimed published dimension."
            ),
            (
                "The 2022 10 W/m2 interface entry is a heat flux, not a thermal "
                "boundary conductance G."
            ),
            (
                "The strict periodic pabs_adv volume-Q versus flux closure is "
                "2.54-2.56% and remains unresolved; no correction, gain, or "
                "rescaling was applied."
            ),
        ],
        "cases": summaries,
    }

    json_path = RESULTS / "substrate_reduction_decision.json"
    json_path.write_text(json.dumps(decision, indent=2) + "\n")

    csv_path = RESULTS / "substrate_reduction_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    labels = ["2022 Z", "2024 T"]
    axes[0].bar(
        labels,
        [row["flux_absorbed_power_difference_pct"] for row in rows],
        color=["#3d6ca8", "#c7782f"],
    )
    axes[0].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[0].set_title("Full substrate vs Au-truncated")
    axes[0].set_ylabel("absorbed-flux difference (%)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        labels,
        [row["top_field_NRMSE_pct"] for row in rows],
        color=["#3d6ca8", "#c7782f"],
    )
    axes[1].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[1].set_title("Field above Au backplane")
    axes[1].set_ylabel("complex-field NRMSE (%)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    plot_path = RESULTS / "substrate_reduction_decision.png"
    figure.savefig(plot_path, dpi=220)
    plt.close(figure)

    report_path = RESULTS / "SUBSTRATE_REDUCTION_DECISION.md"
    report_path.write_text(
        f"""# SiO2/Si versus one-substrate decision

Status: `PUBLISHED_PAPER_STACK_SUBSTRATE_DECISION`

## Direct answer

One substrate is **not** a universal replacement for the paper stacks.

- Basic A without an opaque Au mirror keeps explicit SiO2/Si in Maxwell.
- For the 2022 Z stack, the published 200-nm Au backplate makes the layers
  below it optically irrelevant to the fields above it.
- For the 2024 T stack, the same reduction is allowed only under the explicit
  200-nm numerical closure used here; 200 nm is not presented as a published
  2024 MIR dimension.
- Thermal calculations keep the SiO2/Si heat path. Optical opacity is not a
  thermal boundary condition.

Only the active 2-D material is replaced by the fixed 100-nm TaIrTe4 layer in
the architecture contracts.

## GPU v261 discriminator

| case | absorbed-flux difference | P_Q difference | top-field NRMSE | full transmission |
|---|---:|---:|---:|---:|
| 2022 Z, 285-nm SiO2/Si | {rows[0]['flux_absorbed_power_difference_pct']:.6f}% | {rows[0]['P_Q_difference_pct']:.6f}% | {rows[0]['top_field_NRMSE_pct']:.6f}% | {rows[0]['full_stack_transmission']:.3e} |
| 2024 T main, 1.5-um SiO2/Si | {rows[1]['flux_absorbed_power_difference_pct']:.6f}% | {rows[1]['P_Q_difference_pct']:.6f}% | {rows[1]['top_field_NRMSE_pct']:.6f}% | {rows[1]['full_stack_transmission']:.3e} |

The strict volume-Q/flux closures are {rows[0]['full_Q_flux_closure_pct']:.3f}%
and {rows[1]['full_Q_flux_closure_pct']:.3f}%, respectively. They remain
fail-closed diagnostics. No Q clipping, smoothing, gain, global rescaling, or
empirical normalization was used.

## Supplementary-data corrections retained

- 2022 optical reference: Si / 285-nm thermal SiO2 / 200-nm Au backplate /
  200--270-nm Al2O3 / 5-nm Cr + 50-nm Au antenna / air.
- 2024: 35-nm Al2O3 spacer and 50-nm top Al2O3 passivation are disclosed.
  Main Methods reports 1.5-um thermal SiO2; Supplementary Fig. 17's RF stack
  says 1.0 um. The two values are not averaged.
- The 2022 `10 W/m2` interface value is a heat flux, not a conductance.

![substrate decision](substrate_reduction_decision.png)
"""
    )

    manifest_path = RESULTS / "SUBSTRATE_REDUCTION_RAW_ARTIFACT_MANIFEST.json"
    raw_rows = []
    for summary in summaries.values():
        for label in ("full", "truncated"):
            case_dir = Path(summary["raw_inputs"][label])
            case = json.loads((case_dir / "backplane_case_result.json").read_text())
            raw_rows.extend(case.get("raw_artifacts", []))
    manifest = {
        "status": decision["status"],
        "generated_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (json_path, csv_path, plot_path, report_path)
        ],
        "raw_artifacts_not_committed_to_git": raw_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": decision["status"], "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
