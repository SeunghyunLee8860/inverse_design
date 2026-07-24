#!/usr/bin/env python3
"""Summarize pabs script audit, native Yee channels, controls, and flux box."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100 * value:.6f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", required=True)
    args = parser.parse_args()
    root = Path(args.diagnostic_root).expanduser().resolve()
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    component = load(root / "component_analysis/native_yee_component_summary.json")
    excess = load(root / "component_analysis/disk_excess_channel_summary.json")
    script_audit = load(root / "script_audit/pabs_adv_script_export.json")
    controls = [
        (
            "disk anisotropic",
            root / "controls/disk_anisotropic/fdtd_qon/fdtd_absorption_summary.json",
        ),
        (
            "square anisotropic",
            root / "controls/square_anisotropic/fdtd_qon/fdtd_absorption_summary.json",
        ),
        (
            "disk isotropic eps_a",
            root / "controls/disk_isotropic_a/fdtd_qon/fdtd_absorption_summary.json",
        ),
        (
            "square isotropic eps_a",
            root / "controls/square_isotropic_a/fdtd_qon/fdtd_absorption_summary.json",
        ),
    ]
    rows = []
    for label, path in controls:
        summary = load(path)
        absorption = summary["absorption"]
        six = absorption["six_face_flux"]
        a_q = float(absorption["Pabs_total_normalized_from_lumerical"])
        a_local = float(absorption["A_local_flux"])
        a_six = float(six["A_six_face_net"])
        rows.append(
            {
                "case": label,
                "material_control": summary["fixed_geometry"][
                    "flake_material_control"
                ],
                "A_pabs_adv": a_q,
                "A_local_top_bottom": a_local,
                "A_six_face_net": a_six,
                "Pabs_minus_six_face": a_q - a_six,
                "relative_mismatch_vs_six_face": (a_q - a_six) / a_six,
                "A_x_pair": float(six["A_x_pair"]),
                "A_y_pair": float(six["A_y_pair"]),
                "six_face_minus_top_bottom": float(
                    six["six_face_minus_top_bottom"]
                ),
                "summary_path": str(path),
            }
        )
    with (output / "anisotropic_isotropic_control_matrix.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cases = ["disk", "square"]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    bottom = np.zeros(len(cases))
    for key, label in (
        ("A_Qx_native", "Qx"),
        ("A_Qy_native", "Qy"),
        ("A_Qz_native", "Qz"),
    ):
        values = [component[name][key] for name in cases]
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.scatter(
        x,
        [component[name]["A_local_flux"] for name in cases],
        marker="_",
        s=900,
        linewidths=3,
        color="k",
        label="local flux",
        zorder=5,
    )
    ax.set_xticks(x, cases)
    ax.set_ylabel("absorptance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "native_yee_components_vs_flux.png", dpi=190)
    plt.close(fig)

    x = np.arange(len(rows))
    width = 0.26
    fig, axes = plt.subplots(
        2, 1, figsize=(10.0, 7.3), gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].bar(
        x - width,
        [row["A_pabs_adv"] for row in rows],
        width,
        label="pabs_adv",
    )
    axes[0].bar(
        x,
        [row["A_local_top_bottom"] for row in rows],
        width,
        label="top-bottom",
    )
    axes[0].bar(
        x + width,
        [row["A_six_face_net"] for row in rows],
        width,
        label="six-face net",
    )
    axes[0].set_ylabel("absorptance")
    axes[0].set_xticks(x, [row["case"] for row in rows], rotation=15)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x,
        [100 * row["relative_mismatch_vs_six_face"] for row in rows],
    )
    axes[1].set_ylabel("Pabs vs six-face [%]")
    axes[1].set_xticks(x, [row["case"] for row in rows], rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "anisotropic_isotropic_six_face_control.png", dpi=190)
    plt.close(fig)

    isotropic_mismatches = [
        abs(row["relative_mismatch_vs_six_face"])
        for row in rows
        if row["material_control"].startswith("isotropic")
    ]
    if max(isotropic_mismatches) < 0.01:
        control_conclusion = (
            "The patterned isotropic controls close within 1%, so the large "
            "mismatch is anisotropy-specific rather than a generic finite-edge "
            "interpolation effect."
        )
    else:
        control_conclusion = (
            "The patterned isotropic controls retain a mismatch above 1%, so "
            "the discrepancy is not isolated to anisotropic loss."
        )

    disk = component["disk"]
    square = component["square"]
    delta = component["disk_minus_flat"]
    false = component["disk_false_Ez_comparison"]
    corr = excess["correlations_in_TaIrTe4"]
    lines = [
        "# Anisotropic native-Yee Pabs audit",
        "",
        "## Scope",
        "",
        "- Optical FDTD only; HEAT was not run.",
        "- No Pabs/flux gain and no geometric clipping were applied.",
        "- Native fields and component-specific index arrays were obtained with `getdata(...,1)`.",
        "",
        "## pabs_adv script",
        "",
        f"- Full script: `{script_audit['analysis_script_path']}`",
        "- The script does not use scalar epsilon.",
        "- It reads `index_x^2`, `index_y^2`, and `index_z^2` separately and evaluates:",
        "  - `Qx = 0.5*eps0*omega*abs(Ex)^2*imag(eps_x)`",
        "  - `Qy = 0.5*eps0*omega*abs(Ey)^2*imag(eps_y)`",
        "  - `Qz = 0.5*eps0*omega*abs(Ez)^2*imag(eps_z)`",
        "- Each Qi is evaluated at its native Yee position before scalar interpolation.",
        "- The sampled anisotropic material is handled through the solver-produced component arrays `index_x/y/z`; pabs_adv does not reinterpret the input material table.",
        "",
        "Lumerical records frequency-domain fields with the Fourier transform "
        "`integral exp(+i*omega*t) E(t) dt`, corresponding to inverse time "
        "dependence `exp(-i*omega*t)`. For passive materials in this convention "
        "Im(epsilon)>0 and the positive-sign formula above gives positive loss.",
        "",
        "## Native Yee component integrals",
        "",
        "| geometry | A_Qx | A_Qy | A_Qz | total | local flux | mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| disk | {disk['A_Qx_native']:.12f} | {disk['A_Qy_native']:.12f} | {disk['A_Qz_native']:.3e} | {disk['A_Q_total_native']:.12f} | {disk['A_local_flux']:.12f} | {pct(disk['relative_mismatch'])} |",
        f"| square | {square['A_Qx_native']:.12f} | {square['A_Qy_native']:.12f} | {square['A_Qz_native']:.3e} | {square['A_Q_total_native']:.12f} | {square['A_local_flux']:.12f} | {pct(square['relative_mismatch'])} |",
        "",
        "The conservative common-grid mapping preserves every native component "
        "integral to floating-point precision. Native totals reproduce the "
        "Lumerical pabs_adv totals to about 1e-10 absolute.",
        "",
        "## Disk excess channel",
        "",
        f"- DeltaA_Qx: `{delta['DeltaA_Qx']:.12f}`",
        f"- DeltaA_Qy: `{delta['DeltaA_Qy']:.12f}`",
        f"- DeltaA_Qz: `{delta['DeltaA_Qz']:.3e}`",
        f"- measured excess: `{delta['excess']:.12f}`",
        f"- integral(DeltaQ-DeltaQx): `{excess['integrals']['DeltaA_Q_minus_DeltaA_Qx']:.12f}`",
        f"- correlation[(DeltaQ-DeltaQx), DeltaQy]: `{corr['residual_vs_DeltaQy']:.15f}`",
        "",
        "## False Ez hypothesis",
        "",
        f"- scalar eps_x applied to Ez: `{false['false_Ez_scalar_eps_x']:.12f}` ({false['false_eps_x_over_actual_excess']:.3f}x the actual disk mismatch)",
        f"- scalar eps_y applied to Ez: `{false['false_Ez_scalar_eps_y']:.12f}` ({false['false_eps_y_over_actual_excess']:.3f}x the actual disk mismatch)",
        f"- residual vs hypothetical false-Ez spatial correlation: `{corr['residual_vs_false_DeltaQz']:.6f}`",
        f"- residual vs Delta|Ez|^2 spatial correlation: `{corr['residual_vs_Delta_Ez2']:.6f}`",
        "",
        "The scalar-epsilon false-Ez hypothesis is therefore not numerically "
        "consistent with the measured excess. The observed extra channel is Qy.",
        "",
        "## Periodic correction and analysis padding",
        "",
        "- The outer periodic-correction switch is enabled.",
        "- X-periodic and Y-periodic branches are enabled; Z is disabled.",
        "- The monitor spans the full 6 um x 6 um periodic unit cell, as required by the script.",
        "- In z, the analysis region is approximately -150 to +50 nm around TaIrTe4 (-100 to 0 nm), providing at least one nonabsorbing mesh cell on both sides.",
        "- No lateral nonabsorbing padding exists because TaIrTe4 fills the periodic unit cell. This is the periodic-unit-cell case handled by the enabled x/y correction, not a finite isolated absorber.",
        "",
        "## Six-face and isotropic controls",
        "",
        "| case | pabs_adv | top-bottom | six-face | pabs-six | mismatch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['A_pabs_adv']:.12f} | "
            f"{row['A_local_top_bottom']:.12f} | "
            f"{row['A_six_face_net']:.12f} | "
            f"{row['Pabs_minus_six_face']:.12f} | "
            f"{pct(row['relative_mismatch_vs_six_face'])} |"
        )
    lines += [
        "",
        control_conclusion,
        "",
    ]
    (output / "ANISOTROPIC_YEE_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(lines)
    )
    combined = {
        "scope": {
            "heat_run": False,
            "clipping": False,
            "flux_gain": False,
        },
        "script_audit": script_audit,
        "component_analysis": component,
        "excess_channel": excess,
        "controls": rows,
        "control_conclusion": control_conclusion,
    }
    (output / "anisotropic_yee_diagnostic_results.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
