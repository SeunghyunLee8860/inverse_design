#!/usr/bin/env python3
"""Assemble the Fig.-3I-style SPCM line-scan profile I(s).

Consumes the ``scan40_w6p83_palik_finite_{pol}_{label}_*`` optical
artifacts and the matching ``scan40_thermal_{pol}_{label}_*`` thermal/PTE
summaries, and produces the paper-comparator table: terminal current
versus signed beam distance s from the digitized off-axis edge
(s > 0 inward), per polarization, with the edge-lobe extremum ratio
|I_a|/|I_b| compared against the digitized paper value 0.8366 +/- 0.0085.

Acceptance for scan artifacts (documented, raw values kept): all gates
pass, or the only failures are the auto-shutoff floor and/or the
six-face closure gate provided the ABSOLUTE closure error referred to
the incident power stays below 1% (the relative-to-P_Q metric diverges
by construction when the beam sits off the flake and P_Q is small).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PAPER_RATIO = 0.8365896980461811
PAPER_RATIO_UNCERTAINTY = 0.00852575488454707
POSITIONS = {
    "sm1p5": -1.5,
    "s0": 0.0,
    "s1": 1.0,
    "s2": 2.0,
    "s3": 3.0,
    "s5": 5.0,
}
ALLOWED_GATES = {
    "auto_shutoff_reached_requested_threshold",
    "six_face_closure_lt_0p5_percent",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def optical_acceptance(case: dict[str, Any]) -> dict[str, Any]:
    run_result = case.get("run_result") or {}
    acceptance = run_result.get("acceptance") or {}
    failed = [key for key, value in acceptance.items() if not value]
    closure = run_result.get("six_face_relative_closure")
    p_q = run_result.get("P_Q_W")
    incident = (run_result.get("normalization") or {}).get(
        "incident_power_W_at_1_W_m2"
    )
    absolute_closure_of_incident = (
        abs(closure) * abs(p_q) / abs(incident)
        if None not in (closure, p_q, incident) and incident
        else None
    )
    usable = bool(acceptance) and set(failed) <= ALLOWED_GATES
    if "six_face_closure_lt_0p5_percent" in failed:
        usable = usable and (
            absolute_closure_of_incident is not None
            and absolute_closure_of_incident < 0.01
        )
    return {
        "failed_gates": failed,
        "six_face_relative_closure": closure,
        "absolute_closure_fraction_of_incident": (
            absolute_closure_of_incident
        ),
        "usable": usable,
    }


def beam_readback(case: dict[str, Any]) -> dict[str, Any]:
    run_result = case.get("run_result") or {}
    beam = run_result.get("beam_and_field_readback") or {}
    out = {}
    for key in (
        "waist_x_m",
        "waist_y_m",
        "centre_x_m",
        "centre_y_m",
        "waist_x_moment_m",
        "waist_y_moment_m",
    ):
        if key in beam:
            out[key] = float(beam[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end"
        ),
    )
    parser.add_argument("--stamp", default="20260801")
    parser.add_argument("--gpu-map", default="a=4,b=3")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_of = dict(pair.split("=") for pair in args.gpu_map.split(","))
    rows: list[dict[str, Any]] = []
    for pol in ("a", "b"):
        for label, s_um in POSITIONS.items():
            optical_dir = (
                args.artifact_root
                / f"scan40_w6p83_palik_finite_{pol}_{label}_"
                f"gpu{gpu_of[pol]}_{args.stamp}"
            )
            thermal_dir = (
                args.artifact_root
                / f"scan40_thermal_{pol}_{label}_{args.stamp}"
            )
            case_path = optical_dir / "case_result.json"
            summary_path = thermal_dir / "summary.json"
            if not case_path.is_file() or not summary_path.is_file():
                rows.append(
                    {
                        "polarization": pol,
                        "label": label,
                        "s_um": s_um,
                        "status": "MISSING",
                    }
                )
                continue
            case = json.loads(case_path.read_text())
            summary = json.loads(summary_path.read_text())
            acceptance = optical_acceptance(case)
            run_result = case.get("run_result") or {}
            rows.append(
                {
                    "polarization": pol,
                    "label": label,
                    "s_um": s_um,
                    "status": (
                        "OK" if acceptance["usable"] else "REJECTED"
                    ),
                    "I_A": summary["PTE_current_A_at_285uW_incident"],
                    "P_flake_W": summary["mapping"][
                        "source_power_by_optical_material_support_W"
                    ]["TaIrTe4_exact_support_W"],
                    "Tmax_K": summary["thermal"]["Tmax_rise_K"],
                    "P_Q_W_at_1Wm2": run_result.get("P_Q_W"),
                    "six_face_relative_closure": acceptance[
                        "six_face_relative_closure"
                    ],
                    "closure_fraction_of_incident": acceptance[
                        "absolute_closure_fraction_of_incident"
                    ],
                    "failed_gates": ";".join(acceptance["failed_gates"]),
                    **beam_readback(case),
                }
            )

    usable = [row for row in rows if row.get("status") == "OK"]
    profile: dict[str, dict[float, float]] = {"a": {}, "b": {}}
    for row in usable:
        profile[row["polarization"]][row["s_um"]] = row["I_A"]

    extremum = {}
    for pol in ("a", "b"):
        if profile[pol]:
            s_values = np.asarray(sorted(profile[pol]))
            currents = np.asarray(
                [profile[pol][s] for s in s_values]
            )
            index = int(np.argmax(np.abs(currents)))
            extremum[pol] = {
                "s_um": float(s_values[index]),
                "I_A": float(currents[index]),
            }
    ratio = (
        abs(extremum["a"]["I_A"]) / abs(extremum["b"]["I_A"])
        if "a" in extremum and "b" in extremum
        else None
    )

    pointwise_ratio = {
        f"{s_um:g}um": abs(profile["a"][s_um]) / abs(profile["b"][s_um])
        for s_um in sorted(set(profile["a"]) & set(profile["b"]))
    }

    result = {
        "status": "EDGE_SCAN_PROFILE_ASSEMBLED",
        "comparator": (
            "paper Fig. 3I line-scan edge-lobe extremum ratio; "
            "each polarization takes its own |I|-maximum over scanned "
            "beam positions"
        ),
        "paper_ratio": PAPER_RATIO,
        "paper_ratio_uncertainty": PAPER_RATIO_UNCERTAINTY,
        "scan_positions_um": sorted(POSITIONS.values()),
        "rows": rows,
        "edge_lobe_extremum": extremum,
        "extremum_abs_Ia_over_abs_Ib": ratio,
        "pointwise_abs_Ia_over_abs_Ib": pointwise_ratio,
    }
    (args.report_dir / "edge_scan_profile_summary.json").write_text(
        json.dumps(jsonable(result), indent=2) + "\n"
    )
    with (args.report_dir / "edge_scan_profile.csv").open(
        "w", newline=""
    ) as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure, (axis, ratio_axis) = plt.subplots(
        1, 2, figsize=(12.5, 4.8)
    )
    for pol, color in (("a", "#cc3311"), ("b", "#4477aa")):
        if profile[pol]:
            s_values = sorted(profile[pol])
            axis.plot(
                s_values,
                [profile[pol][s] * 1e9 for s in s_values],
                "o-",
                color=color,
                label=f"E||{pol}",
            )
    axis.axvline(0.0, color="k", lw=1.0, ls=":")
    axis.text(0.05, 0.02, "edge", transform=axis.get_xaxis_transform())
    axis.set_xlabel("signed beam distance from edge s (um, + inward)")
    axis.set_ylabel("terminal current at 285 uW (nA)")
    axis.set_title("Simulated Fig.-3I line scan (Palik SiO2 scenario)")
    axis.grid(alpha=0.25)
    axis.legend()
    if pointwise_ratio:
        s_common = sorted(set(profile["a"]) & set(profile["b"]))
        ratio_axis.plot(
            s_common,
            [
                abs(profile["a"][s]) / abs(profile["b"][s])
                for s in s_common
            ],
            "s-",
            color="#228833",
            label="|I_a|/|I_b| (s)",
        )
    ratio_axis.axhline(
        PAPER_RATIO, color="#cc3311", lw=1.6, label="paper 0.8366"
    )
    ratio_axis.axhline(1.0, color="k", lw=0.8, ls=":")
    if ratio is not None:
        ratio_axis.axhline(
            ratio,
            color="#4477aa",
            lw=1.4,
            ls="--",
            label=f"extremum ratio {ratio:.3f}",
        )
    ratio_axis.set_xlabel("s (um)")
    ratio_axis.set_ylabel("|I_a| / |I_b|")
    ratio_axis.grid(alpha=0.25)
    ratio_axis.legend()
    figure.tight_layout()
    figure.savefig(args.report_dir / "EDGE_SCAN_PROFILE.png", dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            jsonable(
                {
                    "edge_lobe_extremum": extremum,
                    "extremum_abs_Ia_over_abs_Ib": ratio,
                    "pointwise_abs_Ia_over_abs_Ib": pointwise_ratio,
                    "rejected": [
                        f"{row['polarization']}/{row['label']}"
                        for row in rows
                        if row.get("status") != "OK"
                    ],
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
