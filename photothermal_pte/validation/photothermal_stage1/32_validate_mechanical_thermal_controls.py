#!/usr/bin/env python3
"""Generate and run fail-closed Mechanical APDL thermal controls.

The controls use the Mechanical solver directly in batch mode.  They do not
require Workbench or PyMAPDL.  If the MAPDL executable or a Mechanical license
is unavailable, canonical input decks and analytic references are still
written, but the result remains blocked and is never called solver-validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import jsonable, utc_timestamp, write_json


KAPPA_W_MK = np.asarray([14.4, 3.8, 1.0], float)
INTERFACE_G_W_M2K = (7.37e6, 1.1e9)
MECHANICAL_LICENSE_FEATURES = {
    "ansys",
    "mech_1",
    "mech_2",
    "struct",
}
FLUX_ERROR_LIMIT = 0.01
PROFILE_ERROR_LIMIT = 0.01
INTERFACE_JUMP_ERROR_LIMIT = 0.01
ENERGY_ERROR_LIMIT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--report-dir",
        default=str(
            config.REPOSITORY_ROOT
            / "reports"
            / "mechanical_thermal_controls"
        ),
    )
    parser.add_argument("--np", type=int, default=2)
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="run a discovered executable even if license precheck is inconclusive",
    )
    return parser.parse_args()


def output_directory(explicit: str | None) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / "mechanical_thermal_controls"
        / f"{utc_timestamp()}_mechanical_controls"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def find_mapdl_executable(explicit: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for environment_name in ("AWP_ROOT261", "AWP_ROOT252", "AWP_ROOT251"):
        root = os.environ.get(environment_name)
        if root:
            candidates.extend(
                (
                    Path(root) / "ansys" / "bin" / "linx64" / "ansys261",
                    Path(root) / "ansys" / "bin" / "linx64" / "ansys252",
                    Path(root) / "ansys" / "bin" / "linx64" / "ansys251",
                )
            )
    candidates.extend(
        Path(path)
        for path in (
            "/ansys_inc/v261/ansys/bin/linx64/ansys261",
            "/usr/ansys_inc/v261/ansys/bin/linx64/ansys261",
            "/opt/ansys_inc/v261/ansys/bin/linx64/ansys261",
            "/ansys_inc/v252/ansys/bin/linx64/ansys252",
            "/usr/ansys_inc/v252/ansys/bin/linx64/ansys252",
            "/opt/ansys_inc/v252/ansys/bin/linx64/ansys252",
            "/ansys_inc/v251/ansys/bin/linx64/ansys251",
            "/usr/ansys_inc/v251/ansys/bin/linx64/ansys251",
            "/opt/ansys_inc/v251/ansys/bin/linx64/ansys251",
        )
    )
    for command in ("ansys261", "ansys252", "ansys251", "mapdl", "ansys"):
        located = shutil.which(command)
        if located:
            candidates.append(Path(located))
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if str(resolved) in checked:
            continue
        checked.append(str(resolved))
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return {
                "status": "MECHANICAL_EXECUTABLE_FOUND",
                "passed": True,
                "executable": str(resolved),
                "checked_paths": checked,
            }
    return {
        "status": "BLOCKED_MECHANICAL_EXECUTABLE_UNAVAILABLE",
        "passed": False,
        "executable": None,
        "checked_paths": checked,
    }


def probe_license_features() -> dict[str, Any]:
    license_reference = os.environ.get("ANSYSLMD_LICENSE_FILE")
    lmutil_candidates = (
        Path("/ansys_inc/shared_files/licensing/linx64/lmutil"),
        Path("/opt/ansys_inc/shared_files/licensing/linx64/lmutil"),
        Path("/usr/ansys_inc/shared_files/licensing/linx64/lmutil"),
    )
    lmutil = next(
        (
            candidate
            for candidate in lmutil_candidates
            if candidate.is_file() and os.access(candidate, os.X_OK)
        ),
        None,
    )
    result: dict[str, Any] = {
        "status": "BLOCKED_MECHANICAL_LICENSE_UNAVAILABLE",
        "passed": False,
        "license_reference_configured": bool(license_reference),
        "license_reference_kind": (
            "network"
            if license_reference and "@" in license_reference
            else "file_or_other"
            if license_reference
            else "missing"
        ),
        "lmutil": str(lmutil) if lmutil else None,
        "available_feature_names": [],
        "mechanical_feature_names": [],
    }
    if lmutil is None or not license_reference:
        return result
    try:
        completed = subprocess.run(
            [str(lmutil), "lmstat", "-c", license_reference, "-a"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        combined = f"{completed.stdout}\n{completed.stderr}"
        features = sorted(
            set(
                re.findall(
                    r"^Users of ([^:]+):",
                    combined,
                    flags=re.MULTILINE,
                )
            )
        )
        mechanical = sorted(set(features) & MECHANICAL_LICENSE_FEATURES)
        result.update(
            {
                "lmstat_returncode": completed.returncode,
                "license_server_reachable": (
                    "license server UP" in combined
                    and "ansyslmd: UP" in combined
                ),
                "available_feature_names": features,
                "mechanical_feature_names": mechanical,
                "passed": bool(mechanical),
                "status": (
                    "MECHANICAL_LICENSE_FEATURE_AVAILABLE"
                    if mechanical
                    else "BLOCKED_MECHANICAL_LICENSE_UNAVAILABLE"
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        )
    return result


def brick_chain(
    *,
    axis: str,
    start_m: float,
    end_m: float,
    cross_m: float,
    cells: int,
    node_start: int,
    element_start: int,
    material: int,
) -> tuple[list[str], list[str], list[list[int]]]:
    nodes: list[str] = []
    elements: list[str] = []
    planes: list[list[int]] = []
    active_coordinates = np.linspace(start_m, end_m, cells + 1)
    for plane_index, active in enumerate(active_coordinates):
        if axis == "x":
            coordinates = (
                (active, 0.0, 0.0),
                (active, cross_m, 0.0),
                (active, cross_m, cross_m),
                (active, 0.0, cross_m),
            )
        elif axis == "y":
            coordinates = (
                (0.0, active, 0.0),
                (0.0, active, cross_m),
                (cross_m, active, cross_m),
                (cross_m, active, 0.0),
            )
        elif axis == "z":
            coordinates = (
                (0.0, 0.0, active),
                (cross_m, 0.0, active),
                (cross_m, cross_m, active),
                (0.0, cross_m, active),
            )
        else:
            raise ValueError(axis)
        plane_ids = []
        for corner, coordinate in enumerate(coordinates):
            node_id = node_start + 4 * plane_index + corner
            plane_ids.append(node_id)
            nodes.append(
                f"N,{node_id},{coordinate[0]:.16e},"
                f"{coordinate[1]:.16e},{coordinate[2]:.16e}"
            )
        planes.append(plane_ids)
    for cell in range(cells):
        element_id = element_start + cell
        connectivity = [*planes[cell], *planes[cell + 1]]
        elements.extend(
            (
                f"EN,{element_id},{','.join(str(item) for item in connectivity)}",
                f"EMODIF,{element_id},MAT,{material}",
            )
        )
    return nodes, elements, planes


def reaction_sum_commands(
    component: str,
    parameter: str,
) -> list[str]:
    return [
        f"CMSEL,S,{component},NODE",
        f"{parameter}=0",
        "*GET,NCNT,NODE,0,COUNT",
        "*GET,NID,NODE,0,NUM,MIN",
        "*DO,II,1,NCNT",
        "*GET,RVAL,NODE,NID,RF,HEAT",
        f"{parameter}={parameter}+RVAL",
        "NID=NDNEXT(NID)",
        "*ENDDO",
        "ALLSEL,ALL",
    ]


def average_temperature_commands(
    component: str,
    parameter: str,
) -> list[str]:
    return [
        f"CMSEL,S,{component},NODE",
        f"{parameter}=0",
        "*GET,NCNT,NODE,0,COUNT",
        "*GET,NID,NODE,0,NUM,MIN",
        "*DO,II,1,NCNT",
        "*GET,TVAL,NODE,NID,TEMP",
        f"{parameter}={parameter}+TVAL",
        "NID=NDNEXT(NID)",
        "*ENDDO",
        f"{parameter}={parameter}/NCNT",
        "ALLSEL,ALL",
    ]


def profile_output_commands() -> list[str]:
    return [
        "*CFOPEN,node_temperature,csv",
        "*VWRITE,'node_id,x_m,y_m,z_m,temperature_K'",
        "(A)",
        "ALLSEL,ALL",
        "*GET,NCNT,NODE,0,COUNT",
        "*GET,NID,NODE,0,NUM,MIN",
        "*DO,II,1,NCNT",
        "*GET,XX,NODE,NID,LOC,X",
        "*GET,YY,NODE,NID,LOC,Y",
        "*GET,ZZ,NODE,NID,LOC,Z",
        "*GET,TVAL,NODE,NID,TEMP",
        "*VWRITE,NID,XX,YY,ZZ,TVAL",
        "(F12.0,',',E24.16,',',E24.16,',',E24.16,',',E24.16)",
        "NID=NDNEXT(NID)",
        "*ENDDO",
        "*CFCLOS",
    ]


def anisotropic_deck(axis: str) -> str:
    case_id = f"mechanical_kappa_{axis}"
    length_m = 2.0e-6
    cross_m = 1.0e-6
    nodes, elements, planes = brick_chain(
        axis=axis,
        start_m=0.0,
        end_m=length_m,
        cross_m=cross_m,
        cells=24,
        node_start=1,
        element_start=1,
        material=1,
    )
    lines = [
        "/BATCH",
        f"/FILNAME,{case_id},1",
        "/PREP7",
        "/UNITS,SI",
        "ET,1,SOLID70",
        f"MP,KXX,1,{KAPPA_W_MK[0]:.16e}",
        f"MP,KYY,1,{KAPPA_W_MK[1]:.16e}",
        f"MP,KZZ,1,{KAPPA_W_MK[2]:.16e}",
        *nodes,
        "TYPE,1",
        *elements,
        f"NSEL,S,NODE,,{planes[0][0]},{planes[0][-1]}",
        "CM,COLD_FACE,NODE",
        f"NSEL,S,NODE,,{planes[-1][0]},{planes[-1][-1]}",
        "CM,HOT_FACE,NODE",
        "ALLSEL,ALL",
        f"SAVE,{case_id},db",
        "FINISH",
        "/CLEAR,NOSTART",
        f"/FILNAME,{case_id},1",
        f"RESUME,{case_id},db",
        "/PREP7",
        "*GET,KXRB,KXX,1,TEMP,300",
        "*GET,KYRB,KYY,1,TEMP,300",
        "*GET,KZRB,KZZ,1,TEMP,300",
        "FINISH",
        "/SOLU",
        "ANTYPE,STATIC",
        "CMSEL,S,COLD_FACE,NODE",
        "D,ALL,TEMP,300",
        "CMSEL,S,HOT_FACE,NODE",
        "D,ALL,TEMP,310",
        "ALLSEL,ALL",
        "OUTRES,ALL,ALL",
        "SOLVE",
        "FINISH",
        "/POST1",
        "SET,LAST",
        *reaction_sum_commands("COLD_FACE", "QMIN"),
        *reaction_sum_commands("HOT_FACE", "QMAX"),
        "*CFOPEN,case_result,csv",
        "*VWRITE,KXRB,KYRB,KZRB,QMIN,QMAX",
        "(E24.16,',',E24.16,',',E24.16,',',E24.16,',',E24.16)",
        "*CFCLOS",
        *profile_output_commands(),
        "FINISH",
        "/EXIT,NOSAVE",
    ]
    return "\n".join(lines) + "\n"


def interface_deck(G_W_m2K: float) -> str:
    label = f"{G_W_m2K:.8g}".replace(".", "p").replace("+", "")
    case_id = f"mechanical_interface_G_{label}"
    length_m = 1.0e-6
    cross_m = 1.0e-6
    lower_nodes, lower_elements, lower_planes = brick_chain(
        axis="z",
        start_m=-length_m,
        end_m=0.0,
        cross_m=cross_m,
        cells=20,
        node_start=1,
        element_start=1,
        material=1,
    )
    upper_nodes, upper_elements, upper_planes = brick_chain(
        axis="z",
        start_m=0.0,
        end_m=length_m,
        cross_m=cross_m,
        cells=20,
        node_start=10001,
        element_start=1001,
        material=2,
    )
    target_nodes = lower_planes[-1]
    contact_nodes = [
        upper_planes[0][0],
        upper_planes[0][3],
        upper_planes[0][2],
        upper_planes[0][1],
    ]
    lines = [
        "/BATCH",
        f"/FILNAME,{case_id},1",
        "/PREP7",
        "/UNITS,SI",
        "ET,1,SOLID70",
        "ET,2,TARGE170",
        "ET,3,CONTA174",
        "KEYOPT,3,1,2",
        "KEYOPT,3,12,5",
        "MP,KXX,1,5.0",
        "MP,KYY,1,5.0",
        "MP,KZZ,1,5.0",
        "MP,KXX,2,20.0",
        "MP,KYY,2,20.0",
        "MP,KZZ,2,20.0",
        "R,10",
        f"RMODIF,10,14,{G_W_m2K:.16e}",
        *lower_nodes,
        *upper_nodes,
        "TYPE,1",
        *lower_elements,
        *upper_elements,
        "TYPE,2",
        "REAL,10",
        f"E,{','.join(str(item) for item in target_nodes)}",
        "TYPE,3",
        "REAL,10",
        f"E,{','.join(str(item) for item in contact_nodes)}",
        f"NSEL,S,NODE,,{lower_planes[0][0]},{lower_planes[0][-1]}",
        "CM,COLD_FACE,NODE",
        f"NSEL,S,NODE,,{upper_planes[-1][0]},{upper_planes[-1][-1]}",
        "CM,HOT_FACE,NODE",
        f"NSEL,S,NODE,,{lower_planes[-1][0]},{lower_planes[-1][-1]}",
        "CM,LOWER_INTERFACE,NODE",
        f"NSEL,S,NODE,,{upper_planes[0][0]},{upper_planes[0][-1]}",
        "CM,UPPER_INTERFACE,NODE",
        "ALLSEL,ALL",
        f"SAVE,{case_id},db",
        "FINISH",
        "/CLEAR,NOSTART",
        f"/FILNAME,{case_id},1",
        f"RESUME,{case_id},db",
        "/PREP7",
        "*GET,K1RB,KZZ,1,TEMP,300",
        "*GET,K2RB,KZZ,2,TEMP,300",
        "FINISH",
        "/SOLU",
        "ANTYPE,STATIC",
        "CMSEL,S,COLD_FACE,NODE",
        "D,ALL,TEMP,300",
        "CMSEL,S,HOT_FACE,NODE",
        "D,ALL,TEMP,310",
        "ALLSEL,ALL",
        "OUTRES,ALL,ALL",
        "SOLVE",
        "FINISH",
        "/POST1",
        "SET,LAST",
        *reaction_sum_commands("COLD_FACE", "QMIN"),
        *reaction_sum_commands("HOT_FACE", "QMAX"),
        *average_temperature_commands("LOWER_INTERFACE", "TLOW"),
        *average_temperature_commands("UPPER_INTERFACE", "TUP"),
        "*CFOPEN,case_result,csv",
        "*VWRITE,K1RB,K2RB,QMIN,QMAX,TLOW,TUP",
        "(E24.16,',',E24.16,',',E24.16,',',E24.16,',',E24.16,',',E24.16)",
        "*CFCLOS",
        *profile_output_commands(),
        "FINISH",
        "/EXIT,NOSAVE",
    ]
    return "\n".join(lines) + "\n"


def perfect_contact_deck(cells_per_slab: int) -> str:
    case_id = f"mechanical_perfect_contact_{cells_per_slab}x2"
    length_m = 1.0e-6
    cross_m = 1.0e-6
    nodes, elements, planes = brick_chain(
        axis="z",
        start_m=-length_m,
        end_m=length_m,
        cross_m=cross_m,
        cells=2 * cells_per_slab,
        node_start=1,
        element_start=1,
        material=1,
    )
    upper_material_overrides = [
        f"EMODIF,{element_id},MAT,2"
        for element_id in range(
            cells_per_slab + 1,
            2 * cells_per_slab + 1,
        )
    ]
    lines = [
        "/BATCH",
        f"/FILNAME,{case_id},1",
        "/PREP7",
        "/UNITS,SI",
        "ET,1,SOLID70",
        "MP,KXX,1,5.0",
        "MP,KYY,1,5.0",
        "MP,KZZ,1,5.0",
        "MP,KXX,2,20.0",
        "MP,KYY,2,20.0",
        "MP,KZZ,2,20.0",
        *nodes,
        "TYPE,1",
        *elements,
        *upper_material_overrides,
        f"NSEL,S,NODE,,{planes[0][0]},{planes[0][-1]}",
        "CM,COLD_FACE,NODE",
        f"NSEL,S,NODE,,{planes[-1][0]},{planes[-1][-1]}",
        "CM,HOT_FACE,NODE",
        (
            f"NSEL,S,NODE,,{planes[cells_per_slab][0]},"
            f"{planes[cells_per_slab][-1]}"
        ),
        "CM,SHARED_INTERFACE,NODE",
        "ALLSEL,ALL",
        f"SAVE,{case_id},db",
        "FINISH",
        "/CLEAR,NOSTART",
        f"/FILNAME,{case_id},1",
        f"RESUME,{case_id},db",
        "/PREP7",
        "*GET,K1RB,KZZ,1,TEMP,300",
        "*GET,K2RB,KZZ,2,TEMP,300",
        "FINISH",
        "/SOLU",
        "ANTYPE,STATIC",
        "CMSEL,S,COLD_FACE,NODE",
        "D,ALL,TEMP,300",
        "CMSEL,S,HOT_FACE,NODE",
        "D,ALL,TEMP,310",
        "ALLSEL,ALL",
        "OUTRES,ALL,ALL",
        "SOLVE",
        "FINISH",
        "/POST1",
        "SET,LAST",
        *reaction_sum_commands("COLD_FACE", "QMIN"),
        *reaction_sum_commands("HOT_FACE", "QMAX"),
        *average_temperature_commands("SHARED_INTERFACE", "TINT"),
        "*CFOPEN,case_result,csv",
        "*VWRITE,K1RB,K2RB,QMIN,QMAX,TINT,TINT",
        "(E24.16,',',E24.16,',',E24.16,',',E24.16,',',E24.16,',',E24.16)",
        "*CFCLOS",
        *profile_output_commands(),
        "FINISH",
        "/EXIT,NOSAVE",
    ]
    return "\n".join(lines) + "\n"


def write_input_decks(output: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(("x", "y", "z")):
        case_id = f"mechanical_kappa_{axis}"
        case_dir = output / case_id
        case_dir.mkdir(parents=True)
        deck = case_dir / f"{case_id}.inp"
        deck.write_text(anisotropic_deck(axis))
        cases.append(
            {
                "case_id": case_id,
                "control": "anisotropic_kappa",
                "axis": axis,
                "axis_index": axis_index,
                "input_deck": str(deck),
            }
        )
    for conductance in INTERFACE_G_W_M2K:
        label = f"{conductance:.8g}".replace(".", "p").replace("+", "")
        case_id = f"mechanical_interface_G_{label}"
        case_dir = output / case_id
        case_dir.mkdir(parents=True)
        deck = case_dir / f"{case_id}.inp"
        deck.write_text(interface_deck(conductance))
        cases.append(
            {
                "case_id": case_id,
                "control": "internal_interface_G",
                "G_W_m2K": conductance,
                "input_deck": str(deck),
            }
        )
    for cells_per_slab in (10, 20, 40):
        case_id = f"mechanical_perfect_contact_{cells_per_slab}x2"
        case_dir = output / case_id
        case_dir.mkdir(parents=True)
        deck = case_dir / f"{case_id}.inp"
        deck.write_text(perfect_contact_deck(cells_per_slab))
        cases.append(
            {
                "case_id": case_id,
                "control": "perfect_contact",
                "perfect_contact": True,
                "cells_per_slab": cells_per_slab,
                "mesh_edge_z_m": 1.0e-6 / cells_per_slab,
                "input_deck": str(deck),
            }
        )
    return cases


def audit_input_decks(cases: list[dict[str, Any]]) -> dict[str, Any]:
    audited = []
    for case in cases:
        text = Path(case["input_deck"]).read_text()
        common_tokens = (
            "SAVE,",
            "/CLEAR,NOSTART",
            "RESUME,",
            "*GET",
            "RF,HEAT",
            "node_temperature,csv",
        )
        if case["control"] == "anisotropic_kappa":
            required = (
                *common_tokens,
                "MP,KXX,1,1.4400000000000000e+01",
                "MP,KYY,1,3.7999999999999998e+00",
                "MP,KZZ,1,1.0000000000000000e+00",
                "*GET,KXRB,KXX,1,TEMP,300",
                "*GET,KYRB,KYY,1,TEMP,300",
                "*GET,KZRB,KZZ,1,TEMP,300",
            )
        elif case["control"] == "internal_interface_G":
            required = (
                *common_tokens,
                "ET,2,TARGE170",
                "ET,3,CONTA174",
                "KEYOPT,3,1,2",
                "KEYOPT,3,12,5",
                "RMODIF,10,14,",
                "CM,LOWER_INTERFACE,NODE",
                "CM,UPPER_INTERFACE,NODE",
                "*GET,K1RB,KZZ,1,TEMP,300",
                "*GET,K2RB,KZZ,2,TEMP,300",
            )
        else:
            required = (
                *common_tokens,
                "CM,SHARED_INTERFACE,NODE",
                "*GET,K1RB,KZZ,1,TEMP,300",
                "*GET,K2RB,KZZ,2,TEMP,300",
            )
        missing = [token for token in required if token not in text]
        audited.append(
            {
                "case_id": case["case_id"],
                "passed": not missing,
                "missing_required_tokens": missing,
                "line_count": len(text.splitlines()),
                "sha256": sha256(Path(case["input_deck"])),
            }
        )
    passed = all(item["passed"] for item in audited)
    return {
        "status": (
            "PASSED_MECHANICAL_INPUT_DECK_STATIC_AUDIT"
            if passed
            else "FAILED_MECHANICAL_INPUT_DECK_STATIC_AUDIT"
        ),
        "passed": passed,
        "cases": audited,
    }


def run_mapdl_case(
    case: dict[str, Any],
    *,
    executable: Path,
    processor_count: int,
) -> dict[str, Any]:
    case_dir = Path(case["input_deck"]).parent
    jobname = case["case_id"]
    command = [
        str(executable),
        "-b",
        "-p",
        "ansys",
        "-np",
        str(processor_count),
        "-dir",
        str(case_dir),
        "-j",
        jobname,
        "-i",
        str(Path(case["input_deck"]).resolve()),
        "-o",
        str((case_dir / f"{jobname}.out").resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=case_dir,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    return {
        **case,
        "execution_command": shlex.join(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "solver_output": str(case_dir / f"{jobname}.out"),
        "case_result": str(case_dir / "case_result.csv"),
        "node_temperature": str(case_dir / "node_temperature.csv"),
    }


def read_numeric_row(path: Path, expected_count: int) -> list[float]:
    text = path.read_text().strip()
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != expected_count or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid numeric result in {path}: {values}")
    return values


def read_profile(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            [
                float(row["node_id"]),
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"]),
                float(row["temperature_K"]),
            ]
            for row in reader
        ]
    profile = np.asarray(rows, float)
    if profile.ndim != 2 or profile.shape[1] != 5:
        raise ValueError(f"invalid profile in {path}")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"NaN or Inf in {path}")
    return profile


def evaluate_anisotropic_case(case: dict[str, Any]) -> dict[str, Any]:
    case_result = Path(case["case_result"])
    profile_path = Path(case["node_temperature"])
    if case["returncode"] != 0 or not case_result.is_file() or not profile_path.is_file():
        return {
            **case,
            "status": "FAILED_MECHANICAL_SOLVER_EXECUTION",
            "passed": False,
        }
    kx, ky, kz, q_min, q_max = read_numeric_row(case_result, 5)
    readback = np.asarray([kx, ky, kz])
    profile = read_profile(profile_path)
    axis_index = int(case["axis_index"])
    coordinate = profile[:, axis_index + 1]
    unique_coordinate = np.unique(coordinate)
    mean_temperature = np.asarray(
        [
            np.mean(profile[np.isclose(coordinate, location), 4])
            for location in unique_coordinate
        ]
    )
    exact_temperature = 300.0 + 10.0 * unique_coordinate / 2.0e-6
    area_m2 = 1.0e-12
    numerical_flux = 0.5 * (abs(q_min) + abs(q_max)) / area_m2
    analytic_flux = KAPPA_W_MK[axis_index] * 10.0 / 2.0e-6
    flux_error = abs(numerical_flux - analytic_flux) / analytic_flux
    profile_error = np.max(abs(mean_temperature - exact_temperature)) / 10.0
    energy_error = abs(q_min + q_max) / max(abs(q_min), abs(q_max))
    readback_passed = np.allclose(
        readback, KAPPA_W_MK, rtol=0.0, atol=1.0e-12
    )
    passed = bool(
        readback_passed
        and flux_error < FLUX_ERROR_LIMIT
        and profile_error < PROFILE_ERROR_LIMIT
        and energy_error < ENERGY_ERROR_LIMIT
    )
    return {
        **case,
        "status": (
            "PASSED_MECHANICAL_ANISOTROPIC_K_CONTROL"
            if passed
            else "FAILED_MECHANICAL_ANISOTROPIC_K_CONTROL"
        ),
        "passed": passed,
        "property_readback_after_reload_W_mK": readback.tolist(),
        "property_readback_passed": bool(readback_passed),
        "analytic_heat_flux_W_m2": analytic_flux,
        "numerical_heat_flux_W_m2": numerical_flux,
        "heat_flux_relative_error": float(flux_error),
        "temperature_profile_relative_error": float(profile_error),
        "energy_balance_relative_error": float(energy_error),
    }


def evaluate_interface_case(case: dict[str, Any]) -> dict[str, Any]:
    case_result = Path(case["case_result"])
    profile_path = Path(case["node_temperature"])
    if case["returncode"] != 0 or not case_result.is_file() or not profile_path.is_file():
        return {
            **case,
            "status": "FAILED_MECHANICAL_SOLVER_EXECUTION",
            "passed": False,
        }
    k1, k2, q_min, q_max, t_lower, t_upper = read_numeric_row(
        case_result, 6
    )
    read_profile(profile_path)
    perfect_contact = bool(case.get("perfect_contact", False))
    G = None if perfect_contact else float(case["G_W_m2K"])
    analytic_flux = 10.0 / (
        1.0e-6 / 5.0
        + (0.0 if perfect_contact else 1.0 / G)
        + 1.0e-6 / 20.0
    )
    analytic_jump = 0.0 if perfect_contact else analytic_flux / G
    area_m2 = 1.0e-12
    numerical_flux = 0.5 * (abs(q_min) + abs(q_max)) / area_m2
    numerical_jump = abs(t_upper - t_lower)
    flux_error = abs(numerical_flux - analytic_flux) / analytic_flux
    jump_error = (
        numerical_jump
        if perfect_contact
        else abs(numerical_jump - analytic_jump) / analytic_jump
    )
    energy_error = abs(q_min + q_max) / max(abs(q_min), abs(q_max))
    readback_passed = abs(k1 - 5.0) < 1.0e-12 and abs(k2 - 20.0) < 1.0e-12
    passed = bool(
        readback_passed
        and flux_error < FLUX_ERROR_LIMIT
        and (
            numerical_jump < 1.0e-8
            if perfect_contact
            else jump_error < INTERFACE_JUMP_ERROR_LIMIT
        )
        and energy_error < ENERGY_ERROR_LIMIT
    )
    if perfect_contact:
        status = (
            "PASSED_MECHANICAL_PERFECT_CONTACT_CONTROL"
            if passed
            else "FAILED_MECHANICAL_PERFECT_CONTACT_CONTROL"
        )
    else:
        status = (
            "PASSED_MECHANICAL_INTERFACE_G_CONTROL"
            if passed
            else "FAILED_MECHANICAL_INTERFACE_G_CONTROL"
        )
    return {
        **case,
        "status": status,
        "passed": passed,
        "material_readback_after_reload_W_mK": [k1, k2],
        "property_readback_passed": readback_passed,
        "analytic_heat_flux_W_m2": analytic_flux,
        "numerical_heat_flux_W_m2": numerical_flux,
        "heat_flux_relative_error": flux_error,
        "expected_interface_temperature_jump_K": analytic_jump,
        "numerical_interface_temperature_jump_K": numerical_jump,
        "interface_temperature_jump_relative_error": jump_error,
        "energy_balance_relative_error": energy_error,
    }


def offline_references(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = []
    for case in cases:
        if case["control"] == "anisotropic_kappa":
            index = int(case["axis_index"])
            references.append(
                {
                    "case_id": case["case_id"],
                    "control": case["control"],
                    "solver_executed": False,
                    "status": "OFFLINE_ANALYTIC_REFERENCE_ONLY",
                    "expected_kappa_W_mK": float(KAPPA_W_MK[index]),
                    "analytic_heat_flux_W_m2": float(
                        KAPPA_W_MK[index] * 10.0 / 2.0e-6
                    ),
                    "exact_temperature_profile": (
                        "T=300+10*s/(2e-6), where s is the active-axis coordinate"
                    ),
                    "input_deck": case["input_deck"],
                }
            )
        elif case["control"] == "internal_interface_G":
            G = float(case["G_W_m2K"])
            flux = 10.0 / (
                1.0e-6 / 5.0 + 1.0 / G + 1.0e-6 / 20.0
            )
            references.append(
                {
                    "case_id": case["case_id"],
                    "control": case["control"],
                    "solver_executed": False,
                    "status": "OFFLINE_ANALYTIC_REFERENCE_ONLY",
                    "G_W_m2K": G,
                    "thermal_insulance_m2K_W": 1.0 / G,
                    "analytic_heat_flux_W_m2": flux,
                    "expected_interface_temperature_jump_K": flux / G,
                    "input_deck": case["input_deck"],
                }
            )
        else:
            flux = 10.0 / (1.0e-6 / 5.0 + 1.0e-6 / 20.0)
            references.append(
                {
                    "case_id": case["case_id"],
                    "control": case["control"],
                    "solver_executed": False,
                    "status": "OFFLINE_ANALYTIC_REFERENCE_ONLY",
                    "perfect_contact": True,
                    "cells_per_slab": case["cells_per_slab"],
                    "mesh_edge_z_m": case["mesh_edge_z_m"],
                    "analytic_heat_flux_W_m2": flux,
                    "expected_interface_temperature_jump_K": 0.0,
                    "input_deck": case["input_deck"],
                }
            )
    return references


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reports(
    *,
    output: Path,
    report_dir: Path,
    summary: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "mechanical_thermal_controls_summary.json", summary)
    fields = (
        "case_id",
        "control",
        "status",
        "passed",
        "solver_executed",
        "axis",
        "G_W_m2K",
        "analytic_heat_flux_W_m2",
        "numerical_heat_flux_W_m2",
        "heat_flux_relative_error",
        "expected_interface_temperature_jump_K",
        "numerical_interface_temperature_jump_K",
        "interface_temperature_jump_relative_error",
        "temperature_profile_relative_error",
        "energy_balance_relative_error",
        "input_deck",
    )
    with (report_dir / "mechanical_thermal_controls_cases.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        for case in summary["cases"]:
            writer.writerow({key: case.get(key, "") for key in fields})

    native_probe = summary["mechanical_probe"]
    kappa_lines = [
        "# Mechanical anisotropic-kappa solver report",
        "",
        f"**Status: `{summary['anisotropic_kappa_status']}`.**",
        "",
        "Mechanical/MAPDL officially supports orthotropic conductivity through",
        "`MP,KXX`, `MP,KYY`, and `MP,KZZ`; the generated controls assign",
        "`diag(14.4, 3.8, 1.0) W/(m K)` to SOLID70 bricks and independently",
        "drive heat in x, y, and z.",
        "",
        f"- Executable probe: `{native_probe['executable']['status']}`",
        f"- License probe: `{native_probe['license']['status']}`",
        f"- Actual Mechanical solver executed: `{summary['solver_executed']}`",
        "- Canonical input-deck static audit: "
        f"`{summary['input_deck_static_audit']['status']}`",
        "- Isotropic average used: `false`",
        "",
        "The input decks include database `SAVE`, `/CLEAR`, `RESUME`, material",
        "readback, boundary reaction heat flow, nodal temperature output, and",
        "energy-balance quantities. On this server they were not executed",
        "because neither a MAPDL executable nor a Mechanical license feature is",
        "available. Offline analytic values are references only.",
        "",
        "To execute after installing MAPDL and adding an `ansys`, `mech_1`,",
        "`mech_2`, or `struct` solver license:",
        "",
        "```bash",
        "python photothermal_pte/validation/photothermal_stage1/32_validate_mechanical_thermal_controls.py \\",
        "  --executable /path/to/ansys261",
        "```",
        "",
        "Official documentation:",
        "https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/ans_mat/thermalmat.html",
        "https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/ans_cmd/Hlp_C_MP.html",
        "",
    ]
    (report_dir / "MECHANICAL_ANISOTROPIC_K_SOLVER_REPORT.md").write_text(
        "\n".join(kappa_lines)
    )

    interface_lines = [
        "# Mechanical internal-interface-G solver report",
        "",
        f"**Status: `{summary['interface_G_status']}`.**",
        "",
        "The generated two-slab controls use separate coincident meshes joined",
        "by TARGE170/CONTA174. `KEYOPT(1)=2` selects pure thermal contact,",
        "`KEYOPT(12)=5` keeps the interface bonded, and real constant 14",
        "sets `TCC=G` in W/(m2 K). Cases are generated for `7.37e6` and",
        "`1.1e9 W/(m2 K)`.",
        "",
        "- Canonical input-deck static audit: "
        f"`{summary['input_deck_static_audit']['status']}`",
        "- Actual Mechanical solver executed: "
        f"`{summary['solver_executed']}`",
        "",
        "Perfect-contact controls use a shared-node material interface at",
        "100, 50, and 25 nm axial mesh spacing; the expected temperature jump",
        "is exactly zero and the analytic heat flux is `4.0e7 W/m2`.",
        "",
        "The solver-side acceptance criteria are `<1%` for transmitted heat",
        "flux, `Delta T=q''/G`, and global energy balance. These criteria have",
        "not been evaluated by Mechanical on this server because the executable",
        "and license are absent.",
        "",
        "The same execution command runs the finite-G and perfect-contact",
        "controls together with the anisotropic controls.",
        "",
        "Official documentation:",
        "https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/pdf/ANSYS_Mechanical_APDL_Contact_Technology_Guide.pdf",
        "https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/ans_elem/Hlp_E_CONTA174.html",
        "",
    ]
    (report_dir / "MECHANICAL_INTERNAL_INTERFACE_G_SOLVER_REPORT.md").write_text(
        "\n".join(interface_lines)
    )

    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        artifacts.append(
            {
                "category": "raw_mechanical_control",
                "server_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tracked_in_git": False,
            }
        )
    for name in (
        "MECHANICAL_ANISOTROPIC_K_SOLVER_REPORT.md",
        "MECHANICAL_INTERNAL_INTERFACE_G_SOLVER_REPORT.md",
        "mechanical_thermal_controls_summary.json",
        "mechanical_thermal_controls_cases.csv",
    ):
        path = report_dir / name
        artifacts.append(
            {
                "category": "report",
                "repository_path": str(
                    path.resolve().relative_to(config.REPOSITORY_ROOT.parent)
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tracked_in_git": True,
            }
        )
    write_json(
        report_dir / "RAW_ARTIFACT_MANIFEST.json",
        {
            "schema_version": 1,
            "generated_at_utc": utc_timestamp(),
            "status": summary["status"],
            "generated_by": (
                "photothermal_pte/validation/photothermal_stage1/"
                "32_validate_mechanical_thermal_controls.py"
            ),
            "artifacts": artifacts,
        },
    )


def main() -> int:
    args = parse_args()
    if args.np < 1:
        raise ValueError("--np must be positive")
    output = output_directory(args.output_dir)
    executable_probe = find_mapdl_executable(args.executable)
    license_probe = probe_license_features()
    cases = write_input_decks(output)
    can_run = bool(
        executable_probe["passed"]
        and (license_probe["passed"] or args.force_run)
    )
    if can_run:
        executable = Path(executable_probe["executable"])
        executed = [
            run_mapdl_case(
                case,
                executable=executable,
                processor_count=args.np,
            )
            for case in cases
        ]
        evaluated = [
            evaluate_anisotropic_case(case)
            if case["control"] == "anisotropic_kappa"
            else evaluate_interface_case(case)
            for case in executed
        ]
    else:
        evaluated = offline_references(cases)
    kappa_cases = [
        case for case in evaluated if case["control"] == "anisotropic_kappa"
    ]
    interface_cases = [
        case
        for case in evaluated
        if case["control"] in ("internal_interface_G", "perfect_contact")
    ]
    if not can_run:
        overall_status = (
            executable_probe["status"]
            if not executable_probe["passed"]
            else license_probe["status"]
        )
        kappa_status = overall_status
        interface_status = overall_status
    else:
        kappa_status = (
            "VALIDATED_MECHANICAL_ANISOTROPIC_K"
            if all(case["passed"] for case in kappa_cases)
            else "FAILED_MECHANICAL_ANISOTROPIC_K_CONTROL"
        )
        interface_status = (
            "VALIDATED_MECHANICAL_INTERFACE_G"
            if all(case["passed"] for case in interface_cases)
            else "FAILED_MECHANICAL_INTERFACE_G_CONTROL"
        )
        overall_status = (
            "VALIDATED_MECHANICAL_THERMAL_CONTROLS"
            if kappa_status == "VALIDATED_MECHANICAL_ANISOTROPIC_K"
            and interface_status == "VALIDATED_MECHANICAL_INTERFACE_G"
            else "FAILED_MECHANICAL_THERMAL_CONTROLS"
        )
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "status": overall_status,
        "anisotropic_kappa_status": kappa_status,
        "interface_G_status": interface_status,
        "solver_executed": can_run,
        "solver_validation_claimed": bool(
            overall_status == "VALIDATED_MECHANICAL_THERMAL_CONTROLS"
        ),
        "mechanical_probe": {
            "executable": executable_probe,
            "license": license_probe,
        },
        "input_deck_static_audit": audit_input_decks(cases),
        "requested_tensor_W_mK": KAPPA_W_MK.tolist(),
        "requested_interface_G_W_m2K": list(INTERFACE_G_W_M2K),
        "isotropic_average_used": False,
        "full_device_executed": False,
        "cases": evaluated,
        "generation_command": shlex.join([sys.executable, *sys.argv]),
    }
    write_json(output / "mechanical_thermal_controls_result.json", summary)
    write_reports(
        output=output,
        report_dir=Path(args.report_dir).expanduser().resolve(),
        summary=summary,
    )
    print(json.dumps(jsonable(summary), indent=2))
    return 0 if can_run and summary["solver_validation_claimed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
