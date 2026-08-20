#!/usr/bin/env python3
"""Summarize the Au material checkpoint without promoting blocked gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    contract = json.loads((RESULTS / "au_material_contract.json").read_text())
    readback_path = RESULTS / "lumerical_au_readback.json"
    readback = json.loads(readback_path.read_text()) if readback_path.exists() else {
        "status": "PENDING_LUMERICAL_AU_MATERIAL_READBACK",
        "FDTD_solve_run": False,
        "GPU_engine_acquired": False,
    }

    readback_status = readback["status"]
    if readback_status == "VALIDATED_LUMERICAL_AU_MATERIAL_READBACK":
        overall = "VALIDATED_AU_MATERIAL_READBACK_DENSITY_PATH_NOT_YET_CERTIFIED"
    elif readback_status.startswith("BLOCKED_"):
        overall = readback_status
    else:
        overall = "PENDING_LUMERICAL_AU_MATERIAL_READBACK"

    report = f"""# Au-on-fixed-TaIrTe4 validation checkpoint

Status: `{overall}`

## What is complete

- Frozen wavelength: `10 um`.
- Ordal Au endpoint: `n+ik = 12.1 + 69.2i`.
- Relative permittivity: `-4642.23 + 1674.64i`.
- Bulk-reference electrical conductivity: `{contract['transport_references_at_approximately_300K']['electrical_conductivity_S_m']:.9g} S/m`.
- Bulk-reference thermal conductivity: `{contract['transport_references_at_approximately_300K']['thermal_conductivity_W_mK']:.9g} W/(m K)`.
- Candidate density law: interpolate complex `n`, then use `epsilon=n^2`.
- Offline endpoint, analytic derivative, passivity, and JVP/VJP unit tests pass.

The Au transport values are reference scenarios, not certified thin-film or
Au/TaIrTe4 contact properties. The first electrical control will use
`S_Au=0`; nonzero Au thermopower remains a sensitivity case.

## Lumerical readback

- Status: `{readback_status}`.
- FDTD solve executed: `{readback.get('FDTD_solve_run', False)}`.
- GPU engine acquired: `{readback.get('GPU_engine_acquired', False)}`.
- Installation in the retained result: `{readback.get('lumerical_root', 'not opened')}`.

Both the `/opt/lumerical/v261` installation and the user-owned v261
installation were attempted. Session startup stopped before material import
because ANSYSLI did not create/read its license-sharing port file. This is not
a material-fit failure and it is not an optical validation pass.

## What is deliberately not claimed

- No binary Au/air Maxwell control has run.
- No Au density optical AD-FD has run.
- No Au/TaIrTe4 thermal or electrical contact has been selected or validated.
- No electrode weighting-field gradient has been certified.
- No Au topology optimization has started.

The approved fallback to sharp-interface level-set/shape optimization is used
only if the density route fails material readback, binary endpoint equivalence,
or AD-FD after the license/API gate is restored.
"""
    report_path = RESULTS / "AU_VALIDATION_CHECKPOINT.md"
    report_path.write_text(report)

    artifact_names = [
        "au_material_contract.json",
        "au_density_paths.csv",
        "au_density_interpolation_audit.png",
        "AU_MATERIAL_AND_INTERPOLATION_AUDIT.md",
        "lumerical_au_readback.json",
        "AU_VALIDATION_CHECKPOINT.md",
    ]
    files = []
    for name in artifact_names:
        path = RESULTS / name
        if path.exists():
            files.append({
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "status": overall,
        "FDTD_solve_run": bool(readback.get("FDTD_solve_run", False)),
        "GPU_engine_acquired": bool(readback.get("GPU_engine_acquired", False)),
        "files": files,
        "raw_FSP_or_NPZ_committed": False,
    }
    (RESULTS / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": overall, "artifacts": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
