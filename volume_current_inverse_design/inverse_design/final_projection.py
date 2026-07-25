#!/usr/bin/env python3
"""Final manufacturing approver: exact binary -> independent DRC -> exact FDTD.

Transactional & provenance-checked (review P0-4/P0-5, P2-3/P2-4):
  * refuses to finalise a design whose code hash != the current code, or that
    was not flagged had_feasible by the optimiser;
  * deletes any stale SUCCESS.json/exact_binary_fom.json at entry and writes a
    status="pending" manifest, so a failed re-run can never leave a prior
    success in place; SUCCESS.json is written LAST, only when everything passes;
  * SUCCESS.json separates boolean pass flags from measured floats and records
    sha256 of every emitted artifact.

Steps: best-feasible latent -> exact binary (filter>=0.5, beta-independent) ->
independent geometry DRC -> (pass only) exact {0,1} physical mask -> exact FDTD
Fx/Fy/Fx+Fy.  DRC fail -> nonzero exit, no SUCCESS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bundle"))

from geometry_drc import geometry_drc  # noqa: E402
from mapping_diagnostics import mapping_diagnostics  # noqa: E402
from run_constrained_inverse_design import _code_hash, production_code_files  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _exact_binary_unique(mapping, latent) -> np.ndarray:
    filtered = np.asarray(mapping.filter_unique(latent), float)
    return (filtered >= 0.5).astype(np.uint8)


def _mapping_from_env():
    """Build the production mapping directly from env -- no Lumerical import.

    Mirrors tairte4_volume_model's grid/mapping construction so the finalizer
    (esp. --no-fdtd) runs without the Lumerical install, and so its mapping
    identity matches the one the runner saved.
    """
    from periodic_constrained_mapping import PeriodicConstrainedMapping
    period = float(os.environ.get("PERIOD_UM", "6.0"))
    res_xy = int(os.environ.get("DESIGN_RES_XY", "40"))
    res = int(os.environ.get("RES", "20"))
    design_h = float(os.environ.get("DESIGN_H_UM", "0.6"))
    mfs = float(os.environ.get("MFS_UM", "0.5"))
    Nx = int(round(period * res_xy)) + 1
    Nz = int(round(design_h * res)) + 1
    radius = float(os.environ.get("VC_FILTER_RADIUS_UM", str(mfs)))
    eta_e = float(os.environ.get("VC_ETA_ERODE", "0.75"))
    eta_d = float(os.environ.get("VC_ETA_DILATE", "0.25"))
    return PeriodicConstrainedMapping(Nx, Nx, Nz, period, radius, eta_d, eta_e)


def _identity(mapping, mfs, mgs) -> str:
    return json.dumps({
        "mapping_config": mapping.config.to_dict(),
        "isolation_gap_um": float(mapping.isolation_gap_um),
        "mfs_um": mfs, "mgs_um": mgs,
        "mapping_mode": os.environ.get("MSOPT_MAPPING", "periodic_constrained"),
    }, sort_keys=True)


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", help="NPZ with the best feasible 'latent' array")
    ap.add_argument("--output", default="runs/final_projection")
    ap.add_argument("--mfs-um", type=float, default=0.5)
    ap.add_argument("--mgs-um", type=float, default=0.5)
    ap.add_argument("--min-gap-um", type=float, default=None)
    ap.add_argument("--gpu", default=os.environ.get("CL_GPU_DEVICE", "GPU 1"))
    ap.add_argument("--no-fdtd", action="store_true")
    ap.add_argument("--allow-unsafe-provenance", action="store_true",
                    help="(unsafe) skip had_feasible/code_hash/identity provenance checks")
    args = ap.parse_args()
    os.environ["MFS_UM"] = str(args.mfs_um)
    os.environ["MGS_UM"] = str(args.mgs_um)
    os.environ.setdefault("MSOPT_MAPPING", "periodic_constrained")
    os.environ["CL_GPU_DEVICE"] = args.gpu
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu

    if os.environ["MSOPT_MAPPING"] != "periodic_constrained":
        raise SystemExit("final_projection requires MSOPT_MAPPING=periodic_constrained")
    # Mapping is built standalone (no Lumerical); only the FDTD path imports the
    # evaluator.  So --no-fdtd runs without the Lumerical install (P1-5).
    mapping = _mapping_from_env()

    data = np.load(Path(args.design).resolve())
    if "latent" not in data.files:
        raise KeyError(f"{args.design} has no 'latent' array")
    latent = np.asarray(data["latent"], float).reshape(-1)
    if latent.size != mapping.Nux * mapping.Nuy:
        raise ValueError(f"latent size {latent.size} != Nux*Nuy {mapping.Nux*mapping.Nuy}")

    output = Path(args.output).expanduser().resolve()   # P0-1: CWD-relative, matches launcher
    output.mkdir(parents=True, exist_ok=True)
    # transaction start: remove any stale success, write pending manifest
    for stale in ("SUCCESS.json", "exact_binary_fom.json"):
        try:
            (output / stale).unlink()
        except FileNotFoundError:
            pass
    _write(output / "final_manifest.json", {"status": "pending"})

    def fail(category: str, msg: str, extra=None):
        man = {"status": "failed", "category": category, "message": msg}
        if extra:
            man.update(extra)
        _write(output / "final_manifest.json", man)
        print(f"FINALIZE FAILED [{category}]: {msg}")
        raise SystemExit(2)

    # --- provenance: REQUIRE had_feasible(True), code_hash(match), identity(match) ---
    if not args.allow_unsafe_provenance:
        if "had_feasible" not in data.files:
            fail("missing_provenance", "design NPZ has no had_feasible field")
        if bool(data["had_feasible"]) is not True:
            fail("not_feasible", "design was not flagged had_feasible by the optimiser")
        if "code_hash" not in data.files:
            fail("missing_provenance", "design NPZ has no code_hash field")
        current_hash = _code_hash(production_code_files(ROOT, HERE))
        design_hash = str(data["code_hash"])
        if design_hash != current_hash:
            fail("code_hash_mismatch",
                 f"design code_hash {design_hash} != current {current_hash}",
                 {"design_code_hash": design_hash, "current_code_hash": current_hash})
        # P0-6: mapping/mode/isolation/MFS identity must match
        current_identity = _identity(mapping, args.mfs_um, args.mgs_um)
        if "mapping_identity" not in data.files:
            fail("missing_provenance", "design NPZ has no mapping_identity field")
        design_identity = str(data["mapping_identity"])
        if design_identity != current_identity:
            fail("config_mismatch",
                 "mapping/mode/isolation/MFS differs from the design's",
                 {"design_identity": design_identity, "current_identity": current_identity})
    else:
        current_hash = _code_hash(production_code_files(ROOT, HERE))
        design_hash = str(data["code_hash"]) if "code_hash" in data.files else None

    # --- exact binary mask on the unique grid (beta-independent) ---
    mask_u = _exact_binary_unique(mapping, latent)
    spacing_um = mapping.config.dx_um
    if not np.isclose(mapping.config.dx_um, mapping.config.dy_um):
        fail("nonsquare_pixels", "DRC assumes square design pixels")

    drc = geometry_drc(mask_u, spacing_um=spacing_um,
                       min_solid_width_um=args.mfs_um, min_void_width_um=args.mgs_um,
                       min_gap_um=args.min_gap_um)
    _write(output / "geometry_drc.json", drc)
    print(json.dumps(drc, indent=2))

    mp = mapping
    mask_phys = np.asarray(mp.physical_from_unique(mask_u.astype(float)), float).reshape(
        mapping.Nx, mapping.Ny, mapping.Nz).astype(np.uint8)
    diag = mapping_diagnostics(mask_phys.astype(float))
    if not diag.is_exact_binary:
        fail("not_binary", "exact-binary mask is not exactly {0,1}")

    np.savez_compressed(output / "final_mask_unique.npz", mask=mask_u,
                        spacing_um=np.array(spacing_um))
    np.savez_compressed(output / "final_mask_physical.npz", mask=mask_phys,
                        physical=mask_phys.astype(float))

    if not drc["pass"]:
        np.savez_compressed(output / "final_candidate_failed_drc.npz",
                            mask=mask_u, latent=latent)
        fail("geometry_infeasible",
             "minimum solid/void/gap below rule (or trivial topology)",
             {"drc": drc})

    if args.no_fdtd:
        _write(output / "final_manifest.json",
               {"status": "drc_pass_fdtd_skipped", "drc_pass": True, "drc": drc,
                "note": "SUCCESS requires the exact-binary FDTD FOM (omit --no-fdtd)"})
        print("DRC PASS; FDTD skipped (--no-fdtd). No SUCCESS until FOM evaluated.")
        return

    # --- exact-binary FDTD (delta=0 forward on the {0,1} mask) ---
    from volume_current_evaluator import VolumeCurrentEvaluator
    fom = {}
    try:
        for pol in ("x", "y"):
            ev = VolumeCurrentEvaluator(output / f"solver_{pol}", 0.001, pol)
            ev.prepare(force_rebuild=False)
            fom[f"F{pol}"] = float(ev.forward_fom(mask_phys.astype(float), label="final_binary"))
    except Exception as exc:  # noqa: BLE001
        fail("fdtd_failure", f"exact-binary FDTD failed: {type(exc).__name__}: {exc}")
    fom["F_sum"] = fom["Fx"] + fom["Fy"]
    _write(output / "exact_binary_fom.json", {"exact_binary": True, **fom})

    # --- artifact hashes (P2-4) ---
    artifacts = {}
    for name in ("final_mask_unique.npz", "final_mask_physical.npz",
                 "geometry_drc.json", "exact_binary_fom.json"):
        p = output / name
        if p.exists():
            artifacts[name] = _sha256(p)

    manifest = {
        "status": "completed",
        "exact_binary": True,
        "periodic_fencepost": float(diag.periodic_x_max_abs_error) == 0.0
        and float(diag.periodic_y_max_abs_error) == 0.0,
        "z_invariant": float(diag.z_extrusion_max_abs_error) == 0.0,
        "drc_pass": True, "drc": drc,
        "exact_binary_fom": fom,
        "solid_fraction": float(mask_u.mean()),
        "design_code_hash": design_hash, "current_code_hash": current_hash,
        "had_feasible": had_feasible,
        "artifact_sha256": artifacts,
    }
    _write(output / "final_manifest.json", manifest)

    success = {
        # boolean pass flags (P2-3)
        "exact_binary": True,
        "periodic_fencepost": manifest["periodic_fencepost"],
        "z_invariant": manifest["z_invariant"],
        "solid_width_500nm_drc": True,
        "void_width_500nm_drc": True,
        "gap_500nm_drc": True if args.min_gap_um is not None else "not_applicable",
        "exact_binary_fom_evaluated": True,
        # measurements (kept separate from the pass flags)
        "minimum_solid_width_um": drc["minimum_solid_width_um"],
        "minimum_void_width_um": drc["minimum_void_width_um"],
        "minimum_gap_um": drc["minimum_gap_um"],
        "Fx": fom["Fx"], "Fy": fom["Fy"], "F_sum": fom["F_sum"],
        # provenance
        "current_code_hash": current_hash, "design_code_hash": design_hash,
        "artifact_sha256": artifacts,
    }
    _write(output / "SUCCESS.json", success)   # written LAST
    print("SUCCESS: exact binary passed DRC and exact-binary FOM evaluated.")
    print(json.dumps({"Fx": fom["Fx"], "Fy": fom["Fy"], "F_sum": fom["F_sum"]}, indent=2))


if __name__ == "__main__":
    main()
