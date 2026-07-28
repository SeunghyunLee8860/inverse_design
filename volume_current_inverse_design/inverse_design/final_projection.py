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
    ap.add_argument("--drc-cleanup", action="store_true",
                    help="if the raw exact-binary mask fails DRC, apply the "
                    "deterministic drc_cleanup and judge the cleaned mask "
                    "(recorded in the manifest/SUCCESS; a no-op when raw DRC passes)")
    ap.add_argument("--accept-design-code-hash", default=None, metavar="HASH",
                    help="explicitly accept a design produced by this exact older "
                    "code_hash (auditable, narrow alternative to "
                    "--allow-unsafe-provenance; all other strict checks still run)")
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

    # --- provenance ---
    # All provenance values are read up front so they are ALWAYS bound (strict AND
    # unsafe paths) for the manifest/SUCCESS.  A normal runner NPZ always carries
    # had_feasible, code_hash, config_hash, attempt, mapping_identity; strict mode
    # requires every one of them, unsafe mode allows None but records both the
    # design's and the current mapping identity separately.
    def _get(name):
        return data[name] if name in data.files else None
    had_feasible = bool(_get("had_feasible")) if _get("had_feasible") is not None else None
    design_hash = str(_get("code_hash")) if _get("code_hash") is not None else None
    design_config_hash = str(_get("config_hash")) if _get("config_hash") is not None else None
    design_attempt = int(_get("attempt")) if _get("attempt") is not None else None
    design_identity = str(_get("mapping_identity")) if _get("mapping_identity") is not None else None
    current_hash = _code_hash(production_code_files(ROOT, HERE))
    current_identity = _identity(mapping, args.mfs_um, args.mgs_um)

    if not args.allow_unsafe_provenance:
        for field in ("had_feasible", "code_hash", "config_hash", "attempt",
                      "mapping_identity"):
            if field not in data.files:
                fail("missing_provenance", f"design NPZ has no {field} field "
                     "(a normal run always writes it; use --allow-unsafe-provenance "
                     "only for hand-built designs)")
        if had_feasible is not True:
            fail("not_feasible", "design was not flagged had_feasible by the optimiser")
        if design_hash != current_hash and design_hash != args.accept_design_code_hash:
            fail("code_hash_mismatch",
                 f"design code_hash {design_hash} != current {current_hash} "
                 "(pass --accept-design-code-hash to finalise a design from an "
                 "audited older commit)",
                 {"design_code_hash": design_hash, "current_code_hash": current_hash})
        # P0-6: mapping/mode/isolation/MFS identity must match
        if design_identity != current_identity:
            fail("config_mismatch",
                 "mapping/mode/isolation/MFS differs from the design's",
                 {"design_mapping_identity": design_identity,
                  "current_mapping_identity": current_identity})

    # --- exact binary mask on the unique grid (beta-independent) ---
    mask_u = _exact_binary_unique(mapping, latent)
    spacing_um = mapping.config.dx_um
    if not np.isclose(mapping.config.dx_um, mapping.config.dy_um):
        fail("nonsquare_pixels", "DRC assumes square design pixels")

    drc = geometry_drc(mask_u, spacing_um=spacing_um,
                       min_solid_width_um=args.mfs_um, min_void_width_um=args.mgs_um,
                       min_gap_um=args.min_gap_um)
    cleanup_block = None
    if args.drc_cleanup and not drc["pass"]:
        # Deterministic DRC-driven cleanup (see drc_cleanup.py header for the
        # measured rationale).  The cleaned mask is judged by the SAME
        # independent DRC below; nothing about the SUCCESS gates is weakened.
        from drc_cleanup import drc_cleanup as _drc_cleanup
        _write(output / "geometry_drc_precleanup.json", drc)
        np.savez_compressed(output / "final_mask_unique_precleanup.npz",
                            mask=mask_u, spacing_um=np.array(spacing_um))
        res = _drc_cleanup(mask_u, spacing_um=spacing_um,
                           min_solid_width_um=args.mfs_um,
                           min_void_width_um=args.mgs_um,
                           min_gap_um=args.min_gap_um)
        mask_u = np.asarray(res["mask"], np.uint8)
        drc = res["drc"]
        cleanup_block = {
            "applied": True, "version": res["version"],
            "pixels_changed": res["pixels_changed"],
            "pixels_changed_fraction": res["pixels_changed_fraction"],
            "stages": res["stages"],
            "precleanup_mask_sha256": _sha256(output / "final_mask_unique_precleanup.npz"),
        }
        _write(output / "drc_cleanup.json", cleanup_block)
        print(f"[drc-cleanup] raw mask failed DRC -> cleanup: pass={res['pass']} "
              f"pixels_changed={res['pixels_changed']} "
              f"({res['pixels_changed_fraction']*100:.2f}%)")
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
                 "geometry_drc.json", "exact_binary_fom.json",
                 "drc_cleanup.json", "geometry_drc_precleanup.json",
                 "final_mask_unique_precleanup.npz"):
        p = output / name
        if p.exists():
            artifacts[name] = _sha256(p)

    # P1-5: full provenance in the manifest/SUCCESS (scientific reproducibility)
    manifest = {
        "status": "completed",
        "exact_binary": True,
        "periodic_fencepost": float(diag.periodic_x_max_abs_error) == 0.0
        and float(diag.periodic_y_max_abs_error) == 0.0,
        "z_invariant": float(diag.z_extrusion_max_abs_error) == 0.0,
        "drc_pass": True, "drc": drc,
        "drc_cleanup": cleanup_block,
        "exact_binary_fom": fom,
        "solid_fraction": float(mask_u.mean()),
        "accepted_design_code_hash": args.accept_design_code_hash,
        "design_code_hash": design_hash, "current_code_hash": current_hash,
        "design_config_hash": design_config_hash, "design_attempt": design_attempt,
        "design_mapping_identity": design_identity,
        "current_mapping_identity": current_identity,
        "mapping_identity": design_identity if design_identity is not None else current_identity,
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
        # measurements (kept separate from the pass flags); *_capped=True means the
        # value is a LOWER BOUND (feature wider than the search cap), not exact (#1).
        "minimum_solid_width_um": drc["minimum_solid_width_um"],
        "minimum_solid_width_capped": drc.get("minimum_solid_width_capped", False),
        "minimum_void_width_um": drc["minimum_void_width_um"],
        "minimum_void_width_capped": drc.get("minimum_void_width_capped", False),
        "minimum_gap_um": drc["minimum_gap_um"],
        "minimum_gap_capped": drc.get("minimum_gap_capped", False),
        "Fx": fom["Fx"], "Fy": fom["Fy"], "F_sum": fom["F_sum"],
        "drc_cleanup": cleanup_block,
        # provenance (P1-5)
        "accepted_design_code_hash": args.accept_design_code_hash,
        "current_code_hash": current_hash, "design_code_hash": design_hash,
        "design_config_hash": design_config_hash, "design_attempt": design_attempt,
        "design_mapping_identity": design_identity,
        "current_mapping_identity": current_identity,
        "mapping_identity": manifest["mapping_identity"],
        "artifact_sha256": artifacts,
    }
    _write(output / "SUCCESS.json", success)   # written LAST
    print("SUCCESS: exact binary passed DRC and exact-binary FOM evaluated.")
    print(json.dumps({"Fx": fom["Fx"], "Fy": fom["Fy"], "F_sum": fom["F_sum"]}, indent=2))


if __name__ == "__main__":
    main()
