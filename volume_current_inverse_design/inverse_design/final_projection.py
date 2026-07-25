#!/usr/bin/env python3
"""Final manufacturing approver: exact binary -> independent DRC -> exact FDTD.

Replaces the old "smallest beta with binarization>=target" chooser, which saved
a high-beta *continuous* array under a ``binary`` name and never measured
MFS/MGS.  Here:

  1. load the best FEASIBLE latent,
  2. form the exact binary mask  ``(filter(latent) >= 0.5)``  on the 240x240
     unique grid (beta-independent: tanh(.,.,0.5) preserves the 0.5 crossing),
  3. run the INDEPENDENT periodic geometry DRC (geometry_drc.py),
  4. only if DRC passes, extrude to the 241x241x13 exact {0,1} physical mask and
     evaluate Fx, Fy, Fx+Fy with a forward FDTD solve (exact density, delta=0),
  5. write a SUCCESS manifest only when the mask is exactly binary AND DRC passes
     AND the exact-binary FOM was evaluated.

DRC failure -> nonzero exit, ``final_candidate_failed_drc.npz`` only, no SUCCESS.
``binarization >= 0.99`` is never a pass criterion; ``np.unique(mask) == {0,1}``
is.
"""

from __future__ import annotations

import argparse
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


def _exact_binary_unique(model, latent) -> np.ndarray:
    """Beta-independent exact binary mask on the 240x240 unique grid."""
    filtered = np.asarray(model.mapping.filter_unique(latent), float)
    return (filtered >= 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", help="NPZ with the best feasible 'latent' array")
    ap.add_argument("--output", default="runs/final_projection")
    ap.add_argument("--mfs-um", type=float, default=0.5)
    ap.add_argument("--mgs-um", type=float, default=0.5)
    ap.add_argument("--min-gap-um", type=float, default=None)
    ap.add_argument("--gpu", default=os.environ.get("CL_GPU_DEVICE", "GPU 1"))
    ap.add_argument("--no-fdtd", action="store_true",
                    help="DRC only; do not run the exact-binary FDTD evaluation")
    ap.add_argument("--variants", action="store_true",
                    help="also evaluate eroded/dilated binary variants")
    args = ap.parse_args()
    os.environ["MFS_UM"] = str(args.mfs_um)
    os.environ["MGS_UM"] = str(args.mgs_um)
    os.environ.setdefault("MSOPT_MAPPING", "periodic_constrained")
    os.environ["CL_GPU_DEVICE"] = args.gpu
    os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu

    import eqc_lib as lib
    model = lib.load_model()
    if os.environ["MSOPT_MAPPING"] != "periodic_constrained":
        raise SystemExit("final_projection requires MSOPT_MAPPING=periodic_constrained")

    data = np.load(Path(args.design).resolve())
    if "latent" not in data.files:
        raise KeyError(f"{args.design} has no 'latent' array")
    latent = np.asarray(data["latent"], float).reshape(-1)
    if latent.size != model.Nux * model.Nuy:
        raise ValueError(
            f"latent size {latent.size} != Nux*Nuy {model.Nux*model.Nuy}"
        )
    output = (HERE / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 1) exact binary unique mask
    mask_u = _exact_binary_unique(model, latent)
    spacing_um = model.dx_um
    dx = model.dx_um
    if not np.isclose(model.dx_um, model.dy_um):
        raise RuntimeError("non-square design pixels are not supported by the DRC")

    # 2) independent DRC
    drc = geometry_drc(
        mask_u, spacing_um=spacing_um,
        min_solid_width_um=args.mfs_um, min_void_width_um=args.mgs_um,
        min_gap_um=args.min_gap_um,
    )
    (output / "geometry_drc.json").write_text(json.dumps(drc, indent=2) + "\n")
    print(json.dumps(drc, indent=2))

    # exact physical mask (only meaningful if DRC passes, but always recorded)
    mp = model.mapping
    mask_phys = np.asarray(
        mp.physical_from_unique(mask_u.astype(float)), float
    ).reshape(model.Nx, model.Ny, model.Nz).astype(np.uint8)
    diag = mapping_diagnostics(mask_phys.astype(float))
    if not diag.is_exact_binary:
        raise SystemExit("internal error: exact-binary mask is not binary")

    np.savez_compressed(output / "final_mask_unique.npz", mask=mask_u,
                        spacing_um=np.array(spacing_um))
    np.savez_compressed(output / "final_mask_physical.npz", mask=mask_phys,
                        physical=mask_phys.astype(float))

    if not drc["pass"]:
        np.savez_compressed(output / "final_candidate_failed_drc.npz",
                            mask=mask_u, latent=latent)
        (output / "final_manifest.json").write_text(json.dumps({
            "status": "geometry_infeasible",
            "drc_pass": False, "drc": drc,
        }, indent=2) + "\n")
        raise SystemExit(
            "DRC FAILED: minimum solid/void/gap below rule -- no SUCCESS written.")

    if args.no_fdtd:
        (output / "final_manifest.json").write_text(json.dumps({
            "status": "drc_pass_fdtd_skipped",
            "drc_pass": True, "drc": drc,
            "note": "exact-binary FDTD not run (--no-fdtd); SUCCESS requires it",
        }, indent=2) + "\n")
        print("DRC PASS; FDTD skipped (--no-fdtd). No SUCCESS until FOM evaluated.")
        return

    # 3) exact-binary FDTD evaluation (delta=0 forward on the {0,1} mask)
    from volume_current_evaluator import VolumeCurrentEvaluator
    fom = {}
    for pol in ("x", "y"):
        ev = VolumeCurrentEvaluator(output / f"solver_{pol}", 0.001, pol)
        ev.prepare(force_rebuild=False)
        fom[f"F{pol}"] = float(ev.forward_fom(mask_phys.astype(float),
                                              label="final_binary"))
    fom["F_sum"] = fom["Fx"] + fom["Fy"]
    fom_report = {"exact_binary": True, **fom}
    (output / "exact_binary_fom.json").write_text(json.dumps(fom_report, indent=2) + "\n")

    variant_fom = None
    if args.variants:
        variant_fom = {}
        beta_hi = 64.0
        for field in ("eroded", "dilated"):
            fu = np.asarray(mp.field_unique(latent, beta_hi, field), float)
            mv = (fu >= 0.5).astype(float)
            mvp = np.asarray(mp.physical_from_unique(mv), float).reshape(
                model.Nx, model.Ny, model.Nz)
            vals = {}
            for pol in ("x", "y"):
                ev = VolumeCurrentEvaluator(output / f"variant_{field}_{pol}", 0.001, pol)
                ev.prepare(force_rebuild=False)
                vals[f"F{pol}"] = float(ev.forward_fom(mvp, label=f"{field}_binary"))
            vals["F_sum"] = vals["Fx"] + vals["Fy"]
            variant_fom[field] = vals
        (output / "robust_variant_fom.json").write_text(
            json.dumps(variant_fom, indent=2) + "\n")

    manifest = {
        "status": "completed",
        "exact_binary": True,
        "periodic_fencepost": float(diag.periodic_x_max_abs_error) == 0.0
        and float(diag.periodic_y_max_abs_error) == 0.0,
        "z_invariant": float(diag.z_extrusion_max_abs_error) == 0.0,
        "drc_pass": True,
        "drc": drc,
        "exact_binary_fom": fom_report,
        "robust_variant_fom": variant_fom,
        "solid_fraction": float(mask_u.mean()),
    }
    (output / "final_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    success = {
        "exact_binary": True,
        "periodic_fencepost": manifest["periodic_fencepost"],
        "z_invariant": manifest["z_invariant"],
        "solid_width_500nm_drc": drc["minimum_solid_width_um"],
        "void_width_500nm_drc": drc["minimum_void_width_um"],
        "gap_500nm_drc": drc["minimum_gap_um"] if args.min_gap_um is not None else "not_applicable",
        "exact_binary_fom_evaluated": True,
        "Fx": fom["Fx"], "Fy": fom["Fy"], "F_sum": fom["F_sum"],
    }
    (output / "SUCCESS.json").write_text(json.dumps(success, indent=2) + "\n")
    print("SUCCESS: exact binary design passed DRC and exact-binary FOM evaluated.")
    print(json.dumps(fom_report, indent=2))


if __name__ == "__main__":
    main()
