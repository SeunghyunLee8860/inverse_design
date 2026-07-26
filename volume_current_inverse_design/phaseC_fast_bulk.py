#!/usr/bin/env python3
"""Phase C: fast bulk mesh (far-bulk z coarsening) vs accuracy-5 baseline.

fast_bulk (bundle/tairte4_volume_model.py) keeps the ENTIRE production
contract -- auto non-uniform, accuracy 5, CV1, 5 nm flake dz, 25 nm in-plane,
identical source/monitor/geometry coordinates -- and adds two z-only override
caps in the far bulk (top air, deep Si).  A rectilinear Yee grid pins the
in-plane mesh via the design override at every z, and dt is set by the 5 nm
flake cells, so only the z axis can change.

Per configuration (each in a fresh subprocess; BULK_MESH_MODE is read at
model import):
  1. mesh probe: full-height z-line DFT monitor + 0.2 ps solve -> realized
     z axis, per-band cell counts, dt;
  2. combined-mode objective evaluation (x and y) on the SAME latent as
     Phases A/B at the Phase-B-selected simulation time.

Gates (fail-closed):
  interface-band z axis identical to baseline   (flake/interface contract)
  F_sum rel difference          < 1%
  gradient cosine (phys+latent) > 0.995
  directional-deriv rel diff    < 2%

Usage:
  phaseC_fast_bulk.py <outdir> [--sim-time 4e-12] [--beta 4.0]
                      [--band -0.45,0.70]
  (internal) --single : run one configuration in this process
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "bundle"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TINY = np.finfo(float).tiny
PROBE_NAME = "mesh_probe_z"


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), TINY)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), TINY))


def probe_mesh(lib, model, control_base: Path, out: Path) -> dict:
    """Realized full-height z axis from a 0.2 ps probe solve."""
    z_lo = (model.Z_min + 0.4) * 1e-6
    z_hi = (model.Z_max - 0.4) * 1e-6
    with lib.open_control(control_base) as fdtd:
        fdtd.switchtolayout()
        if int(fdtd.getnamednumber(PROBE_NAME)):
            fdtd.select(PROBE_NAME)
            fdtd.delete()
        fdtd.adddftmonitor()
        fdtd.set("name", PROBE_NAME)
        fdtd.set("monitor type", "Linear Z")
        fdtd.set("x", 0.0)
        fdtd.set("y", 0.0)
        fdtd.set("z min", z_lo)
        fdtd.set("z max", z_hi)
        fdtd.set("override global monitor settings", True)
        fdtd.set("use source limits", False)
        fdtd.set("use wavelength spacing", True)
        fdtd.set("wavelength center", lib.WL)
        fdtd.set("wavelength span", 0.0)
        fdtd.set("frequency points", 1)
        try:
            fdtd.setnamed(PROBE_NAME, "spatial interpolation", "none")
        except Exception:
            pass
        fdtd.setnamed("FDTD", "simulation time", 2.0e-13)
        result = lib.run_project(
            fdtd, "mesh_probe", {"E": lambda f: f.getresult(PROBE_NAME, "E")})
        z = np.asarray(result["E"]["z"], float).reshape(-1)
        dt = float(np.asarray(fdtd.getnamed("FDTD", "dt")).reshape(-1)[0])
        fdtd.switchtolayout()
        fdtd.select(PROBE_NAME)
        fdtd.delete()
        fdtd.save(str(control_base))   # control back to production layout
    spacing = np.diff(z)
    np.savez_compressed(out / "mesh_probe_z.npz", z_m=z)
    return {
        "z_count": int(z.size),
        "z_lo_m": float(z[0]), "z_hi_m": float(z[-1]),
        "dt_s": dt,
        "min_dz_m": float(np.min(spacing)),
        "max_dz_m": float(np.max(spacing)),
        "band_counts": {
            "air_above_1um": int(np.sum(z > 1.0e-6)),
            "design_0_to_0.6um": int(np.sum((z >= 0) & (z <= 0.6e-6))),
            "flake_-0.1_to_0um": int(np.sum((z >= -0.1e-6) & (z < 0))),
            "sio2_-0.385_to_-0.1um": int(
                np.sum((z >= -0.385e-6) & (z < -0.1e-6))),
            "si_below_-0.385um": int(np.sum(z < -0.385e-6)),
        },
    }


def run_single(args) -> int:
    requested_mode = os.environ.get("BULK_MESH_MODE", "auto")

    import eqc_lib as lib

    model = lib.load_model()
    lib.assert_approved_lumapi()
    # Fail closed if anything downstream overrode the requested mesh mode --
    # the first Phase-C attempt silently re-ran baseline because
    # eqc_lib.bootstrap_env() hard-assigned BULK_MESH_MODE="auto".
    realized_mode = getattr(model, "BULK_MESH_MODE", "auto")
    if realized_mode != requested_mode:
        raise SystemExit(
            f"BULK_MESH_MODE requested={requested_mode!r} but the model "
            f"realized {realized_mode!r}; refusing to measure the wrong config")
    from autograd import tensor_jacobian_product
    from volume_current_evaluator import VolumeCurrentEvaluator

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping = model.mapping
    shape = (model.Nx, model.Ny, model.Nz)
    latent = np.asarray(model.x0, float).reshape(-1)
    phys = np.asarray(mapping(latent, args.beta), float).reshape(shape)

    os.environ["VC_ADJOINT_COMPONENT_MODE"] = "combined"
    record: dict = {
        "bulk_mesh_mode": os.environ.get("BULK_MESH_MODE", "auto"),
        "sim_time_s": lib.SIM_TIME_S,
        "beta": args.beta,
        "fb_env": {k: os.environ.get(k) for k in (
            "FB_AIR_DZ_NM", "FB_SI_DZ_NM", "FB_AIR_ZMIN_UM", "FB_SI_ZMAX_UM")},
    }
    evaluators = {}
    for pol in ("x", "y"):
        ev = VolumeCurrentEvaluator(out / f"solver_{pol}", args.rho_step, pol)
        ev.prepare(force_rebuild=False)
        evaluators[pol] = ev
    record["mesh"] = probe_mesh(lib, model, evaluators["x"].control_base, out)
    print(f"[phaseC {record['bulk_mesh_mode']}] mesh: "
          + json.dumps(record["mesh"]), flush=True)

    for pol in ("x", "y"):
        t0 = time.time()
        evaluation = evaluators[pol].value_and_gradient(
            phys, label=f"phaseC_{pol}", density_mode="probe_safe")
        g_phys = np.asarray(evaluation.gradient_physical, float)
        g_lat = np.asarray(tensor_jacobian_product(
            lambda z: mapping(z, args.beta))(latent, g_phys.reshape(-1)),
            float).reshape(-1)
        np.savez_compressed(out / f"gradient_{pol}.npz",
                            gradient_physical=g_phys, gradient_latent=g_lat,
                            fom=np.array(evaluation.fom))
        meta = evaluation.metadata
        record[pol] = {
            "fom": float(evaluation.fom),
            "eval_total_s": time.time() - t0,
            "wall_time_s": meta["wall_time_s"],
            "pairing": float(meta["periodic_source_pairing_relative_error"]),
            "roundtrips": {k: float(v["roundtrip_max_abs_error"])
                           for k, v in meta["source"].items()},
        }
        print(f"[phaseC {record['bulk_mesh_mode']}] {pol}: "
              f"fom={evaluation.fom:.9e} "
              f"times={json.dumps(meta['wall_time_s'])}", flush=True)
    (out / "condition_report.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n")
    return 0


def load_condition(directory: Path) -> dict:
    report = json.loads((directory / "condition_report.json").read_text())
    report["z_axis"] = np.asarray(
        np.load(directory / "mesh_probe_z.npz")["z_m"], float)
    for pol in ("x", "y"):
        z = np.load(directory / f"gradient_{pol}.npz")
        report[pol]["gradient_physical"] = np.asarray(z["gradient_physical"])
        report[pol]["gradient_latent"] = np.asarray(z["gradient_latent"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--sim-time", default="4e-12")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--rho-step", type=float, default=0.001)
    ap.add_argument("--band", default="-0.45,0.70",
                    help="interface band (um) that must stay identical")
    ap.add_argument("--direction-seed", type=int, default=20260726)
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()
    if args.single:
        return run_single(args)

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    configs = {
        "baseline": {"BULK_MESH_MODE": "auto"},
        "fast_bulk": {"BULK_MESH_MODE": "fast_bulk"},
    }
    loaded = {}
    for name, extra_env in configs.items():
        cond_dir = out / name
        env = dict(os.environ)
        env.update(extra_env)
        env["VC_SIM_TIME_S"] = args.sim_time
        env["VC_ADJOINT_COMPONENT_MODE"] = "combined"
        print(f"[phaseC] launching {name} -> {cond_dir}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(cond_dir),
             "--single", "--beta", str(args.beta),
             "--rho-step", str(args.rho_step)],
            env=env, cwd=str(HERE))
        if proc.returncode != 0:
            raise SystemExit(f"config {name} failed rc={proc.returncode}")
        loaded[name] = load_condition(cond_dir)

    base, fast = loaded["baseline"], loaded["fast_bulk"]
    lo_um, hi_um = (float(v) for v in args.band.split(","))
    def band(z):
        return z[(z >= lo_um * 1e-6) & (z <= hi_um * 1e-6)]
    zb, zf = band(base["z_axis"]), band(fast["z_axis"])
    band_identical = bool(
        zb.size == zf.size and np.allclose(zb, zf, rtol=0.0, atol=1e-12))

    rng = np.random.default_rng(args.direction_seed)
    dir_phys = rng.standard_normal(base["x"]["gradient_physical"].size)
    dir_phys /= np.linalg.norm(dir_phys)
    dir_lat = rng.standard_normal(base["x"]["gradient_latent"].size)
    dir_lat /= np.linalg.norm(dir_lat)

    def dd(vec, direction):
        return float(np.dot(vec.reshape(-1), direction))

    fsum_b = base["x"]["fom"] + base["y"]["fom"]
    fsum_f = fast["x"]["fom"] + fast["y"]["fom"]
    comparison = {
        "band_um": [lo_um, hi_um],
        "interface_band_identical": band_identical,
        "band_cells": [int(zb.size), int(zf.size)],
        "z_cells_total": [base["mesh"]["z_count"], fast["mesh"]["z_count"]],
        "dt_s": [base["mesh"]["dt_s"], fast["mesh"]["dt_s"]],
        "band_counts": {"baseline": base["mesh"]["band_counts"],
                        "fast_bulk": fast["mesh"]["band_counts"]},
        "F_sum": [fsum_b, fsum_f],
        "F_sum_rel_diff": rel_diff(fsum_b, fsum_f),
        "wall_time_s": {n: {p: loaded[n][p]["wall_time_s"] for p in "xy"}
                        for n in loaded},
        "eval_total_s": {n: {p: loaded[n][p]["eval_total_s"] for p in "xy"}
                         for n in loaded},
    }
    for pol in ("x", "y"):
        b, f = base[pol], fast[pol]
        comparison[f"fom_{pol}"] = [b["fom"], f["fom"]]
        comparison[f"fom_rel_diff_{pol}"] = rel_diff(b["fom"], f["fom"])
        comparison[f"grad_cosine_phys_{pol}"] = cosine(
            b["gradient_physical"], f["gradient_physical"])
        comparison[f"grad_cosine_latent_{pol}"] = cosine(
            b["gradient_latent"], f["gradient_latent"])
        comparison[f"dir_deriv_rel_diff_phys_{pol}"] = rel_diff(
            dd(b["gradient_physical"], dir_phys),
            dd(f["gradient_physical"], dir_phys))
        comparison[f"dir_deriv_rel_diff_latent_{pol}"] = rel_diff(
            dd(b["gradient_latent"], dir_lat),
            dd(f["gradient_latent"], dir_lat))

    checks = {
        "interface_band_identical": band_identical,
        "dt_identical": bool(np.isclose(
            base["mesh"]["dt_s"], fast["mesh"]["dt_s"], rtol=5e-10)),
        "F_sum_rel_lt_1pct": comparison["F_sum_rel_diff"] < 1e-2,
        "grad_cosine_gt_0.995": all(
            comparison[f"grad_cosine_phys_{p}"] > 0.995
            and comparison[f"grad_cosine_latent_{p}"] > 0.995 for p in "xy"),
        "dir_deriv_lt_2pct": all(
            comparison[f"dir_deriv_rel_diff_phys_{p}"] < 2e-2
            and comparison[f"dir_deriv_rel_diff_latent_{p}"] < 2e-2
            for p in "xy"),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed_checks",
        "sim_time_s": float(args.sim_time),
        "checks": checks,
        "comparison": comparison,
    }
    (out / "phaseC_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float), flush=True)
    print("PHASE C " + ("PASSED" if all(checks.values()) else
                        "FAILED: " + ", ".join(
                            k for k, v in checks.items() if not v)),
          flush=True)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
