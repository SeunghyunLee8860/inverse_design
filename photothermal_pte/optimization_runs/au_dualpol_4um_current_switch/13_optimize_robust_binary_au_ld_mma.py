#!/usr/bin/env python3
"""Robust-projection LD_MMA recovery for a binary dual-polarization Au switch.

The nominal gray design is not promoted: exact binarization reverses the Eb
current.  This restart optimizes dilated (eta=0.35), nominal (eta=0.50), and
eroded (eta=0.65) physical projections simultaneously.  The epigraph is
bounded by Ia and -Ib, and grayness is constrained, for all three
realizations.  A 500 nm conic filter drives a fabrication-scale topology; an
exact solid/void audit remains the only final authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from scipy import ndimage

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
    combined_gradient,
    evaluate_forward_multiphysics,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.dfm import (
    MAPPING,
    exact_500nm_audit,
    physical_disk_footprint,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.robust_contract import (
    POLARIZATIONS,
    ROBUST_ETAS,
    audit as robust_contract_audit,
    current_constraint_keys,
    eta_key,
    gray_constraint_keys,
    grayness,
    grayness_cotangent,
    scenario_key,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    require_production_readiness,
)
from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_dualpol_au_robust_projection_ld_mma"
RAW = Path(
    "/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/robust_projection_ld_mma"
)
INITIAL = Path(
    "/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/optimization_ld_mma/"
    "stage_04_beta_12.npz"
)
CALIBRATION = HERE / "results_fdtdx_4um_source_calibration/fdtdx_4um_source_calibration.json"
CURRENT_SCALE_A = 1.0e-9
ETAS = ROBUST_ETAS
STAGES = (
    (12.0, 0.40, 12),
    (16.0, 0.32, 12),
    (24.0, 0.24, 10),
    (32.0, 0.16, 10),
    (48.0, 0.10, 10),
    (64.0, 0.06, 10),
    (80.0, 0.035, 8),
    (96.0, 0.025, 10),
    (128.0, 0.015, 10),
    (192.0, 0.0075, 8),
    (256.0, 0.0035, 8),
)
NLOPT_TOL = 2.0e-5


ROBUST_MAPPINGS = {
    eta: ProductionDensityMapping(
        shape=CONTRACT.design_shape,
        spacing_m=CONTRACT.design_pitch_m,
        radius_m=CONTRACT.filter_radius_m,
        eta=eta,
    )
    for eta in ETAS
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def optical_closure(result, runner, scale):
    eta0 = float(runner.model["fdtdx"].constants.eta0)
    p_six = scale * eta0 * float(
        np.mean(
            np.asarray(
                result["optical_output"].detector_states["material_flux_td"][
                    "poynting_flux"
                ]
            )[:, 0]
        )
    )
    p_q = float(np.sum(result["source_power_W"]))
    return p_q, p_six, abs(p_q-p_six)/max(abs(p_six), np.finfo(float).tiny)


def physics_gates(result, closure):
    gates = {
        "optical_closure_lt_0p5pct": closure < 0.005,
        "thermal_energy_balance_lt_1pct": result["thermal_audit"]["energy_balance_relative"] < 0.01,
        "thermal_residual_lt_1e8": result["thermal_audit"]["relative_residual"] < 1e-8,
        "electrical_residual_lt_1e8": result["electrical_audit"]["relative_residual"] < 1e-8,
        "mapping_transpose_lt_1e12": result["weighted_contraction_relative_error"] < 1e-12,
        "finite_nonnegative_q": all(np.all(np.isfinite(q)) and float(np.min(q)) >= 0 for q in result["q_fields_W_m3"].values()),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"fail-closed robust physics gate: closure={closure:.9e}, gates={gates}"
        )
    return gates


@dataclass
class RobustPoint:
    latent: np.ndarray
    rho_nominal: np.ndarray
    densities: dict[float, np.ndarray]
    scenarios: dict[str, dict[str, object]]
    grayness: dict[float, float]
    gray_gradients: dict[float, np.ndarray]


class RobustEvaluator:
    def __init__(self, runners, source_scale, cuda_device, beta, gray_cap, history, manifest):
        self.runners = runners
        self.source_scale = float(source_scale)
        self.cuda_device = int(cuda_device)
        self.beta = float(beta)
        self.gray_cap = float(gray_cap)
        self.history = history
        self.manifest = manifest
        self.cached_latent = None
        self.cached_point = None

    def evaluate(self, latent):
        latent = np.asarray(latent, dtype=float).reshape(CONTRACT.design_shape)
        if self.cached_latent is not None and np.array_equal(latent, self.cached_latent):
            return self.cached_point
        scenarios = {}
        densities = {}
        for eta, mapping in ROBUST_MAPPINGS.items():
            rho = mapping.physical(latent, self.beta)
            densities[eta] = rho
            for pol in POLARIZATIONS:
                result = combined_gradient(self.runners[pol], rho, self.source_scale, self.cuda_device)
                p_q, p_six, closure = optical_closure(result, self.runners[pol], self.source_scale)
                gates = physics_gates(result, closure)
                key = scenario_key(eta, pol)
                scenarios[key] = {
                    "eta": eta,
                    "pol": pol,
                    "rho": rho,
                    "current_A": float(result["objective_A"]),
                    "gradient_latent_A": mapping.vjp(latent, np.asarray(result["gradient_total_A"]), self.beta),
                    "P_Q_W": p_q,
                    "P_six_W": p_six,
                    "closure_relative": closure,
                    "Tmax_K": float(np.max(result["temperature"])),
                    "gates": gates,
                }
                del result
                gc.collect()
        rho_nominal = densities[0.50]
        gray_values = {
            eta: grayness(densities[eta]) for eta in ROBUST_ETAS
        }
        gray_gradients = {
            eta: ROBUST_MAPPINGS[eta].vjp(
                latent, grayness_cotangent(densities[eta]), self.beta
            )
            for eta in ROBUST_ETAS
        }
        point = RobustPoint(
            latent.copy(),
            rho_nominal,
            densities,
            scenarios,
            gray_values,
            gray_gradients,
        )
        self.record(point)
        self.cached_latent = latent.copy(); self.cached_point = point
        return point

    def record(self, point):
        evaluation = len(self.history)+1
        utilities = []
        summary = {}
        for key, scenario in point.scenarios.items():
            useful = scenario["current_A"] if scenario["pol"] == "Ea" else -scenario["current_A"]
            utilities.append(useful)
            summary[key] = {name:value for name,value in scenario.items() if name not in ("rho","gradient_latent_A")}
            summary[key]["useful_current_A"] = useful
        exact = exact_500nm_audit(point.rho_nominal)
        row = {
            "evaluation": evaluation,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "beta": self.beta,
            "gray_cap": self.gray_cap,
            "grayness": {
                eta_key(eta): point.grayness[eta] for eta in ROBUST_ETAS
            },
            "gray_constraints": {
                eta_key(eta): point.grayness[eta] / self.gray_cap - 1.0
                for eta in ROBUST_ETAS
            },
            "maximum_grayness": max(point.grayness.values()),
            "robust_min_utility_A": float(min(utilities)),
            "exact_bad_cells": int(exact["solid_bad_cell_count"]+exact["void_bad_cell_count"]),
            "scenarios": summary,
        }
        self.history.append(row)
        raw = RAW/f"evaluation_{evaluation:04d}.npz"
        raw_payload = {
            "latent": point.latent,
            "rho_nominal": point.rho_nominal,
        }
        for eta in ROBUST_ETAS:
            slug = eta_key(eta).replace(".", "p")
            raw_payload[f"rho_{slug}"] = point.densities[eta]
            raw_payload[f"gray_gradient_{slug}"] = point.gray_gradients[eta]
        np.savez_compressed(raw, **raw_payload)
        self.manifest.setdefault("evaluations", {})[f"{evaluation:04d}"] = {"path":str(raw.resolve()),"bytes":raw.stat().st_size,"sha256":sha256(raw)}
        write_json(OUT/"optimization_history.json",self.history)
        write_json(OUT/"RAW_ARTIFACT_MANIFEST.json",self.manifest)
        plot(self.history, point)
        print(f"[robust eval {evaluation:04d}] beta={self.beta:g} min={1e9*min(utilities):.5f} nA max_gray={max(point.grayness.values()):.4f}/{self.gray_cap:.4f} bad={row['exact_bad_cells']}",flush=True)

    def objective(self, vector, gradient):
        if gradient.size:
            gradient[:] = 0; gradient[-1] = 1
        return float(vector[-1])

    def constraints(self, values, vector, gradient):
        point = self.evaluate(vector[:-1]); t=float(vector[-1])
        keys = current_constraint_keys()
        for index,key in enumerate(keys):
            scenario=point.scenarios[key]
            values[index] = t - scenario["current_A"]/CURRENT_SCALE_A if scenario["pol"]=="Ea" else t + scenario["current_A"]/CURRENT_SCALE_A
        gray_offset = len(keys)
        for gray_index, eta in enumerate(ROBUST_ETAS):
            values[gray_offset + gray_index] = (
                point.grayness[eta] / self.gray_cap - 1.0
            )
        if gradient.size:
            gradient[:] = 0
            for index,key in enumerate(keys):
                scenario=point.scenarios[key]
                sign=-1 if scenario["pol"]=="Ea" else 1
                gradient[index,:-1]=sign*scenario["gradient_latent_A"].ravel()/CURRENT_SCALE_A
                gradient[index,-1]=1
            for gray_index, eta in enumerate(ROBUST_ETAS):
                gradient[gray_offset + gray_index, :-1] = (
                    point.gray_gradients[eta].ravel() / self.gray_cap
                )


def plot(history, point):
    fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    axes[0,0].imshow(point.rho_nominal.T,origin="lower",cmap="gray_r",vmin=0,vmax=1,extent=(-4,4,-4,4)); axes[0,0].set_title("nominal physical density")
    axes[0,1].hist(point.rho_nominal.ravel(),bins=40,range=(0,1)); axes[0,1].set_title("density histogram")
    x=[row["evaluation"] for row in history]
    axes[1,0].plot(x,[1e9*row["robust_min_utility_A"] for row in history],"ko-"); axes[1,0].axhline(0,color="black",lw=.8); axes[1,0].set_title("worst-case min(Ia,-Ib)"); axes[1,0].set_ylabel("nA")
    for eta in ROBUST_ETAS:
        key = eta_key(eta)
        axes[1,1].plot(x,[row["grayness"][key] for row in history],label=key)
    axes[1,1].plot(x,[row["gray_cap"] for row in history],"k--",label="cap"); axes[1,1].legend(); axes[1,1].set_title("all-scenario binary continuation")
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.suptitle(f"Robust Au recovery; eval={history[-1]['evaluation']}, beta={history[-1]['beta']:g}")
    fig.savefig(OUT/f"evaluation_{history[-1]['evaluation']:04d}.png",dpi=160); fig.savefig(OUT/"latest_iteration.png",dpi=160); plt.close(fig)


def exact_candidates(rho):
    footprint=physical_disk_footprint(.5*CONTRACT.minimum_solid_feature_m,CONTRACT.design_pitch_m)
    rows=[]
    for threshold in np.linspace(.1,.9,17):
        seed=rho>=threshold
        # Targeted exact repair changes only the cells identified by the
        # independent opening audit.  Unlike a global close/open it does not
        # erase already valid, current-carrying topology far from a violation.
        for void_mode in ("fill", "expand"):
            binary=seed.copy()
            for _ in range(24):
                audit=exact_500nm_audit(binary.astype(float))
                if audit["solid_pass"] and audit["void_pass"]: break
                before=binary.copy()
                binary[audit["bad_solid"]]=False
                audit=exact_500nm_audit(binary.astype(float))
                if void_mode=="fill":
                    binary[audit["bad_void"]]=True
                else:
                    binary &= ~ndimage.binary_dilation(
                        audit["bad_void"],structure=footprint
                    )
                if np.array_equal(binary,before): break
            audit=exact_500nm_audit(binary.astype(float))
            if audit["solid_pass"] and audit["void_pass"]:
                score=float(np.mean(np.abs(binary-rho)))
                rows.append(
                    (score,f"threshold_{threshold:.2f}_targeted_remove_{void_mode}",binary.astype(float))
                )
        for name,seq in (("oco",("open","close","open")),("co",("close","open")),("coc",("close","open","close"))):
            binary=seed.copy()
            for op in seq:
                binary=ndimage.binary_opening(binary,structure=footprint,border_value=0) if op=="open" else ~ndimage.binary_opening(~binary,structure=footprint,border_value=1)
            audit=exact_500nm_audit(binary.astype(float))
            if audit["solid_pass"] and audit["void_pass"]:
                score=float(np.mean(np.abs(binary-rho)))
                rows.append((score,f"threshold_{threshold:.2f}_{name}",binary.astype(float)))
    unique=[]
    for item in sorted(rows,key=lambda row:(row[0],row[1])):
        if not any(np.array_equal(item[2],prior[2]) for prior in unique): unique.append(item)
    return unique[:6]


def forward_binary(runners,rho,scale,cuda_device):
    cases={}
    for pol in ("Ea","Eb"):
        result=evaluate_forward_multiphysics(runners[pol],rho,scale,cuda_device,need_gradient=False)
        p_q,p_six,closure=optical_closure(result,runners[pol],scale)
        gates={
            "optical_closure_lt_0p5pct":closure<.005,
            "thermal_energy_balance_lt_1pct":result["thermal_audit"]["energy_balance_relative"]<.01,
            "thermal_residual_lt_1e8":result["thermal_audit"]["relative_residual"]<1e-8,
            "electrical_residual_lt_1e8":result["electrical_audit"]["relative_residual"]<1e-8,
        }
        if not all(gates.values()): raise RuntimeError(f"binary gate {pol}: {gates}")
        cases[pol]={"current_A":float(result["objective_A"]),"P_Q_W":p_q,"P_six_W":p_six,"closure_relative":closure,"Tmax_K":float(np.max(result["temperature"])),"gates":gates}
        del result; gc.collect()
    ia=cases["Ea"]["current_A"]; ib=cases["Eb"]["current_A"]
    return {"I_a_A":ia,"I_b_A":ib,"balanced_utility_A":min(ia,-ib),"cases":cases}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None: raise RuntimeError("GPU required")
    readiness = require_production_readiness()
    cuda_device=int(os.environ.get("THERMAL_CUDA_DEVICE","0")); OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
    calibration=json.loads(CALIBRATION.read_text()); scale=CONTRACT.reporting_incident_power_W/float(calibration["common_reference_incident_power_W"])
    finalize_only=os.environ.get("AU_ROBUST_FINALIZE_ONLY","0")=="1"
    resume_high=os.environ.get("AU_ROBUST_RESUME_HIGH_BETA","0")=="1"
    resume_final=RAW/"stage_06_beta_80.npz"
    starting_path=resume_final if (finalize_only or resume_high) else INITIAL
    with np.load(starting_path,allow_pickle=False) as data: latent=np.asarray(data["latent"],dtype=float)
    # The eroded/dilated material layouts can ring longer than the nominal
    # beta-continuation layout.  Use a 50% longer observable window while
    # keeping geometry, source, mesh and all material parameters unchanged.
    runners={
        pol:CompiledOpticalRunner.create(
            pol,np.full(CONTRACT.design_shape,.5),total_periods=24,window_periods=6
        )
        for pol in ("Ea","Eb")
    }
    if finalize_only or resume_high:
        history=json.loads((OUT/"optimization_history.json").read_text())
        stages=json.loads((OUT/"continuation_stages.json").read_text())
        manifest=json.loads((OUT/"RAW_ARTIFACT_MANIFEST.json").read_text())
        if manifest.get("au_material_fraction") != material_fraction_audit():
            raise RuntimeError("robust resume uses the historical O3/TE1 law; start a new shared-law run")
        if manifest.get("robust_contract") != robust_contract_audit():
            raise RuntimeError("robust resume omits nominal-current or all-scenario grayness constraints")
        if manifest.get("production_readiness") != readiness:
            raise RuntimeError("robust resume is not linked to the current mesh/gradient certificates")
    else:
        history=[]; stages=[]; manifest={"schema":"au-dualpol-robust-projection-v3","raw_artifacts_committed_to_git":False,"etas":list(ETAS),"filter":MAPPING.audit(),"au_material_fraction":material_fraction_audit(),"robust_contract":robust_contract_audit(),"production_readiness":readiness,"evaluations":{}}
    vector=np.concatenate((latent.ravel(),[0.0]))
    run_stages=() if finalize_only else (STAGES[7:] if resume_high else STAGES)
    stage_offset=7 if resume_high else 0
    for local_stage_index,(beta,gray_target,maxeval) in enumerate(run_stages):
        stage_index=stage_offset+local_stage_index
        entry_gray=max(
            grayness(mapping.physical(latent,beta))
            for mapping in ROBUST_MAPPINGS.values()
        ); gray_cap=max(gray_target,.88*entry_gray)
        evaluator=RobustEvaluator(runners,scale,cuda_device,beta,gray_cap,history,manifest)
        entry=evaluator.evaluate(latent); useful=[s["current_A"] if s["pol"]=="Ea" else -s["current_A"] for s in entry.scenarios.values()]
        vector[:-1]=latent.ravel(); vector[-1]=min(useful)/CURRENT_SCALE_A-1e-5
        opt=nlopt.opt(nlopt.LD_MMA,vector.size); opt.set_lower_bounds(np.r_[np.zeros(vector.size-1),-100.]); opt.set_upper_bounds(np.r_[np.ones(vector.size-1),1000.]); opt.set_max_objective(evaluator.objective); opt.add_inequality_mconstraint(evaluator.constraints,np.full(len(current_constraint_keys())+len(gray_constraint_keys()),NLOPT_TOL)); opt.set_initial_step(np.r_[np.full(vector.size-1,.04),.1]); opt.set_ftol_rel(0); opt.set_xtol_rel(0); opt.set_maxeval(maxeval)
        start=time.perf_counter(); vector=opt.optimize(vector); latent=vector[:-1].reshape(CONTRACT.design_shape); returned=evaluator.evaluate(latent)
        returned_min = min(s["current_A"] if s["pol"]=="Ea" else -s["current_A"] for s in returned.scenarios.values())
        constraint_values = np.empty(
            len(current_constraint_keys()) + len(gray_constraint_keys()),
            dtype=np.float64,
        )
        evaluator.constraints(
            constraint_values,
            vector,
            np.empty((0, 0), dtype=np.float64),
        )
        stage_feasible = bool(
            np.all(np.isfinite(constraint_values))
            and returned_min > 0.0
            and float(vector[-1]) > 0.0
            and float(np.max(constraint_values)) <= 10.0 * NLOPT_TOL
        )
        stage={"stage":stage_index,"beta":beta,"gray_cap":gray_cap,"maxeval":maxeval,"nlopt_result":int(opt.last_optimize_result()),"numevals":int(opt.get_numevals()),"runtime_s":time.perf_counter()-start,"returned_robust_min_A":returned_min,"returned_epigraph_scaled":float(vector[-1]),"constraint_values":constraint_values.tolist(),"stage_feasible":stage_feasible}
        stages.append(stage); np.savez_compressed(RAW/f"stage_{stage_index:02d}_beta_{beta:g}.npz",latent=latent,vector=vector,beta=beta,gray_cap=gray_cap); write_json(OUT/"continuation_stages.json",stages); print(f"[stage complete] {stage}",flush=True)
        if not stage_feasible:
            raise RuntimeError(
                f"robust beta={beta:g} returned a non-promotable stage: {stage}"
            )
    final_beta=STAGES[-1][0] if not finalize_only else float(stages[-1]["beta"])
    nominal=MAPPING.physical(latent,final_beta); candidates=exact_candidates(nominal); binary_rows=[]
    if not candidates:
        raise RuntimeError("no exact 500 nm solid/void binary candidate")
    for _,name,rho in candidates:
        row=forward_binary(runners,rho,scale,cuda_device); row["name"]=name; row["exact_bad_cells"]=0; raw=RAW/f"final_{name}.npz"; np.savez_compressed(raw,physical_density=rho); row["raw"]={"path":str(raw.resolve()),"bytes":raw.stat().st_size,"sha256":sha256(raw)}; binary_rows.append(row); print(f"[binary] {name} Ia={1e9*row['I_a_A']:.5f} Ib={1e9*row['I_b_A']:.5f} min={1e9*row['balanced_utility_A']:.5f}",flush=True)
    best=max(binary_rows,key=lambda row:row["balanced_utility_A"]); success=best["I_a_A"]>0 and best["I_b_A"]<0
    final={"status":"VALIDATED_4UM_DUALPOL_AU_CURRENT_SWITCH_EXACT_BINARY" if success else "BLOCKED_ROBUST_PROJECTION_EXACT_BINARY_SIGN","timestamp_utc":datetime.now(timezone.utc).isoformat(),"etas":list(ETAS),"stages":stages,"binary_candidates":binary_rows,"promoted":best if success else None,"best_diagnostic":best,"opposite_sign_gate":success,"exact_500nm_gate":True,"history_evaluations":len(history)}
    write_json(OUT/"FINAL_RESULT.json",final); write_json(OUT/"RAW_ARTIFACT_MANIFEST.json",manifest); return 0 if success else 2


if __name__=="__main__": raise SystemExit(main())
