#!/usr/bin/env python3
"""Run one Run-002 production thermal/PTE scenario on CUDA."""
from __future__ import annotations
import argparse, hashlib, json, sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path: sys.path.insert(0, str(REPOSITORY))
FVM = REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1"
if str(FVM) not in sys.path: sys.path.insert(0, str(FVM))
from anisotropic_heat_fvm import assemble_steady_diagonal_kappa  # noqa: E402
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import solve_forward_adjoint_cuda  # noqa: E402
from photothermal_pte.thermal_adjoint.local_pte_functional import build_local_pte_functional  # noqa: E402

T_BATH = 300.0
SCENARIOS = {
    "grown_grown": (7.37e6, 7.37e6),
    "grown_evaporated": (7.37e6, 7.37e4),
    "evaporated_grown": (7.37e4, 7.37e6),
    "evaporated_evaporated": (7.37e4, 7.37e4),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def face_before(edges: np.ndarray, value: float) -> int:
    found = np.flatnonzero(np.isclose(edges, value, rtol=0, atol=2e-18))
    if found.size != 1: raise RuntimeError(f"missing face {value}")
    return int(found[0] - 1)

def material_specific_exposed(system, material_id, widths):
    active, ids = system.active_mask, system.active_ids
    additions = np.zeros(system.matrix_W_K.shape[0], float)
    face_ids, conductances = [], []
    def coefficient(material): return np.where(material == 3, 1.0, 10.0)
    for axis in range(3):
        if axis == 0:
            area = widths[1][None,:,None] * widths[2][None,None,:]
            lower, upper = active[:-1], active[1:]
            for cell, mat, exposed in ((ids[:-1], material_id[:-1], lower & ~upper), (ids[1:], material_id[1:], upper & ~lower)):
                g = np.broadcast_to(area, lower.shape)[exposed] * coefficient(mat[exposed])
                face_ids.append(cell[exposed]); conductances.append(g)
        elif axis == 1:
            area = widths[0][:,None,None] * widths[2][None,None,:]
            lower, upper = active[:,:-1], active[:,1:]
            for cell, mat, exposed in ((ids[:,:-1], material_id[:,:-1], lower & ~upper), (ids[:,1:], material_id[:,1:], upper & ~lower)):
                g = np.broadcast_to(area, lower.shape)[exposed] * coefficient(mat[exposed]); face_ids.append(cell[exposed]); conductances.append(g)
        else:
            area = widths[0][:,None,None] * widths[1][None,:,None]
            lower, upper = active[:,:,:-1], active[:,:,1:]
            for cell, mat, exposed in ((ids[:,:,:-1], material_id[:,:,:-1], lower & ~upper), (ids[:,:,1:], material_id[:,:,1:], upper & ~lower)):
                g = np.broadcast_to(area, lower.shape)[exposed] * coefficient(mat[exposed]); face_ids.append(cell[exposed]); conductances.append(g)
    # External z_max is exposed; x/y exterior and z_min are Dirichlet.
    top = active[:,:,-1]
    top_area = widths[0][:,None] * widths[1][None,:]
    face_ids.append(ids[:,:,-1][top]); conductances.append(top_area[top] * coefficient(material_id[:,:,-1][top]))
    flat_ids, flat_g = np.concatenate(face_ids), np.concatenate(conductances)
    np.add.at(additions, flat_ids, flat_g)
    terms = dict(system.boundary_terms)
    terms["material_specific_exposed"] = (flat_ids, flat_g, T_BATH)
    return replace(system, matrix_W_K=(system.matrix_W_K + sparse.diags(additions)).tocsr(), diagonal_W_K=system.diagonal_W_K + additions, boundary_load_W=system.boundary_load_W + additions*T_BATH, boundary_terms=terms)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapped-q", required=True, type=Path)
    parser.add_argument("--mapped-q-sha256", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="grown_grown")
    parser.add_argument("--cuda-device", type=int, default=3)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    q_path = args.mapped_q.resolve()
    if sha256(q_path) != args.mapped_q_sha256: raise RuntimeError("mapped-Q SHA mismatch")
    data = np.load(q_path)
    edges = tuple(np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
    widths = tuple(np.diff(edge) for edge in edges); shape = tuple(width.size for width in widths)
    masks = {name: np.asarray(data[f"mask_{name}"], bool) for name in ("Si","bottom_SiO2","physical_TaIrTe4","design_effective_SiO2")}
    material = np.zeros(shape, np.uint8); material[masks["Si"]]=1; material[masks["bottom_SiO2"]]=2; material[masks["physical_TaIrTe4"]]=3; material[masks["design_effective_SiO2"]]=4
    active = material > 0; rho = 0.5
    kappa = np.full((*shape,3), 0.026, float); kappa[masks["Si"]]=145.0; kappa[masks["bottom_SiO2"]]=1.38; kappa[masks["physical_TaIrTe4"]]=[14.4,3.8,1.0]; kappa[masks["design_effective_SiO2"]]=0.026+rho*(1.38-0.026)
    rx=np.zeros((shape[0]-1,shape[1],shape[2])); ry=np.zeros((shape[0],shape[1]-1,shape[2])); rz=np.zeros((shape[0],shape[1],shape[2]-1))
    si_face=face_before(edges[2],-0.385e-6); rz[:,:,si_face]=1/1.1e9
    bottom_face=face_before(edges[2],-0.100e-6); connected=masks["bottom_SiO2"][:,:,bottom_face] & masks["physical_TaIrTe4"][:,:,bottom_face+1]; rz[:,:,bottom_face][connected]=1/SCENARIOS[args.scenario][0]
    top_face=face_before(edges[2],0.0); connected=masks["physical_TaIrTe4"][:,:,top_face] & masks["design_effective_SiO2"][:,:,top_face+1]; gtop=1+rho*(SCENARIOS[args.scenario][1]-1); rz[:,:,top_face][connected]=1/gtop
    started=perf_counter()
    system=assemble_steady_diagonal_kappa(x_edges_m=edges[0],y_edges_m=edges[1],z_edges_m=edges[2],kappa_W_mK=kappa,active_mask=active,interface_resistance_m2K_W={"x":rx,"y":ry,"z":rz},dirichlet_temperature_K={"x_min":T_BATH,"x_max":T_BATH,"y_min":T_BATH,"y_max":T_BATH,"z_min":T_BATH})
    system=material_specific_exposed(system,material,widths); assembly_s=perf_counter()-started
    pte=build_local_pte_functional(x_edges_m=edges[0],y_edges_m=edges[1],z_edges_m=edges[2],active_mask=active,active_ids=system.active_ids,fom_mask=masks["physical_TaIrTe4"])
    wx=wy=1/(64e-6)
    c=-(wx*pte.sigma_a_S_m*pte.seebeck_a_V_K*(pte.derivative_x_m_inv.T@pte.integration_volume_m3)+wy*pte.sigma_b_S_m*pte.seebeck_b_V_K*(pte.derivative_y_m_inv.T@pte.integration_volume_m3))
    source=np.asarray(data["Q_total_W_m3"],float); source_active=system.active_source(source); source_power=np.asarray(system.source_volume_operator_m3@source_active)
    pair=solve_forward_adjoint_cuda(system.matrix_W_K,source_power,np.asarray(c).reshape(-1),cuda_device=args.cuda_device,relative_tolerance=1e-10,max_iterations=30000)
    theta=pair.forward.solution; objective=float(np.dot(c,theta)); reciprocal=float(np.dot(pair.adjoint.solution,source_power))
    reciprocity_cauchy=abs(objective-reciprocal)/max(float(np.linalg.norm(c)*np.linalg.norm(theta)),float(np.linalg.norm(pair.adjoint.solution)*np.linalg.norm(source_power)),np.finfo(float).tiny)
    boundary={name:float(np.sum(g*theta[cell_ids])) for name,(cell_ids,g,_) in system.boundary_terms.items()}
    energy=abs(sum(boundary.values())-float(np.sum(source_power)))/max(abs(float(np.sum(source_power))),1e-300)
    theta_full=system.full_field(theta); out_npz=output/f"thermal_pte_{args.scenario}.npz"; np.savez_compressed(out_npz,theta_K=theta_full,x_edges_m=edges[0],y_edges_m=edges[1],z_edges_m=edges[2])
    result={"status":"VALIDATED_PRODUCTION_CUDA_THERMAL_PTE" if max(pair.forward.explicit_relative_residual,pair.adjoint.explicit_relative_residual)<1e-8 and energy<.01 and reciprocity_cauchy<1e-8 else "FAILED_PRODUCTION_CUDA_THERMAL_PTE","scenario":args.scenario,"shape_xyz":list(shape),"active_unknowns":int(system.matrix_W_K.shape[0]),"matrix_nnz":int(system.matrix_W_K.nnz),"assembly_seconds":assembly_s,"source_power_W":float(np.sum(source_power)),"Tmax_rise_K":float(np.nanmax(theta_full)),"TaIrTe4_volume_average_rise_K":float(np.mean(theta_full[masks["physical_TaIrTe4"]])),"PTE_current_A":objective,"PTE_reciprocal_A":reciprocal,"weighting_field_m_inv":[wx,wy],"forward":{"iterations":pair.forward.iterations,"residual":pair.forward.explicit_relative_residual,"seconds":pair.forward.solve_seconds},"adjoint":{"iterations":pair.adjoint.iterations,"residual":pair.adjoint.explicit_relative_residual,"seconds":pair.adjoint.solve_seconds},"reciprocity_raw_near_null_relative_error":pair.reciprocity_relative_error,"reciprocity_Cauchy_normalized_error":reciprocity_cauchy,"energy_balance_error":energy,"boundary_power_W":boundary,"artifact":{"path":str(out_npz),"size_bytes":out_npz.stat().st_size,"sha256":sha256(out_npz)},"CPU_linear_solve_fallback":False,"empirical_normalization_or_gradient_rescaling":False,"optimization_iterations":0}
    result["passed"]=result["status"].startswith("VALIDATED"); (output/"production_cuda_thermal_pte_result.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if result["passed"] else 1
if __name__=="__main__": raise SystemExit(main())
