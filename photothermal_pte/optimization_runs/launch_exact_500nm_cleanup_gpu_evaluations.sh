#!/usr/bin/env bash
set -euo pipefail

worktree=/home/seunghyun/tairte4/worktrees/pte_true_mma
python_bin=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
module=photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective
base_fsp=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp
base_sha=454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83
gpu=${EXACT_CLEANUP_GPU:-5}

export PYTHONPATH=$worktree
export PATH=/opt/lumerical/v261/bin:/home/eidl/miniconda3/envs/EIDL-Lumapi/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/eidl/miniconda3/envs/EIDL-Lumapi/lib:/opt/lumerical/v261/api/python:/opt/lumerical/v261/lib
export MPLCONFIGDIR=/tmp/seunghyun_matplotlib_exact_cleanup
export CUDA_VISIBLE_DEVICES=$gpu
export TAIRTE4_TOPOLOGY_GEOMETRY=contact_anchored
export LUMERICAL_GPU_ENGINE_LOCK=/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock
mkdir -p "$MPLCONFIGDIR"
cd "$worktree"

run_candidate() {
  local polarization=$1
  local order=$2
  local root=$3
  local reference=$4
  local density="${root}/${order}_exact_binary_candidate.npz"
  local output_base="${root}/${order}_${polarization}_gpu_objective"
  test -f "$density"
  local candidate_output
  for candidate_output in "$output_base" "${output_base}_retry1" "${output_base}_retry2" "${output_base}_retry3"; do
    if [[ -f "${candidate_output}/binary_objective_result.json" ]] && \
       jq -e 'has("objective_A")' "${candidate_output}/binary_objective_result.json" >/dev/null; then
      echo "[skip completed] ${polarization} ${order}: ${candidate_output}"
      return 0
    fi
  done

  local attempt output xdg status
  for attempt in 1 2 3; do
    output="${output_base}_retry${attempt}"
    if [[ -e "$output" ]]; then
      continue
    fi
    xdg="/tmp/seunghyun_lumerical_exact_cleanup_${polarization}_${order}_${attempt}_$$"
    mkdir -p "$xdg"
    echo "[attempt ${attempt}/3] ${polarization} ${order}; XDG_CONFIG_HOME=${xdg}"
    set +e
    XDG_CONFIG_HOME="$xdg" "$python_bin" -m "$module" \
      --base-fsp "$base_fsp" \
      --base-sha256 "$base_sha" \
      --rho-npz "$density" \
      --output-dir "$output" \
      --polarization "$polarization" \
      --gpu-device "GPU $gpu" \
      --cuda-device 0 \
      --reference-objective-A "$reference"
    status=$?
    set -e
    if [[ -f "${output}/binary_objective_result.json" ]] && \
       jq -e 'has("objective_A")' "${output}/binary_objective_result.json" >/dev/null; then
      echo "[completed] ${polarization} ${order}; evaluator_status=${status}; output=${output}"
      return 0
    fi
    echo "[retryable non-objective failure] ${polarization} ${order}; output=${output}" >&2
    if [[ -f "${output}/binary_objective_result.json" ]]; then
      jq '{status,error}' "${output}/binary_objective_result.json" >&2
    fi
    sleep 15
  done
  echo "all GPU attempts failed before producing an objective: ${polarization} ${order}" >&2
  return 5
}

ea_root=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_exact_500nm_cleanup_20260811
eb_root=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_exact_500nm_cleanup_20260811
ea_reference=$(jq -r '.[-1].objective_A' /data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_ansys_dfm_Ea_20260811/history.json)
eb_reference=$(jq -r '.[-1].objective_A' /data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_ansys_dfm_Eb_20260811/history.json)

run_candidate Ea solid_first "$ea_root" "$ea_reference"
run_candidate Ea void_first "$ea_root" "$ea_reference"
run_candidate Eb solid_first "$eb_root" "$eb_reference"
run_candidate Eb void_first "$eb_root" "$eb_reference"
echo "all exact-cleanup GPU objective evaluations completed"
