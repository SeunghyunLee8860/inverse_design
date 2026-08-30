#!/usr/bin/env bash
set -euo pipefail

worktree=/home/seunghyun/tairte4/worktrees/pte_true_mma
python_bin=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
module=photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization
base_fsp=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp
base_sha=454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83
jacobian=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/production_input_component_yee_jacobian_v1

if [[ $# -ne 1 || ( "$1" != "Ea" && "$1" != "Eb" ) ]]; then
  echo "usage: $0 Ea|Eb" >&2
  exit 2
fi

polarization=$1
if [[ "$polarization" == "Ea" ]]; then
  gpu=4
  raw=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_ansys_dfm_Ea_20260811
  published=$worktree/photothermal_pte/optimization_runs/run_044_ansys_dfm_Ea_current_max
  xdg=/tmp/seunghyun_lumerical_run044
  mpl=/tmp/seunghyun_matplotlib_run044
else
  gpu=5
  raw=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_ansys_dfm_Eb_20260811
  published=$worktree/photothermal_pte/optimization_runs/run_045_ansys_dfm_Eb_current_max
  xdg=/tmp/seunghyun_lumerical_run045
  mpl=/tmp/seunghyun_matplotlib_run045
fi

stage_json=$published/latest_stage.json
history_json=$raw/history.json
test -f "$stage_json"
test -f "$history_json"

stage_stop_reason=$(jq -r '.stage_stop_reason' "$stage_json")
stage_beta=$(jq -r '.beta' "$stage_json")
history_beta=$(jq -r '.[-1].beta' "$history_json")
evaluation_id=$(jq -r '.[-1].evaluation_id' "$history_json")
if [[ "$stage_stop_reason" != "adaptive_plateau" ]]; then
  echo "refusing recovery: latest stage did not end at an audited plateau" >&2
  exit 3
fi
if ! "$python_bin" -c 'import math,sys; assert math.isclose(float(sys.argv[1]),float(sys.argv[2]),rel_tol=0.0,abs_tol=1e-10)' "$stage_beta" "$history_beta"; then
  echo "refusing recovery: latest history row is not the completed plateau stage" >&2
  exit 4
fi

printf -v evaluation_padded '%04d' "$evaluation_id"
initial=$(find "$raw" -maxdepth 1 -type f -name "evaluation_${evaluation_padded}_beta*_ansys_dfm_ld_mma_adaptive_recovery3_latent.npz" -print -quit)
if [[ -z "$initial" || ! -f "$initial" ]]; then
  echo "refusing recovery: completed latent checkpoint was not found" >&2
  exit 5
fi
next_beta=$("$python_bin" -c 'import sys; print(f"{min(128.0, 2.0*float(sys.argv[1])):.12g}")' "$stage_beta")

mkdir -p "$xdg" "$mpl"
export PYTHONPATH=$worktree
export PATH=/opt/lumerical/v261/bin:/home/eidl/miniconda3/envs/EIDL-Lumapi/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/eidl/miniconda3/envs/EIDL-Lumapi/lib:/opt/lumerical/v261/api/python:/opt/lumerical/v261/lib
export XDG_CONFIG_HOME=$xdg
export MPLCONFIGDIR=$mpl
export CUDA_VISIBLE_DEVICES=$gpu
export TAIRTE4_TOPOLOGY_GEOMETRY=contact_anchored
export LUMERICAL_LICENSE_RETRY_SECONDS=30
export LUMERICAL_GPU_ENGINE_LOCK=/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock

cd "$worktree"
exec "$python_bin" -m "$module" \
  --polarization "$polarization" \
  --raw-root "$raw" \
  --published-dir "$published" \
  --gpu "$gpu" \
  --base-fsp "$base_fsp" \
  --base-sha256 "$base_sha" \
  --jacobian-dir "$jacobian" \
  --constraint-device cuda \
  --initial-latent-npz "$initial" \
  --recovery-append \
  --start-beta "$next_beta" \
  --output-slug factor2_dfm_adaptive_recovery4

