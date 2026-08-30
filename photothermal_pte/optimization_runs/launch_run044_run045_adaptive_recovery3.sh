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
  initial=$raw/evaluation_0053_beta1.2_ansys_dfm_ld_mma_recovery2_latent.npz
  xdg=/tmp/seunghyun_lumerical_run044
  mpl=/tmp/seunghyun_matplotlib_run044
else
  gpu=5
  raw=/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_ansys_dfm_Eb_20260811
  published=$worktree/photothermal_pte/optimization_runs/run_045_ansys_dfm_Eb_current_max
  initial=$raw/evaluation_0045_beta1.2_ansys_dfm_ld_mma_recovery2_latent.npz
  xdg=/tmp/seunghyun_lumerical_run045
  mpl=/tmp/seunghyun_matplotlib_run045
fi

test -f "$initial"
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
  --start-beta 1.44 \
  --output-slug ansys_dfm_ld_mma_adaptive_recovery3
