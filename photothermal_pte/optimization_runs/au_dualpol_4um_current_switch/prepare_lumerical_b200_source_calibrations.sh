#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
output_root="${1:?usage: $0 NEW_CALIBRATION_ROOT}"
gpu_index="${LUMERICAL_B200_GPU_INDEX:?set the physical B200 GPU index}"
python_bin="${AU_LUMERICAL_PYTHON:?set the absolute Lumerical Python path}"
lumerical_root="${AU_LUMERICAL_ROOT:?set the absolute Lumerical v261 root}"
runres_bin="${AU_RUNRES_BIN:?set the absolute runres path}"
reserve_module="${LUM_RESERVE_MODULE_DIR:?set the directory containing lum_reserve.py}"
threads="${FDTD_THREADS:-8}"

if [[ -e "$output_root" ]]; then
  echo "refusing existing calibration root: $output_root" >&2
  exit 2
fi
for required in \
  "$python_bin" \
  "$lumerical_root/api/python/lumapi.py" \
  "$runres_bin" \
  "$reserve_module/lum_reserve.py"; do
  if [[ ! -e "$required" ]]; then
    echo "missing B200 prerequisite: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_root"
export PATH="$(dirname -- "$python_bin"):${PATH}"
export PYTHONPATH="$lumerical_root/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"
export VC_LUMERICAL_ROOT="$lumerical_root"
export LUMERICAL_ROOT="$lumerical_root"
export AU_LUMERICAL_ROOT="$lumerical_root"
export AU_LUMERICAL_PYTHON="$python_bin"
export AU_LUMERICAL_ACCELERATOR_POLICY="b200"
export LUMERICAL_GPU_INDEX="$gpu_index"
export LUMERICAL_B200_GPU_INDEX="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export LUMERICAL_SESSION_GPU_DEVICE="GPU $gpu_index"
export LUM_RESERVE_MODULE_DIR="$reserve_module"

"$python_bin" "$script_dir/21_audit_lumerical_maxwell_preflight.py" \
  --gpu-index "$gpu_index" --accelerator-policy b200 --require-ready

run_source_only() {
  local polarization="$1"
  local flake_dxy_nm="$2"
  local mesh_label="$3"
  local destination="$output_root/xy${flake_dxy_nm}/${polarization}"
  mkdir -p "$destination"
  "$runres_bin" \
    --reserve-count 9 \
    --reserve-wait "${AU_LUMERICAL_RESERVE_WAIT_S:-1800}" \
    --reserve-tag "au4um_b200_source_xy${flake_dxy_nm}_${polarization}" \
    "$script_dir/25_run_lumerical_4um_exact_au_control.py" \
    --case source_only \
    --polarization "$polarization" \
    --gpu-index "$gpu_index" \
    --accelerator-policy b200 \
    --output-dir "$destination" \
    --source-object-w0-um 3.956143303046142 \
    --mesh-label "$mesh_label" \
    --flake-dxy-nm "$flake_dxy_nm" \
    --outer-dxy-nm 200 \
    --stack-dz-nm 2.5 \
    --bulk-dz-nm 50 \
    --mesh-refinement "conformal variant 0" \
    --pml-layers 8 \
    --lateral-span-um 20 \
    --z-min-um -3 \
    --z-max-um 3 \
    --simulation-time-ps 1 \
    --threads "$threads" \
    -th "$threads" -GPU "$gpu_index"
  local result="$destination/source_only_${polarization}_${mesh_label}.json"
  if [[ ! -f "$result" ]]; then
    echo "missing source-calibration result: $result" >&2
    exit 3
  fi
  "$python_bin" -c \
    'import json,sys; p=sys.argv[1]; r=json.load(open(p)); assert r.get("all_gates_passed") is True and str(r.get("status", "")).startswith("PASSED")' \
    "$result"
}

coarse_label="fine_z2p5_bulk50_xy100_cv0_pml8_span20_z6_t1ps"
fine_label="fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps"
for polarization in Ea Eb; do
  run_source_only "$polarization" 100 "$coarse_label"
  run_source_only "$polarization" 50 "$fine_label"
done

echo "B200 source calibrations passed: $output_root"
