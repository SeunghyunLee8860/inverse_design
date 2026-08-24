#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${1:?usage: $0 OUTPUT_ROOT}"
if [[ -d "$output_root" ]] && [[ -n "$(find "$output_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty output root: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"

runner="$script_dir/run_lumerical_b200.sh"
case_script="$script_dir/25_run_lumerical_4um_exact_au_control.py"
gpu_index="${LUMERICAL_B200_GPU_INDEX:?set LUMERICAL_B200_GPU_INDEX to the physical NVIDIA B200 index}"
source_object_w0_um="${AU4UM_SOURCE_OBJECT_W0_UM:-3.956143303046143}"
mesh_refinement="${AU4UM_MESH_REFINEMENT:-conformal variant 1}"
mesh_label="baseline_xy100_z20_pml8_span20_z6_t1ps"

for polarization in Ea Eb; do
  source_dir="$output_root/${polarization}_source_only"
  "$runner" "$case_script" \
    --output-dir "$source_dir" --case source_only \
    --polarization "$polarization" --gpu-index "$gpu_index" \
    --source-object-w0-um "$source_object_w0_um" \
    --mesh-refinement "$mesh_refinement"
  source_json="$source_dir/source_only_${polarization}_${mesh_label}.json"
  "$runner" "$case_script" \
    --output-dir "$output_root/${polarization}_exact_empty" \
    --case empty --polarization "$polarization" --gpu-index "$gpu_index" \
    --source-object-w0-um "$source_object_w0_um" \
    --mesh-refinement "$mesh_refinement" \
    --source-calibration-json "$source_json"
  "$runner" "$case_script" \
    --output-dir "$output_root/${polarization}_import_rho0" \
    --case import_density --rho 0 --polarization "$polarization" \
    --gpu-index "$gpu_index" --source-object-w0-um "$source_object_w0_um" \
    --mesh-refinement "$mesh_refinement" \
    --source-calibration-json "$source_json"
  "$runner" "$case_script" \
    --output-dir "$output_root/${polarization}_import_rho1" \
    --case import_density --rho 1 --polarization "$polarization" \
    --gpu-index "$gpu_index" --source-object-w0-um "$source_object_w0_um" \
    --mesh-refinement "$mesh_refinement" \
    --source-calibration-json "$source_json"
  "$runner" "$case_script" \
    --output-dir "$output_root/${polarization}_exact_full" \
    --case full --polarization "$polarization" --gpu-index "$gpu_index" \
    --source-object-w0-um "$source_object_w0_um" \
    --mesh-refinement "$mesh_refinement" \
    --source-calibration-json "$source_json"
done
