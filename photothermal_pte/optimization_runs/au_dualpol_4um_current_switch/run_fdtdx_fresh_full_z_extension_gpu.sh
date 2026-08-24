#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${FDTDX_FRESH_PYTHON:-/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python}"
gpu_index="${FDTDX_FRESH_GPU_INDEX:?set FDTDX_FRESH_GPU_INDEX to one idle physical NVIDIA GPU index}"
campaign_root="${1:?pass the existing absolute external full-z campaign root}"
level="${2:?pass z16 or z32}"
expected_prior_case_sha="${3:?pass the prior-level case-file SHA256}"
expected_prior_pair_sha="${4:?pass the prior-level source-pair SHA256}"
gpu_wrapper="$script_dir/run_fdtdx_fresh_gpu.sh"

case "$level" in
  z16) prior_level="z8" ;;
  z32) prior_level="z16" ;;
  *)
    echo "extension level must be z16 or z32" >&2
    exit 2
    ;;
esac
if [[ "$campaign_root" != /* || ! -d "$campaign_root" || ! -w "$campaign_root" ]]; then
  echo "campaign root must be an existing writable absolute directory" >&2
  exit 2
fi
case "$campaign_root" in
  "$repository"|"$repository"/*)
    echo "campaign root must be outside the Git repository" >&2
    exit 2
    ;;
esac
if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
  echo "repository must be clean before the full-z extension" >&2
  exit 2
fi
if [[ ! -x "$python_bin" || ! -x "$gpu_wrapper" ]]; then
  echo "missing Python or GPU wrapper executable" >&2
  exit 2
fi

prior_case="$campaign_root/contracts/l500_full_z_${prior_level}.json"
prior_pair="$campaign_root/source_pair_full_z_${prior_level}/FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
if [[ ! -f "$prior_case" || ! -f "$prior_pair" ]]; then
  echo "prior level contract and source pair must exist" >&2
  exit 2
fi
if [[ "$(sha256sum "$prior_case" | cut -d " " -f 1)" != "$expected_prior_case_sha" ]]; then
  echo "prior level contract SHA256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$prior_pair" | cut -d " " -f 1)" != "$expected_prior_pair_sha" ]]; then
  echo "prior level source-pair SHA256 mismatch" >&2
  exit 2
fi
for polarization in Ea Eb; do
  if [[ ! -f "$campaign_root/l500_full_z_${prior_level}_${polarization}/FDTDX_FRESH_EXACT_BINARY_PILOT.json" ]]; then
    echo "prior material report is missing for $polarization" >&2
    exit 2
  fi
done

case_path="$campaign_root/contracts/l500_full_z_${level}.json"
pair_dir="$campaign_root/source_pair_full_z_${level}"
pair_path="$pair_dir/FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
for target in   "$case_path"   "$campaign_root/source_full_z_${level}_Ea"   "$campaign_root/source_full_z_${level}_Eb"   "$pair_dir"   "$campaign_root/l500_full_z_${level}_Ea"   "$campaign_root/l500_full_z_${level}_Eb"
do
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite extension target: $target" >&2
    exit 2
  fi
done

cd "$repository"
"$python_bin" -m   photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_extension_case   --level "$level" --output "$case_path"
case_sha="$(sha256sum "$case_path" | cut -d " " -f 1)"

for polarization in Ea Eb; do
  source_dir="$campaign_root/source_full_z_${level}_${polarization}"
  mkdir "$source_dir"
  FDTDX_FRESH_GPU_INDEX="$gpu_index"   FDTDX_FRESH_OUTPUT_DIR="$source_dir"     "$gpu_wrapper" -m     photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only     --polarization "$polarization"     --case-contract "$case_path"     --case-contract-sha256 "$case_sha"
done

mkdir "$pair_dir"
"$python_bin" -m   photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair   --ea "$campaign_root/source_full_z_${level}_Ea/FDTDX_FRESH_SOURCE_ONLY.json"   --eb "$campaign_root/source_full_z_${level}_Eb/FDTDX_FRESH_SOURCE_ONLY.json"   --output-dir "$pair_dir"
pair_sha="$(sha256sum "$pair_path" | cut -d " " -f 1)"

for polarization in Ea Eb; do
  material_dir="$campaign_root/l500_full_z_${level}_${polarization}"
  mkdir "$material_dir"
  FDTDX_FRESH_GPU_INDEX="$gpu_index"   FDTDX_FRESH_OUTPUT_DIR="$material_dir"     "$gpu_wrapper" -m     photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot     --reference l_shape_4um_with_500nm_arms     --polarization "$polarization"     --case-contract "$case_path"     --case-contract-sha256 "$case_sha"     --source-pair "$pair_path"     --source-pair-sha256 "$pair_sha"
done

echo "FDTDX_FRESH_FULL_Z_EXTENSION_COMPLETE level=$level case_sha256=$case_sha source_pair_sha256=$pair_sha root=$campaign_root"
