#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${FDTDX_FRESH_PYTHON:-/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python}"
gpu_index="${FDTDX_FRESH_GPU_INDEX:?set FDTDX_FRESH_GPU_INDEX to one idle physical NVIDIA GPU index}"
campaign_root="${1:?pass one new absolute external campaign root}"
gpu_wrapper="$script_dir/run_fdtdx_fresh_gpu.sh"

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
shopt -s nullglob dotglob
campaign_entries=("$campaign_root"/*)
shopt -u nullglob dotglob
if (( ${#campaign_entries[@]} != 0 )); then
  echo "campaign root must be empty to prevent result overwrite" >&2
  exit 2
fi
if [[ -n "$(git -C "$repository" status --porcelain --untracked-files=all)" ]]; then
  echo "repository must be clean before the Courant campaign" >&2
  exit 2
fi
if [[ ! -x "$python_bin" || ! -x "$gpu_wrapper" ]]; then
  echo "missing Python or GPU wrapper executable" >&2
  exit 2
fi

mkdir "$campaign_root/contracts"
cd "$repository"

run_level() {
  local level="$1"
  local courant="$2"
  local case_path="$campaign_root/contracts/l500_anchor_t24_${level}.json"
  local case_sha
  local pair_dir="$campaign_root/source_pair_t24_${level}"
  local pair_path="$pair_dir/FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
  local pair_sha

  "$python_bin" -m \
    photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract \
    --output "$case_path" \
    --mesh-axis anchor --mesh-level 0 \
    --total-periods 24 --window-periods 4 --courant-factor "$courant"
  case_sha="$(sha256sum "$case_path" | cut -d " " -f 1)"

  for polarization in Ea Eb; do
    local source_dir="$campaign_root/source_t24_${level}_${polarization}"
    mkdir "$source_dir"
    FDTDX_FRESH_GPU_INDEX="$gpu_index" \
    FDTDX_FRESH_OUTPUT_DIR="$source_dir" \
      "$gpu_wrapper" -m \
      photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only \
      --polarization "$polarization" \
      --case-contract "$case_path" \
      --case-contract-sha256 "$case_sha"
  done

  mkdir "$pair_dir"
  "$python_bin" -m \
    photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair \
    --ea "$campaign_root/source_t24_${level}_Ea/FDTDX_FRESH_SOURCE_ONLY.json" \
    --eb "$campaign_root/source_t24_${level}_Eb/FDTDX_FRESH_SOURCE_ONLY.json" \
    --output-dir "$pair_dir"
  pair_sha="$(sha256sum "$pair_path" | cut -d " " -f 1)"

  for polarization in Ea Eb; do
    local material_dir="$campaign_root/l500_t24_${level}_${polarization}"
    mkdir "$material_dir"
    FDTDX_FRESH_GPU_INDEX="$gpu_index" \
    FDTDX_FRESH_OUTPUT_DIR="$material_dir" \
      "$gpu_wrapper" -m \
      photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot \
      --reference l_shape_4um_with_500nm_arms \
      --polarization "$polarization" \
      --case-contract "$case_path" \
      --case-contract-sha256 "$case_sha" \
      --source-pair "$pair_path" \
      --source-pair-sha256 "$pair_sha"
  done

  echo "COURANT_LEVEL_COMPLETE level=$level courant=$courant case_sha256=$case_sha source_pair_sha256=$pair_sha"
}

run_level c0p5 0.5
run_level c0p375 0.375
run_level c0p25 0.25

echo "FDTDX_FRESH_COURANT_CAMPAIGN_COMPLETE root=$campaign_root"
