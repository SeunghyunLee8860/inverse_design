#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${AU_LUMERICAL_PYTHON:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
lumerical_root="${AU_LUMERICAL_ROOT:-/opt/lumerical/v261}"

if [[ ! -x "$python_bin" ]]; then
  echo "missing executable Python: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$lumerical_root/api/python/lumapi.py" ]]; then
  echo "missing Lumerical lumapi.py below: $lumerical_root" >&2
  exit 2
fi

export AU_LUMERICAL_ROOT="$lumerical_root"
export VC_LUMERICAL_ROOT="$lumerical_root"
export LUMERICAL_ROOT="$lumerical_root"
export PYTHONPATH="$lumerical_root/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"

# This launcher is only for layout/index_detail work. The called command must
# not invoke fdtd.run or claim a GPU/B200 Maxwell certificate.
exec "$python_bin" "$@"
