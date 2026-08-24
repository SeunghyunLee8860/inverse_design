#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
python_bin="${AU_LUMERICAL_PYTHON:-/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python}"
# Keep layout-only material/Jacobian reads on the same audited API/CAD build
# as the Maxwell forward and adjoint launchers. The system /opt tree is the
# incompatible R1.0 build 4413.
lumerical_root="${AU_LUMERICAL_ROOT:-/home/seunghyun/lumerical_r12/opt/lumerical/v261}"

if [[ ! -x "$python_bin" ]]; then
  echo "missing executable Python: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$lumerical_root/api/python/lumapi.py" ]]; then
  echo "missing Lumerical lumapi.py below: $lumerical_root" >&2
  exit 2
fi
if [[ ! -f "$lumerical_root/VERSION" ]] || \
   ! grep -qx 'MAJORRELEASE=2026R1' "$lumerical_root/VERSION" || \
   ! grep -qx 'MINORRELEASE=2' "$lumerical_root/VERSION" || \
   ! grep -qx 'BUILDNUMBER=4522' "$lumerical_root/VERSION"; then
  echo "layout launcher requires Lumerical 2026 R1.2 build 4522: $lumerical_root" >&2
  exit 2
fi

export AU_LUMERICAL_ROOT="$lumerical_root"
export VC_LUMERICAL_ROOT="$lumerical_root"
export LUMERICAL_ROOT="$lumerical_root"
export PYTHONPATH="$lumerical_root/api/python:$repository${PYTHONPATH:+:$PYTHONPATH}"

# This launcher is only for layout/index_detail work. The called command must
# not invoke fdtd.run or claim a GPU/B200 Maxwell certificate.
exec "$python_bin" "$@"
