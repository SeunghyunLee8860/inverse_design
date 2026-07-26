# Shared production environment for the TaIrTe4 500 nm inverse design.
#
# Sourced by BOTH run_inverse_design.sh and smoke_single_evaluator.sh, so a
# smoke test provably runs under the same environment as production instead of
# a hand-copied approximation that can drift.
#
# Caller must set: GPU (e.g. GPU="GPU 0").  Optional: FDTD_THREADS.

# --- Lumerical API pinning -------------------------------------------------
# The system install /opt/lumerical is FORBIDDEN: its lumapi fails on
# importdataset ("Failed to evaluate code") inside the FieldRegion adjoint, and
# a wrong API paired with the r12 fdtd-engine only shows up minutes into a
# solve.  Pin every variable the discovery paths consult, and strip any
# /opt/lumerical entry a parent shell put on PYTHONPATH (observed polluted 4x).
# Dropping PYTHONPATH entirely is safe -- conda site-packages does not need it.
R12="${VC_LUMERICAL_ROOT:-/home/seunghyun/lumerical_r12/opt/lumerical/v261}"
case "$R12" in
  /opt/lumerical|/opt/lumerical/*)
    echo "[fatal] approved root must not be under /opt/lumerical: $R12" >&2; exit 3 ;;
esac
if [ ! -f "$R12/api/python/lumapi.py" ]; then
  echo "[fatal] no lumapi under approved root: $R12" >&2; exit 3
fi
export VC_LUMERICAL_ROOT="$R12"
export LUMERICAL_ROOT="$R12"
export LUMERICAL_PYTHONPATH="$R12/api/python"

if [ -n "${PYTHONPATH:-}" ]; then
  # Anchored filter: keep entries that are NOT under /opt/lumerical.  A plain
  # substring match would also drop the approved r12 tree, whose path contains
  # the text "/opt/lumerical".
  CLEAN_PP=""
  OLD_IFS="$IFS"; IFS=":"
  for _p in $PYTHONPATH; do
    case "$_p" in
      ""|/opt/lumerical|/opt/lumerical/*) continue ;;
    esac
    if [ -z "$CLEAN_PP" ]; then CLEAN_PP="$_p"; else CLEAN_PP="$CLEAN_PP:$_p"; fi
  done
  IFS="$OLD_IFS"; unset _p OLD_IFS
  if [ "$CLEAN_PP" != "$PYTHONPATH" ]; then
    echo "[env] purged /opt/lumerical from PYTHONPATH"
    echo "[env]   was: $PYTHONPATH"
    echo "[env]   now: ${CLEAN_PP:-<unset>}"
  fi
  if [ -n "$CLEAN_PP" ]; then export PYTHONPATH="$CLEAN_PP"; else unset PYTHONPATH; fi
fi

# --- solver / GPU ----------------------------------------------------------
export CL_GPU_DEVICE="$GPU" LUMERICAL_SESSION_GPU_DEVICE="$GPU"
export FDTD_THREADS="${FDTD_THREADS:-16}"

# --- production optical contract (see IMPLEMENTATION_STATUS.md) ------------
export PERIOD_UM=6.0 TARGET_WL_UM=4.0
export SOURCE_WL_START_UM=3.0 SOURCE_WL_STOP_UM=6.0        # broadband source
export MATERIAL_FIT_START_UM=2.7 MATERIAL_FIT_STOP_UM=13.2 # material fit range
export BULK_MESH_MODE=auto MESH_ACCURACY=5                 # fast_bulk REJECTED (Phase C)
export MFS_UM=0.5 MGS_UM=0.5 ID_SEED=7 ID_SEED_AMP=0.10
export MSOPT_MAPPING=periodic_constrained

# --- certified runtime configuration (Phase A-D, 2026-07-26) ----------------
# combined vector adjoint + 2 ps: full-chain AD/FD 1.80%/2.36%/1.27% (<5%),
# objective evaluation 2518 s -> 915 s.  Explicit env overrides still win;
# unset VC_* here reproduces the legacy split/4 ps path bit-identically.
export VC_ADJOINT_COMPONENT_MODE="${VC_ADJOINT_COMPONENT_MODE:-combined}"
export VC_SIM_TIME_S="${VC_SIM_TIME_S:-2e-12}"
export VC_AUTO_SHUTOFF_MIN="${VC_AUTO_SHUTOFF_MIN:-1e-8}"
