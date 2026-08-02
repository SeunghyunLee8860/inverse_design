#!/bin/bash
# Track (2b): waist sensitivity of the E||a edge hotspot on the paper's OWN
# idealized geometry (smooth straight 45-degree edge, no staircase, no
# electrodes).  w0 = 11.68 um is the alternative reading of the paper's quoted
# spot size (FWHM = lambda/2NA = 13.75 um at 11 um); the existing w0 = 8.75 um
# runs are the lambda/(pi*NA) reading.  A bigger beam dilutes the ~1 um edge
# hotspot relative to the bulk, so this tests whether the hotspot mechanism
# survives a realistic beam.
set -u
REPO=/home/seunghyun/tairte4/worktrees/device_a_waist_sweep  # branch agent/validate-inverse-design-pte-adfd: same code that produced the w0=8.75 straight-45 control
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
A=/home/seunghyun/tairte4/artifacts/paper_ir_straight_45_edge_palik_w8p75
WS=/data/seunghyun/tairte4/sanity_v2_workspace
LOG=$WS/waist1168.log
W0=11.5812
# one-step multiplicative calibration reused from the validated w0=8.75 point
# (source object 8.610602974768 for target 8.75 -> factor 0.9840689)
SRCOBJ=11.493924885176   # verified: realizes w0 = 11.5812 um at the flake plane
declare -A GPU; GPU[a]=4; GPU[b]=4   # serialize on the one verified-idle GPU

echo "WAIST-SENSITIVITY w0=${W0} srcobj=${SRCOBJ} START $(date)" >> "$LOG"

# ---- step 1: source-only calibration check -------------------------------
SRC=$A/source_w11p58_cal_gpu4_20260802
if [ ! -f "$SRC/source_only_case_result.json" ]; then
  ( cd "$REPO" && $PY photothermal_pte/validation/paper_ir_sanity/validate_paper_ir_source_only_gpu.py \
      --output-dir "$SRC" --target-waist-um $W0 --source-object-waist-um $SRCOBJ \
      --duration-ps 4 --auto-shutoff-min 1e-5 --gpu-device 'GPU 4' --threads 8 \
      --mesh-accuracy 5 ) >> "$WS/waist1168_source.log" 2>&1
fi
$PY - "$SRC/source_only_case_result.json" <<'EOF' >> "$LOG" 2>&1
import json,sys
r=json.load(open(sys.argv[1]))
p=r["planes"]["flake_target_plane"]; a=r["acceptance"]
print(f'  SOURCE CAL: realized w0 = {p["fitted_waist_effective_m"]*1e6:.4f} um '
      f'(target 11.5812), ellipticity={p["fitted_xy_ellipticity"]:.4f}, acceptance={a}')
# The acceptance flags compare against the source-object input, not our target,
# so they are informational here.  Only require that the run produced a fit.
sys.exit(0 if r.get("source_only_gate_passed") else 1)
EOF
if [ $? -ne 0 ]; then echo "  SOURCE CAL FAILED -> stop (recalibrate before spending edge runs)" >> "$LOG"; exit 1; fi

# ---- step 2+3: empty reference then edge, per polarization ---------------
for pol in a b; do
  g=${GPU[$pol]}
  EMP=$A/empty_${pol}_palik_w11p58_gpu${g}_20260802
  EDG=$A/edge_${pol}_palik_w11p58_gpu${g}_20260802
  COMMON="--geometry straight-45-edge --polarization $pol --domain-um 60 --pml-layers 24 \
    --flake-dz-nm 5 --local-xy-mesh-nm 100 --simulation-time-ps 4 --auto-shutoff-min 1e-5 \
    --source-span-um 50 --waist-um $W0 --source-object-waist-um $SRCOBJ \
    --execution-contract waist-sensitivity --substrate-optical-model lumerical-palik-11um \
    --matched-lossy-control-volume --source-only-reference $SRC/source_only_case_result.json \
    --gpu-device 'GPU ${g}'"
  if [ ! -f "$EMP/case_result.json" ]; then
    ( cd "$REPO" && eval $PY photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py \
        --output-dir "$EMP" --case empty-stack $COMMON ) >> "$WS/waist1168_empty_${pol}.log" 2>&1
  fi
  [ -f "$EMP/case_result.json" ] || { echo "EMPTYFAIL $pol" >> "$LOG"; continue; }
  echo "OK empty/$pol" >> "$LOG"
  if [ ! -f "$EDG/case_result.json" ]; then
    ( cd "$REPO" && eval $PY photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py \
        --output-dir "$EDG" --case finite-flake $COMMON \
        --incident-reference "$EMP/case_result.json" ) >> "$WS/waist1168_edge_${pol}.log" 2>&1
  fi
  [ -f "$EDG/case_result.json" ] || { echo "EDGEFAIL $pol" >> "$LOG"; continue; }
  echo "OK edge/$pol" >> "$LOG"
done
echo "WAIST-SENSITIVITY OPTICS END $(date)" >> "$LOG"
