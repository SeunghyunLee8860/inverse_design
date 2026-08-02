#!/bin/bash
# Thermal step for the w0=11.5812 straight-45 waist-sensitivity optics.
# Mirrors the w0=8.75 control runs so the two waists are directly comparable.
set -u
REPO=/home/seunghyun/tairte4/worktrees/device_a_waist_sweep
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
A=/home/seunghyun/tairte4/artifacts/paper_ir_straight_45_edge_palik_w8p75
WS=/data/seunghyun/tairte4/sanity_v2_workspace
LOG=$WS/waist_thermal.log

echo "WAIST THERMAL START $(date)" >> "$LOG"
for pol in a b; do
  EDG=$A/edge_${pol}_palik_w11p58_gpu4_20260802
  OUT=$A/thermal_${pol}_w11p58_core100_dz10_L60_Si20_20260802
  # wait for the optical run to land (the optics chain may still be running)
  for i in $(seq 1 240); do
    [ -f "$EDG/case_result.json" ] && break
    sleep 60
  done
  [ -f "$EDG/case_result.json" ] || { echo "NOOPTICS $pol" >> "$LOG"; continue; }
  [ -f "$OUT/summary.json" ] && { echo "SKIP $pol" >> "$LOG"; continue; }
  rm -rf "$OUT"
  ( cd "$REPO" && $PY photothermal_pte/validation/paper_ir_sanity/run_device_a_explicit_thermal_pte.py \
      --optical-case-dir "$EDG" --output-dir "$OUT" \
      --thermal-domain-um 60 --si-depth-um 20 --core-step-nm 100 --flake-dz-nm 10 \
      --geometry straight-45-edge --thermal-model expanded \
      --metal-thermalization fail-closed --q-source TaIrTe4-only \
      ) >> "$WS/waist_thermal_${pol}.log" 2>&1
  if [ -f "$OUT/summary.json" ]; then echo "OK thermal/$pol" >> "$LOG"
  else echo "FAIL thermal/$pol" >> "$LOG"; fi
done
echo "WAIST THERMAL END $(date)" >> "$LOG"
