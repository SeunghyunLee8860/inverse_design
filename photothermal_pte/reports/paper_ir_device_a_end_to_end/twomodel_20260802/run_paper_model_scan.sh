#!/bin/bash
# Track (1): PAPER-REPLICATION optics -> same thermal/Ramo pipeline as full-wave.
# CPU only (no Lumerical, no GPU, no licence).
set -u
REPO=/home/seunghyun/tairte4/pte_inverse_design_adfd
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
ART=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end
WS=/data/seunghyun/tairte4/sanity_v2_workspace
GEO=photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json
LOG=$WS/papermodel_scan.log
W0=${W0:-8.75}          # matched to the full-wave scan so ONLY the optics differ
PINC=4.2655553e-10      # common incident power (mean of the two empty refs)

declare -A CX CY
CX[dm2]=-12.61148133392282;  CY[dm2]=3.4865669456968704
CX[dm1]=-11.745178384230758; CY[dm1]=2.848871718840215
CX[dp0]=-10.749441789928829; CY[dp0]=2.1158989480346273
CX[dp1]=-9.74944178992883;   CY[dp1]=1.379787836923517
CX[dp2]=-8.749441789928829;  CY[dp2]=0.6436767258124059
CX[dp3]=-7.749441789928829;  CY[dp3]=-0.09243438529870551
CX[dp5]=-5.858130778810259;  CY[dp5]=-1.4846494351498754
declare -A GPU; GPU[a]=4; GPU[b]=3

echo "PAPER-MODEL SCAN START $(date) w0=${W0}um" >> "$LOG"
for pol in a b; do
  for lab in dm2 dm1 dp0 dp1 dp2 dp3 dp5; do
    opt=$ART/papermodel_optics_${pol}_${lab}_w${W0}
    out=$ART/papermodel_thermal_${pol}_${lab}_w${W0}
    [ -f "$out/summary.json" ] && { echo "SKIP $pol/$lab" >> "$LOG"; continue; }
    rm -rf "$opt" "$out"
    ( cd "$REPO" && $PY $WS/make_paper_model_optics.py \
        --template-case-dir $ART/edgetrue_finite_${pol}_${lab}_gpu${GPU[$pol]}_20260802 \
        --output-dir "$opt" --polarization $pol \
        --beam-center-x-um ${CX[$lab]} --beam-center-y-um ${CY[$lab]} \
        --waist-um $W0 --incident-power-w $PINC ) >> "$LOG" 2>&1 || { echo "OPTFAIL $pol/$lab" >> "$LOG"; continue; }
    ( cd "$REPO" && $PY photothermal_pte/validation/paper_ir_sanity/run_device_a_explicit_thermal_pte.py \
        --optical-case-dir "$opt" --output-dir "$out" \
        --thermal-domain-um 60 --si-depth-um 20 --core-step-nm 100 --flake-dz-nm 10 \
        --geometry device-a-polygon --geometry-contract-json $GEO \
        --thermal-model expanded --metal-thermalization isolated-lower-bound \
        --q-remap material-overlap ) >> "$WS/papermodel_thermal_${pol}_${lab}.log" 2>&1
    if [ -f "$out/summary.json" ]; then echo "OK $pol/$lab" >> "$LOG"
    else echo "THERMFAIL $pol/$lab" >> "$LOG"; fi
  done
done
echo "PAPER-MODEL SCAN END $(date)" >> "$LOG"
