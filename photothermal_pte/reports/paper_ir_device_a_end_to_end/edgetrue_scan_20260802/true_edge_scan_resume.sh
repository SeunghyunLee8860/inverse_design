#!/bin/bash
# Resume corrected true-edge scan (a: dp2,dp3,dp5 then b: all 7).
# Gates: license (>=9 free lum_fdtd_solve seats) + root-disk (>=3GB avail).
set -u
REPO=/home/seunghyun/tairte4/pte_inverse_design_adfd
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
ART=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end
WS=/data/seunghyun/tairte4/sanity_v2_workspace
LMUTIL=/home/seunghyun/lumerical_r12/opt/lumerical/v261/licensingclient/linx64/lmutil
LOG=$WS/edgetrue_resume.log
GEO=photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json

declare -A OX OY
OX[dm2]=-5.659265665288116;  OY[dm2]=4.165848336948196
OX[dm1]=-4.792962715596055;  OY[dm1]=3.52815311009154
OX[dp0]=-3.797226121294125;  OY[dp0]=2.7951803392859524
OX[dp1]=-2.797226121294127;  OY[dp1]=2.059069228174842
OX[dp2]=-1.7972261212941252; OY[dp2]=1.322958117063731
OX[dp3]=-0.7972261212941252; OY[dp3]=0.5868470059526196
OX[dp5]=1.0940848898244449;  OY[dp5]=-0.8053680438985502

gate() {
  while true; do
    avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "${avail:-0}" -lt 3 ]; then echo "$(date +%T) DISKWAIT avail=${avail}G" >> "$LOG"; sleep 120; continue; fi
    line=$($LMUTIL lmstat -c 1055@localhost -f lum_fdtd_solve 2>/dev/null | grep "Users of lum_fdtd_solve")
    tot=$(echo "$line" | sed -n 's/.*Total of \([0-9]*\) licenses issued.*/\1/p')
    use=$(echo "$line" | sed -n 's/.*Total of \([0-9]*\) licenses in use.*/\1/p')
    free=$(( ${tot:-0} - ${use:-99} ))
    if [ "$free" -ge 9 ]; then return 0; fi
    echo "$(date +%T) LICWAIT free=$free" >> "$LOG"; sleep 120
  done
}

run_case() {
  pol=$1; lab=$2; gpu=$3
  out=$ART/edgetrue_finite_${pol}_${lab}_gpu${gpu}_20260802
  if $PY - "$out/case_result.json" <<'EOF' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get("status")=="COMPLETED" else 1)
EOF
  then echo "SKIP edgetrue_finite_${pol}_${lab} (done)" >> "$LOG"; return 0; fi
  for att in 1 2 3 4; do
    gate
    rm -rf "$out"
    ( cd "$REPO" && XDG_CONFIG_HOME=$WS/lumcfg_$pol MPLCONFIGDIR=$WS/mpl \
      $PY photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py \
      --output-dir "$out" --case finite-flake --polarization $pol \
      --geometry device-a-polygon --device-a-geometry-json $GEO \
      --include-electrodes --local-xy-mesh-nm 50 --refinement-half-span-um 7 \
      --beam-offset-x-um ${OX[$lab]} --beam-offset-y-um ${OY[$lab]} \
      --incident-reference $ART/edgetrue_empty_${pol}_gpu${gpu}_20260802/case_result.json \
      --domain-um 60 --source-span-um 32 --waist-um 12 --scenario-waist-um 8.75 \
      --sio2-model palik-lossy --pml-layers 24 --flake-dz-nm 10 \
      --simulation-time-ps 4 --auto-shutoff-min 1e-5 \
      --execution-contract production --epsilon-c-model paper-b-closure \
      --gpu-device "GPU $gpu" --threads 3 ) >> "$WS/edgetrue_resume_${pol}_${lab}.log" 2>&1
    if $PY - "$out/case_result.json" <<'EOF' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get("status")=="COMPLETED" else 1)
EOF
    then echo "OK edgetrue_finite_${pol}_${lab} att$att" >> "$LOG"; return 0
    else echo "FAIL edgetrue_finite_${pol}_${lab} att$att" >> "$LOG"; sleep 90; fi
  done
  echo "GIVEUP edgetrue_finite_${pol}_${lab}" >> "$LOG"; return 1
}

echo "RESUME SCAN START $(date)" >> "$LOG"
for lab in dm2 dm1 dp0 dp1 dp2 dp3 dp5; do run_case a "$lab" 4; done
for lab in dm2 dm1 dp0 dp1 dp2 dp3 dp5; do run_case b "$lab" 3; done
echo "RESUME SCAN END $(date)" >> "$LOG"
