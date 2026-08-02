#!/usr/bin/env python3
"""Watch completed edgetrue optical artifacts, run thermal/PTE for each,
and stream per-position currents and a/b ratios to a live log."""
import json, subprocess, time
from pathlib import Path

R = Path('/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end')
REPO = Path('/home/seunghyun/tairte4/pte_inverse_design_adfd')
PY = '/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python'
GEO = 'photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json'
LABELS = ['dm2', 'dm1', 'dp0', 'dp1', 'dp2', 'dp3', 'dp5']
D_UM = {'dm2': -2, 'dm1': -1, 'dp0': 0, 'dp1': 1, 'dp2': 2, 'dp3': 3, 'dp5': 5}
GPU = {'a': 4, 'b': 3}
LIVE = Path('/data/seunghyun/tairte4/sanity_v2_workspace/live_ratio.log')
running: dict[tuple, subprocess.Popen] = {}
reported_current = set()
reported_ratio = set()

def log(msg):
    with LIVE.open('a') as f:
        f.write(msg + '\n')

def optics_done(pol, lab):
    p = R / f'edgetrue_finite_{pol}_{lab}_gpu{GPU[pol]}_20260802/case_result.json'
    try:
        return json.loads(p.read_text()).get('status') == 'COMPLETED'
    except Exception:
        return False

def thermal_dir(pol, lab):
    return R / f'edgetrue_thermal_{pol}_{lab}_20260802'

def thermal_done(pol, lab):
    return (thermal_dir(pol, lab) / 'summary.json').is_file()

def current_nA(pol, lab):
    d = json.loads((thermal_dir(pol, lab) / 'summary.json').read_text())
    return d['PTE_current_A_at_285uW_incident'] * 1e9

log('WATCHER START')
while True:
    for pol in ('a', 'b'):
        for lab in LABELS:
            key = (pol, lab)
            if thermal_done(pol, lab):
                if key in running:
                    running.pop(key, None)
                if key not in reported_current:
                    reported_current.add(key)
                    i_na = current_nA(pol, lab)
                    log(f'CURRENT {pol} d={D_UM[lab]:+d}um : I = {i_na:+.3f} nA')
                    other = 'b' if pol == 'a' else 'a'
                    if thermal_done(other, lab) and lab not in reported_ratio:
                        reported_ratio.add(lab)
                        ia = abs(current_nA('a', lab))
                        ib = abs(current_nA('b', lab))
                        log(f'RATIO d={D_UM[lab]:+d}um : |Ia|/|Ib| = {ia/ib:.4f}   (paper edge 0.8366)')
                continue
            if key in running:
                if running[key].poll() is not None:
                    running.pop(key, None)  # finished (summary check next loop)
                continue
            if optics_done(pol, lab) and len(running) < 6:
                out = thermal_dir(pol, lab)
                subprocess.run(['rm', '-rf', str(out)])
                cmd = [PY,
                       'photothermal_pte/validation/paper_ir_sanity/run_device_a_explicit_thermal_pte.py',
                       '--optical-case-dir', str(R / f'edgetrue_finite_{pol}_{lab}_gpu{GPU[pol]}_20260802'),
                       '--output-dir', str(out),
                       '--thermal-domain-um', '60', '--si-depth-um', '20',
                       '--core-step-nm', '100', '--flake-dz-nm', '10',
                       '--geometry', 'device-a-polygon',
                       '--geometry-contract-json', GEO,
                       '--thermal-model', 'expanded',
                       '--metal-thermalization', 'isolated-lower-bound',
                       '--q-remap', 'material-overlap']
                lf = open(f'/data/seunghyun/tairte4/sanity_v2_workspace/edgetrue_thermal_{pol}_{lab}.log', 'w')
                running[key] = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=lf)
                log(f'THERMAL LAUNCH {pol}/{lab}')
    if len(reported_ratio) == len(LABELS):
        log('ALL RATIOS DONE')
        break
    time.sleep(45)
