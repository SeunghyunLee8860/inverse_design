#!/usr/bin/env python3
"""Two-model comparison figure: paper-replication optics vs full-wave optics,
both pushed through the identical thermal / weighting-potential / Shockley-Ramo
pipeline on the identical Device-A geometry.  Only the optics differ.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path('/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end')
A45 = Path('/home/seunghyun/tairte4/artifacts/paper_ir_straight_45_edge_palik_w8p75')
OUT = R / 'twomodel_figures_20260802'
OUT.mkdir(exist_ok=True)

LAB = ['dm2', 'dm1', 'dp0', 'dp1', 'dp2', 'dp3', 'dp5']
DIST = {'dm2': -2, 'dm1': -1, 'dp0': 0, 'dp1': 1, 'dp2': 2, 'dp3': 3, 'dp5': 5}
PAPER_R = 0.8366
TMM_R = 0.17673296 / 0.26328721
C_PM, C_FW, C_PAPER = '#0f766e', '#b45309', '#be123c'


def cur(pol, lab, mode):
    p = (R / f'papermodel_thermal_{pol}_{lab}_w8.75/summary.json' if mode == 'pm'
         else R / f'edgetrue_thermal_{pol}_{lab}_20260802/summary.json')
    return json.loads(p.read_text())['PTE_current_A_at_285uW_incident'] * 1e9


d = np.array([DIST[l] for l in LAB], float)
Ia_pm = np.array([cur('a', l, 'pm') for l in LAB])
Ib_pm = np.array([cur('b', l, 'pm') for l in LAB])
Ia_fw = np.array([cur('a', l, 'fw') for l in LAB])
Ib_fw = np.array([cur('b', l, 'fw') for l in LAB])
r_pm, r_fw = np.abs(Ia_pm / Ib_pm), np.abs(Ia_fw / Ib_fw)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 8.0), sharex=True,
                               gridspec_kw={'height_ratios': [3, 2.2]})
ax1.axvline(0, color='0.85', lw=1)
ax1.plot(d, Ia_pm, 'o--', color=C_PM, lw=1.8, ms=6, mfc='w', label='E∥a  paper-model optics')
ax1.plot(d, Ib_pm, 's--', color=C_PM, lw=1.8, ms=6, label='E∥b  paper-model optics')
ax1.plot(d, Ia_fw, 'o-', color=C_FW, lw=2.2, ms=7, mfc='w', label='E∥a  full-wave optics')
ax1.plot(d, Ib_fw, 's-', color=C_FW, lw=2.2, ms=7, label='E∥b  full-wave optics')
ax1.set_ylabel('I (nA) at 285 µW incident')
ax1.set_title('Device A, λ = 11 µm — identical thermal/Shockley-Ramo pipeline,\n'
              'only the optical model differs', fontsize=11)
ax1.legend(frameon=False, fontsize=9, ncol=2)

ax2.axvline(0, color='0.85', lw=1)
ax2.axhline(1.0, color='0.9', lw=1)
ax2.plot(d, r_pm, 'D--', color=C_PM, lw=2, ms=6, label='paper-model optics (this work)')
ax2.plot(d, r_fw, 'D-', color=C_FW, lw=2.2, ms=7, label='full-wave optics (this work)')
ax2.axhline(PAPER_R, color=C_PAPER, lw=1.6, ls=':', label=f'paper Fig. 3J measured  {PAPER_R}')
ax2.axhline(TMM_R, color='0.45', lw=1.2, ls='-.', label=f'TMM absorption ratio  {TMM_R:.3f}')
ax2.set_xlabel('distance from the flake boundary d (µm), + = into the flake')
ax2.set_ylabel('|I$_a$| / |I$_b$|')
ax2.legend(frameon=False, fontsize=8.5, loc='upper left')
ax2.annotate('paper-model ratio is position-independent\n(spread < 0.002) — it is pinned to the\nTMM absorption ratio by construction',
             xy=(1.0, r_pm[3]), xytext=(-1.6, 0.93), fontsize=8, color=C_PM,
             arrowprops=dict(arrowstyle='->', color=C_PM, lw=1))
fig.tight_layout()
fig.savefig(OUT / 'twomodel_current_and_ratio.png', dpi=220)
plt.close(fig)

# ---- waist sensitivity on the paper's own smooth 45-degree edge ----
S = {('8.75', 'a'): 'thermal_a_explicit3d_core100_dz10_L60_Si20_20260801_retry4',
     ('8.75', 'b'): 'thermal_b_explicit3d_core100_dz10_L60_Si20_20260801',
     ('11.58', 'a'): 'thermal_a_w11p58_core100_dz10_L60_Si20_20260802',
     ('11.58', 'b'): 'thermal_b_w11p58_core100_dz10_L60_Si20_20260802'}
M = {k: json.loads((A45 / v / 'summary.json').read_text()) for k, v in S.items()}
metrics = [('max_abs_grad_T_y_K_m', 'max |∇T| along a\n(paper Fig. 3G comparator)'),
           ('max_abs_edge_normal_gradient_K_m', 'max |∇T| edge-normal'),
           ('p99_abs_edge_normal_gradient_K_m', 'p99 |∇T| edge-normal'),
           ('TaIrTe4_area_average_rise_K', 'area-average ΔT'),
           ]
w = ['8.75', '11.58']
fig, ax = plt.subplots(figsize=(7.4, 4.6))
x = np.arange(len(metrics))
for i, ww in enumerate(w):
    vals = [M[(ww, 'a')]['straight_edge_metrics'][k] / M[(ww, 'b')]['straight_edge_metrics'][k]
            for k, _ in metrics]
    ax.bar(x + (i - 0.5) * 0.34, vals, 0.32,
           color=[C_FW, C_PM][i], label=f'w₀ = {ww} µm')
ax.axhline(1.0, color='0.6', lw=1)
ax.axhline(PAPER_R, color=C_PAPER, lw=1.5, ls=':', label=f'paper measured {PAPER_R}')
ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=8.5)
ax.set_ylabel('a / b')
ax.set_title('Waist sensitivity on the paper\'s own smooth 45° edge:\n'
             'the edge hotspot STRENGTHENS with a larger beam', fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'waist_sensitivity_45deg.png', dpi=220)
plt.close(fig)

json.dump({'d_um': d.tolist(),
           'paper_model': {'I_a_nA': Ia_pm.tolist(), 'I_b_nA': Ib_pm.tolist(), 'ratio': r_pm.tolist()},
           'full_wave': {'I_a_nA': Ia_fw.tolist(), 'I_b_nA': Ib_fw.tolist(), 'ratio': r_fw.tolist()},
           'paper_measured_ratio': PAPER_R, 'TMM_absorption_ratio': TMM_R,
           'waist_sensitivity_45deg': {
               ww: {k: M[(ww, 'a')]['straight_edge_metrics'][k] / M[(ww, 'b')]['straight_edge_metrics'][k]
                    for k, _ in metrics} for ww in w}},
          open(OUT / 'twomodel_results.json', 'w'), indent=1)
print('DONE ->', OUT)
print(f'paper-model r: {r_pm.min():.4f}..{r_pm.max():.4f}   full-wave r: {r_fw.min():.4f}..{r_fw.max():.4f}')
