#!/usr/bin/env python3
"""Paper-style figures from the corrected true-edge scan.

Fig2G-style: weighting potential psi streamlines over device outline.
Fig3F-style: flake-average temperature-rise maps, beam at true edge (d=0), a & b.
Fig3I-style: I(d) for both polarizations + ratio r(d) with paper edge reference.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path('/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end')
WS = Path('/data/seunghyun/tairte4/sanity_v2_workspace')
OUT = R / 'edgetrue_paper_figures_20260802'
OUT.mkdir(exist_ok=True)

LABELS = ['dm2', 'dm1', 'dp0', 'dp1', 'dp2', 'dp3', 'dp5']
DIST = {'dm2': -2, 'dm1': -1, 'dp0': 0, 'dp1': 1, 'dp2': 2, 'dp3': 3, 'dp5': 5}
COL_A = '#2563eb'   # a-polarization (blue)
COL_B = '#d97706'   # b-polarization (amber) — CVD-safe pair with blue
PAPER_EDGE_RATIO = 0.8366

intent = json.load(open(WS / 'scan_true_positions.json'))

def summary(pol, lab):
    return json.load(open(R / f'edgetrue_thermal_{pol}_{lab}_20260802/summary.json'))

flake = np.array(summary('a', 'dp0')['geometry']['flake_vertices_um'])

def fields(pol, lab):
    return np.load(R / f'edgetrue_thermal_{pol}_{lab}_20260802/thermal_pte_fields.npz')

def centers(edges):
    return 0.5 * (edges[:-1] + edges[1:]) * 1e6

def draw_flake(ax, lw=1.2, color='0.25'):
    v = np.vstack([flake, flake[:1]])
    ax.plot(v[:, 0], v[:, 1], color=color, lw=lw, zorder=5)

# ---------------- Fig 2G style: psi streamlines ----------------
f = fields('a', 'dp0')
x, y = centers(f['x_edges_m']), centers(f['y_edges_m'])
psi = f['weighting_potential']
gx, gy = f['weighting_grad_x_m_inv'], f['weighting_grad_y_m_inv']
mask = f['flake_mask'].any(axis=2)

fig, ax = plt.subplots(figsize=(6.4, 5.4))
P = np.where(mask, psi, np.nan)
cs = ax.contour(x, y, P.T, levels=np.linspace(0.05, 0.95, 10),
                colors='0.55', linewidths=0.7)
sx = np.where(mask, -gx, np.nan)
sy = np.where(mask, -gy, np.nan)
xu = np.linspace(x.min(), x.max(), 300)
yu = np.linspace(y.min(), y.max(), 300)
from scipy.interpolate import RegularGridInterpolator
Xu, Yu = np.meshgrid(xu, yu, indexing='ij')
pts = np.stack([Xu.ravel(), Yu.ravel()], axis=1)
def to_uniform(a):
    itp = RegularGridInterpolator((x, y), np.nan_to_num(a),
                                  bounds_error=False, fill_value=0.0)
    out = itp(pts).reshape(Xu.shape)
    inside = RegularGridInterpolator((x, y), mask.astype(float),
                                     bounds_error=False, fill_value=0.0)(pts)
    out[inside.reshape(Xu.shape) < 0.5] = np.nan
    return out
ax.streamplot(xu, yu, to_uniform(sx).T, to_uniform(sy).T, color='#2563eb',
              density=1.1, linewidth=1.0, arrowsize=1.2)
draw_flake(ax)
ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
ax.set_title('Weighting potential ψ and −∇ψ streamlines (Fig. 2G style)')
ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(OUT / 'fig2G_style_psi_streamlines.png', dpi=220)
plt.close(fig)

# ---------------- Fig 3F style: dT maps at true edge ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
vmax = 0.0
maps = {}
for pol in ('a', 'b'):
    f = fields(pol, 'dp0')
    T = f['temperature_flake_average_K']
    m = f['flake_mask'].any(axis=2)
    maps[pol] = (centers(f['x_edges_m']), centers(f['y_edges_m']),
                 np.where(m, T, np.nan))
    vmax = max(vmax, np.nanmax(maps[pol][2]))
for ax, pol in zip(axes, ('a', 'b')):
    x, y, T = maps[pol]
    im = ax.pcolormesh(x, y, T.T * 1e3, cmap='inferno', vmin=0, vmax=vmax * 1e3,
                       shading='auto')
    draw_flake(ax, color='w')
    c = intent['positions']['dp0']['center_um']
    ax.plot(*c, marker='+', ms=12, mew=2, color='#22d3ee')
    ax.set_xlabel('x (µm)')
    ax.set_title(f'E ∥ {pol}   (beam at true edge, d = 0)')
    ax.set_aspect('equal')
axes[0].set_ylabel('y (µm)')
cb = fig.colorbar(im, ax=axes, shrink=0.85)
cb.set_label('flake-average ΔT (mK) at 285 µW incident')
fig.suptitle('Temperature rise, corrected beam placement (Fig. 3F style)')
fig.savefig(OUT / 'fig3F_style_dT_maps.png', dpi=220)
plt.close(fig)

# ---------------- Fig 3I style: I(d) + ratio ----------------
d_um = np.array([DIST[l] for l in LABELS], float)
Ia = np.array([summary('a', l)['PTE_current_A_at_285uW_incident'] for l in LABELS]) * 1e9
Ib = np.array([summary('b', l)['PTE_current_A_at_285uW_incident'] for l in LABELS]) * 1e9
ratio = np.abs(Ia) / np.abs(Ib)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 7.6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 2]})
ax1.axvline(0, color='0.8', lw=1)
ax1.plot(d_um, Ia, 'o-', color=COL_A, lw=2, ms=7, label='E ∥ a')
ax1.plot(d_um, Ib, 's-', color=COL_B, lw=2, ms=7, label='E ∥ b')
ax1.set_ylabel('I (nA) at 285 µW incident')
ax1.set_title('Photocurrent vs beam–edge distance (Fig. 3I style)')
ax1.legend(frameon=False)
ax1.annotate('flake edge', (0, ax1.get_ylim()[0]), textcoords='offset points',
             xytext=(4, 8), color='0.45', fontsize=9)

ax2.axvline(0, color='0.8', lw=1)
ax2.axhline(1.0, color='0.85', lw=1)
ax2.plot(d_um, ratio, 'D-', color='0.2', lw=2, ms=6, label='|I$_a$| / |I$_b$| (this work)')
ax2.axhline(PAPER_EDGE_RATIO, color='#dc2626', lw=1.5, ls='--',
            label=f'paper edge value {PAPER_EDGE_RATIO}')
ax2.set_xlabel('distance from true edge d (µm), + = into flake')
ax2.set_ylabel('|I$_a$| / |I$_b$|')
ax2.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig3I_style_current_and_ratio.png', dpi=220)
plt.close(fig)

table = {
    'd_um': d_um.tolist(),
    'I_a_nA': Ia.tolist(),
    'I_b_nA': Ib.tolist(),
    'ratio_abs_a_over_b': ratio.tolist(),
    'paper_edge_ratio': PAPER_EDGE_RATIO,
    'beam_readback': {
        'fit_center_err_um': {'a': 0.003, 'b': 0.016},
        'realized_w0_um': {'a': [9.033, 10.253], 'b': [10.313, 9.019]},
        'nominal_w0_um': 8.75,
    },
}
(OUT / 'scan_results_table.json').write_text(json.dumps(table, indent=1))
print('FIGURES DONE ->', OUT)
for l, da, db, r_ in zip(LABELS, Ia, Ib, ratio):
    print(f'{l}: Ia={da:+.3f} Ib={db:+.3f} r={r_:.4f}')
