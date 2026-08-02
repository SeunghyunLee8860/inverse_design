import json, os
import numpy as np
os.environ['MPLCONFIGDIR'] = '/data/seunghyun/tairte4/sanity_v2_workspace/mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Circle, Rectangle, FancyArrow

J = json.load(open('/home/seunghyun/tairte4/pte_inverse_design_adfd/photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json'))
flake = np.asarray(J['flake_vertices_code_um'], float)
top = np.asarray(J['top_metal_polygon_code_um'], float)
bot = np.asarray(J['bottom_metal_polygon_code_um'], float)
beam0 = np.asarray(J['pre_registered_beam_center_code_um'], float)
tseg = np.asarray(J['top_electrical_contact_segment_code_um'], float)
bseg = np.asarray(J['bottom_electrical_contact_segment_code_um'], float)
i4, i7 = J['off_axis_edge_vertex_indices']

# span-40 recentering (the frame of the actual scan runs)
half_source = 20.0
allm = np.vstack((top, bot))
omin = np.minimum(beam0 - half_source, allm.min(0))
omax = np.maximum(beam0 + half_source, allm.max(0))
sh = -0.5 * (omin + omax)
flake += sh; top += sh; bot += sh; beam0 = beam0 + sh; tseg += sh; bseg += sh

v4, v7 = flake[i4], flake[i7]
t_hat = (v7 - v4) / np.linalg.norm(v7 - v4)
n_hat = np.array([t_hat[1], -t_hat[0]])       # inward normal
mid = 0.5 * (v4 + v7)
S = [-1.5, 0.0, 1.0, 2.0, 3.0, 5.0]
centers = [mid + s * n_hat for s in S]

WX, WY = 7.30, 6.95   # realized 1/e2 radii (measured from incident_reference.npz)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1)
az = fig.add_subplot(1, 2, 2)

# ---------------- top view ----------------
ax.add_patch(MPoly(flake, closed=True, fc='#e8a33d', ec='#8a5a00', lw=1.5, label='TaIrTe$_4$ 130 nm', zorder=2))
ax.add_patch(MPoly(top, closed=True, fc='#ffd700', ec='#8a7000', alpha=0.85, lw=1.2, zorder=3))
ax.add_patch(MPoly(bot, closed=True, fc='#ffd700', ec='#8a7000', alpha=0.85, lw=1.2, zorder=3, label='Au 50/Ti 5 nm electrodes'))
ax.plot([v4[0], v7[0]], [v4[1], v7[1]], '-', color='#cc3311', lw=3.5, zorder=4, label='off-axis edge')
ax.plot(tseg[:, 0], tseg[:, 1], '-', color='#0077bb', lw=4, zorder=5, label='contacts ($\\psi$=1 / $\\psi$=0)')
ax.plot(bseg[:, 0], bseg[:, 1], '-', color='#0077bb', lw=4, zorder=5)

# scan line and positions
line_lo = mid - 4.0 * n_hat
line_hi = mid + 7.0 * n_hat
ax.plot([line_lo[0], line_hi[0]], [line_lo[1], line_hi[1]], '--', color='#228833', lw=1.4, zorder=6)
for s, c in zip(S, centers):
    ax.plot(c[0], c[1], 'x', color='#228833', ms=9, mew=2.4, zorder=7)
    off = 1.1 * np.array([-n_hat[1], n_hat[0]])
    ax.annotate(f's={s:g}', (c[0] + off[0], c[1] + off[1]), fontsize=8, color='#228833', ha='center', zorder=8)
# 1/e2 beam footprint at two representative positions
from matplotlib.patches import Ellipse
for s_show, ls in ((0.0, '-'), (5.0, '--')):
    c = mid + s_show * n_hat
    ax.add_patch(Ellipse(c, 2 * WX, 2 * WY, fill=False, ec='#228833', lw=1.8, ls=ls, zorder=6))
ax.add_patch(Ellipse((99, 99), 1, 1, fill=False, ec='#228833', lw=1.8, label='realized beam 1/e$^2$ (7.3/7.0 um)'))

d = 30.0
ax.add_patch(Rectangle((-d, -d), 2 * d, 2 * d, fill=False, ec='k', lw=1.0, ls=':'))
ax.add_patch(Rectangle((beam0[0] - 20, beam0[1] - 20), 40, 40, fill=False, ec='#888', lw=1.0, ls='--'))
ax.add_patch(Rectangle((beam0[0] - 15, beam0[1] - 15), 30, 30, fill=False, ec='#bb66bb', lw=1.2, ls='-.'))
ax.annotate('60 um FDTD domain (PML x24)', (-d + 1, d - 2.4), fontsize=9)
ax.annotate('40 um source aperture', (beam0[0] - 19.5, beam0[1] + 18.3), fontsize=9, color='#666')
ax.annotate('fixed 50-nm fine mesh (+/-15 um)', (beam0[0] - 14.5, beam0[1] - 14.4), fontsize=8.5, color='#bb66bb')
ax.annotate('', xytext=(22, -27), xy=(28, -27), arrowprops=dict(arrowstyle='->', color='k'))
ax.text(23.2, -26.2, 'b (E$\\parallel$b)', fontsize=10)
ax.annotate('', xytext=(-27, 20), xy=(-27, 26), arrowprops=dict(arrowstyle='->', color='k'))
ax.text(-26.4, 21.0, 'a (E$\\parallel$a)', rotation=90, fontsize=10)
ax.set_xlim(-31, 31); ax.set_ylim(-31, 31); ax.set_aspect('equal')
ax.set_xlabel('lab x = crystal b (um)'); ax.set_ylabel('lab y = crystal a (um)')
ax.set_title('TOP VIEW - Fig.3I line scan: 6 beam positions across the off-axis edge')
ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95)

# ---------------- side view (cut along the scan line) ----------------
# horizontal axis: signed distance s along n_hat through the edge midpoint
az.axhspan(-3.415, -0.415, color='#bbbbbb')
az.axhspan(-0.415, -0.130, color='#88ccee')
az.axhspan(-0.130, 0.0, xmin=0.36, color='#e8a33d')   # flake exists for s>0 (inside)
az.axhspan(0.055, 10.5, color='#f4f8fb')
az.axhspan(0.0, 0.055, xmin=0.0, xmax=0.0)  # placeholder
az.text(6.9, -1.8, 'Si (n=3.425, lossless)', fontsize=9, ha='right')
az.text(6.9, -0.27, 'SiO$_2$ 285 nm  Palik lossy $\\epsilon$=3.73+0.19j', fontsize=9, ha='right')
az.text(6.8, -0.062, 'TaIrTe$_4$ 130 nm', fontsize=9, ha='right', color='#5a3a00')
az.text(-3.8, -0.062, '(no flake, s<0)', fontsize=8, color='#666')
az.text(6.9, 3.1, 'air', fontsize=9, ha='right')

# source plane and rays for each scan position
az.axhline(5.0, color='#228833', ls='--', lw=2)
az.text(-3.9, 5.15, 'Gaussian source plane z=+5 um  ($\\lambda$=11 um, -z, scalar)', fontsize=9, color='#228833')
for s in S:
    az.annotate('', xytext=(s, 5.0), xy=(s, 0.35), arrowprops=dict(arrowstyle='->', color='#228833', lw=1.6, alpha=0.85))
    az.plot([s], [-0.065], 'x', color='#228833', ms=8, mew=2)
az.text(0.1, 2.3, 'beam center scanned:\ns = -1.5, 0, 1, 2, 3, 5 um\n(focus @ flake midplane)', fontsize=9, color='#228833')

# edge position marker
az.axvline(0.0, color='#cc3311', lw=2.5)
az.text(0.08, -0.95, 'off-axis edge (s=0)', color='#cc3311', fontsize=9, rotation=90, va='center')

# Q analysis box (z-range)
az.add_patch(Rectangle((-4.0, -0.130), 11.0, 0.185, fill=False, ec='#bb66bb', lw=1.6, ls='-.'))
az.text(6.9, 0.14, 'Q control volume z: [-130 nm, +55 nm]', fontsize=8.5, color='#bb66bb', ha='right')
# reference plane
az.axhline(0.6, color='#4477aa', ls=':', lw=1.4)
az.text(-3.9, 0.68, 'incident-reference plane z=+0.6 um (empty stack run)', fontsize=8, color='#4477aa')

az.set_xlim(-4.0, 7.0)
az.set_ylim(-3.6, 6.2)
az.set_yscale('symlog', linthresh=0.7)
az.set_xlabel('signed distance from edge s along scan direction (um, + into flake)')
az.set_ylabel('z (um, symlog)')
az.set_title('SIDE VIEW - cut along the scan line (beam positions & stack)')
fig.tight_layout()
out = '/data/seunghyun/tairte4/sanity_v2_workspace/inverse_design/photothermal_pte/reports/paper_ir_device_a_sanity_v2/lossy_sio2_scenario/EDGE_SCAN_SETUP.png'
fig.savefig(out, dpi=170)
print(out)
print('edge mid (span40 frame):', mid, ' beam frozen:', beam0, ' n_hat:', n_hat)
