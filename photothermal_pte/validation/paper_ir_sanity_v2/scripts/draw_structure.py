import json, os
import numpy as np
os.environ['MPLCONFIGDIR'] = '/data/seunghyun/tairte4/sanity_v2_workspace/mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Circle

J = json.load(open('/home/seunghyun/tairte4/pte_inverse_design_adfd/photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json'))
flake = np.asarray(J['flake_vertices_code_um'], float)
top = np.asarray(J['top_metal_polygon_code_um'], float)
bot = np.asarray(J['bottom_metal_polygon_code_um'], float)
beam = np.asarray(J['pre_registered_beam_center_code_um'], float)
tseg = np.asarray(J['top_electrical_contact_segment_code_um'], float)
bseg = np.asarray(J['bottom_electrical_contact_segment_code_um'], float)
i4, i7 = J['off_axis_edge_vertex_indices']
allm = np.vstack((top, bot))
omin = np.minimum(beam - 25.0, allm.min(0))
omax = np.maximum(beam + 25.0, allm.max(0))
sh = -0.5 * (omin + omax)
flake += sh; top += sh; bot += sh; beam += sh; tseg += sh; bseg += sh

fig, (ax, az) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={'width_ratios': [1.35, 1]})
ax.add_patch(MPoly(flake, closed=True, fc='#e8a33d', ec='#8a5a00', lw=1.5, label='TaIrTe$_4$ flake (130 nm)', zorder=2))
ax.add_patch(MPoly(top, closed=True, fc='#ffd700', ec='#8a7000', alpha=0.85, lw=1.2, zorder=3))
ax.add_patch(MPoly(bot, closed=True, fc='#ffd700', ec='#8a7000', alpha=0.85, lw=1.2, zorder=3, label='Au(50nm)/Ti(5nm) electrodes'))
v4, v7 = flake[i4], flake[i7]
ax.plot([v4[0], v7[0]], [v4[1], v7[1]], '-', color='#cc3311', lw=3.5, zorder=4, label='off-axis edge (Fig.3 comparator)')
ax.plot(tseg[:, 0], tseg[:, 1], '-', color='#0077bb', lw=4, zorder=5, label='electrical contact segments ($\\psi$=1 / $\\psi$=0)')
ax.plot(bseg[:, 0], bseg[:, 1], '-', color='#0077bb', lw=4, zorder=5)
ax.add_patch(Circle(beam, 6.83, fill=False, ec='#cc3311', lw=2.2, zorder=6, label='beam w$_0$=6.83 um (this scenario)'))
ax.add_patch(Circle(beam, 12.0, fill=False, ec='#cc3311', lw=1.4, ls='--', zorder=6, label='baseline w$_0$=12 um'))
ax.plot([beam[0]], [beam[1]], 'x', color='#cc3311', ms=10, mew=2.5, zorder=7)
d = 30.0
ax.add_patch(plt.Rectangle((-d, -d), 2 * d, 2 * d, fill=False, ec='k', lw=1.0, ls=':'))
ax.add_patch(plt.Rectangle((-25, -25), 50, 50, fill=False, ec='#888', lw=1.0, ls='--'))
ax.annotate('60 um FDTD domain (PML)', (-d + 1, d - 2.6), fontsize=9)
ax.annotate('50 um source aperture', (-24, 23), fontsize=9, color='#666')
ax.annotate('', xytext=(22, -27), xy=(28, -27), arrowprops=dict(arrowstyle='->', color='k'))
ax.text(24.3, -26.2, 'b (E$\\parallel$b, lab x)', fontsize=10)
ax.annotate('', xytext=(-27, 20), xy=(-27, 26), arrowprops=dict(arrowstyle='->', color='k'))
ax.text(-26.4, 21.5, 'a (E$\\parallel$a, lab y)', rotation=90, fontsize=10)
ax.set_xlim(-31, 31); ax.set_ylim(-31, 31); ax.set_aspect('equal')
ax.set_xlabel('lab x = crystal b (um)'); ax.set_ylabel('lab y = crystal a (um)')
ax.set_title('Device A top view (digitized from paper Fig. 2)')
ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95)

layers = [
    ('Si substrate (n=3.425)', -3.415, -0.415, '#bbbbbb'),
    ('SiO$_2$ 285 nm - Palik lossy $\\epsilon$=3.73+0.19j', -0.415, -0.130, '#88ccee'),
    ('TaIrTe$_4$ 130 nm ($\\epsilon_a$=-43+205j, $\\epsilon_b$=13+26j)', -0.130, 0.0, '#e8a33d'),
    ('Ti 5 nm + Au 50 nm (electrode regions only)', 0.0, 0.055, '#ffd700'),
    ('air', 0.055, 5.2, '#f4f8fb'),
]
for name, z0, z1, c in layers:
    az.axhspan(z0, z1, color=c)
    az.text(0.03, (z0 + z1) / 2, name, fontsize=9, va='center', transform=az.get_yaxis_transform())
az.axhline(5.0, color='#228833', ls='--', lw=2)
az.text(0.03, 5.05, 'Gaussian source plane z=+5 um, lambda=11 um, normal incidence (-z)', fontsize=9, color='#228833', transform=az.get_yaxis_transform())
az.plot([0.5], [-0.065], 'x', color='#228833', ms=9, mew=2, transform=az.get_yaxis_transform())
az.text(0.53, -0.25, 'focus @ flake midplane', fontsize=8, color='#228833', transform=az.get_yaxis_transform())
az.set_ylim(-3.6, 5.6); az.set_xticks([]); az.set_ylabel('z (um)')
az.set_title('Vertical stack (symlog z)')
az.set_yscale('symlog', linthresh=0.5)
fig.tight_layout()
out = '/data/seunghyun/tairte4/sanity_v2_workspace/inverse_design/photothermal_pte/reports/paper_ir_device_a_sanity_v2/lossy_sio2_scenario/DEVICE_A_STRUCTURE_ANNOTATED.png'
fig.savefig(out, dpi=170)
print(out)
