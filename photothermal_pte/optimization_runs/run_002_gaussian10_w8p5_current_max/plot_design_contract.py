#!/usr/bin/env python3
"""Plot the reviewed Run-002 optical/design/thermal geometry contract."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


HERE = Path(__file__).resolve().parent


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), constrained_layout=True)
    xy, xz = axes

    # Optical xy contract.
    xy.add_patch(Rectangle((-24, -24), 48, 48, color="#dceaf7", zorder=0))
    xy.add_patch(
        Rectangle((-24, -24), 48, 48, fill=False, lw=4, ls=":", ec="#6a3d9a")
    )
    xy.add_patch(
        Rectangle((-20, -20), 40, 40, fill=False, lw=2, ec="#1f78b4", label="40 µm source")
    )
    xy.add_patch(
        Rectangle((-10, -10), 20, 20, fill=False, lw=2, ls="--", ec="#333333", label="20 µm coarse gradient canvas")
    )
    candidates = [
        (-6, 1, 12, 6),
        (-6, -7, 12, 6),
        (1, -6, 6, 12),
        (-7, -6, 6, 12),
    ]
    for index, bounds in enumerate(candidates):
        xy.add_patch(
            Rectangle(
                bounds[:2], bounds[2], bounds[3],
                facecolor="#ff7f00", edgecolor="#b35806", alpha=0.16,
                label="candidate finite design strips" if index == 0 else None,
            )
        )
    xy.add_patch(
        Circle((0, 0), 8.5, fill=False, lw=2.2, ec="#e31a1c", label=r"target $w_0=8.5$ µm")
    )
    xy.plot(0, 0, "+", ms=14, mew=2.3, color="#e31a1c")
    xy.set_xlim(-26, 26)
    xy.set_ylim(-26, 26)
    xy.set_aspect("equal")
    xy.set_xlabel("x=b (µm)")
    xy.set_ylabel("y=a (µm)")
    xy.set_title("A. Optical xy and finite design-window candidates")
    xy.text(-23, 22, "TaIrTe₄ optical background extends through transverse PML", fontsize=9)
    xy.legend(loc="lower right", fontsize=8.5)

    # x-z contract; vertical dimensions intentionally expanded for legibility.
    xz.add_patch(Rectangle((-24, -8), 48, 16, color="#edf4fa"))
    xz.add_patch(Rectangle((-24, -8), 48, 16, fill=False, lw=4, ls=":", ec="#6a3d9a"))
    xz.add_patch(Rectangle((-24, -0.05), 48, 0.10, color="#ef6a62", label="TaIrTe₄ 100 nm"))
    xz.add_patch(Rectangle((-24, -0.335), 48, 0.285, color="#76c7cf", label="bottom SiO₂ 285 nm"))
    xz.add_patch(Rectangle((-24, -5.5), 48, 5.165, color="#516a8a", label="Si to lower optical PML"))
    xz.add_patch(Rectangle((-6, 0.05), 12, 1.0, color="#f6c34a", alpha=0.9, label="candidate design, 1.0 µm"))
    xz.annotate(
        "scalar Gaussian source\nz=+5 µm, propagation −z",
        xy=(0, 5), xytext=(-18, 5.8),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#1f78b4"},
        color="#1f78b4",
    )
    xz.axhline(0, lw=1, ls="--", color="black", alpha=0.5)
    xz.set_xlim(-26, 26)
    xz.set_ylim(-6.2, 7.3)
    xz.set_xlabel("x=b (µm)")
    xz.set_ylabel("z (µm; thin layers expanded visually)")
    xz.set_title("B. Optical xz stack and source position")
    xz.legend(loc="lower right", fontsize=8.5)
    xz.text(
        -23.5, -5.95,
        "Thermal model: finite 32 µm flake in a 64 µm lateral domain; 20 µm Si-depth audit",
        color="white", fontsize=8.5,
    )

    fig.suptitle(
        "Run 002 reviewed geometry — λ=10 µm scalar Gaussian; no periodic boundary",
        fontsize=15,
    )
    output = HERE / "plots" / "design_domain_contract.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
