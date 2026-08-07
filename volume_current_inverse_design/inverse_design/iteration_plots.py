"""Per-iteration production visualisation (design, FOM, constraints, binarization).

Called by the runner after EVERY objective evaluation.  Writes to
``<run_root>/plots/``:

  design_it####_beta<b>.png : 3 panels -- nominal density rho (240x240 unique
                              torus), exact-binary preview (rho>=0.5), and the
                              rho histogram (binarization at a glance)
  progress.png              : rolling dashboard -- Fx/Fy/F_sum (log),
                              g_solid/g_void vs 0, binarization/rail
                              fractions, latent step size; beta-stage
                              boundaries marked

Plotting must NEVER kill a multi-hour production run: the runner wraps the
call in try/except, and this module only needs numpy + matplotlib (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def density_metrics(rho_unique: np.ndarray) -> dict:
    """Binarization statistics of a [0,1] density field (JSON-friendly)."""
    u = np.asarray(rho_unique, float)
    grayness = float(np.mean(4.0 * u * (1.0 - u)))
    return {
        "solid_fraction": float(np.mean(u)),
        "grayness": grayness,
        "binarization": float(1.0 - grayness),
        "frac_below_0.01": float(np.mean(u < 0.01)),
        "frac_above_0.99": float(np.mean(u > 0.99)),
        "frac_gray_band": float(np.mean((u >= 0.3) & (u <= 0.7))),
    }


def save_design_snapshot(plots_dir: Path, rho_unique: np.ndarray,
                         iteration: int, beta: float, record: dict) -> Path:
    u = np.asarray(rho_unique, float)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    im = axes[0].imshow(u.T, origin="lower", cmap="viridis", vmin=0, vmax=1,
                        extent=[0, 6, 0, 6])
    axes[0].set_title(f"rho nominal (iter {iteration}, beta={beta:g})")
    axes[0].set_xlabel("x [um]"); axes[0].set_ylabel("y [um]")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].imshow((u >= 0.5).T, origin="lower", cmap="gray_r",
                   vmin=0, vmax=1, extent=[0, 6, 0, 6])
    axes[1].set_title("exact-binary preview (rho>=0.5)")
    axes[1].set_xlabel("x [um]")

    axes[2].hist(u.reshape(-1), bins=60, range=(0, 1), color="#4477aa")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("rho"); axes[2].set_title(
        f"binarization={record.get('binarization', float('nan')):.3f}  "
        f"solid={record.get('solid_fraction', float('nan')):.3f}")

    fx = record.get("Fx"); fy = record.get("Fy"); ftot = record.get("objective")
    fig.suptitle(
        f"Fx={fx:.3e}  Fy={fy:.3e}  F_sum={ftot:.3e}   "
        f"g_solid={record.get('g_solid', float('nan')):.3e}  "
        f"g_void={record.get('g_void', float('nan')):.3e}  "
        f"feasible={record.get('constraint_feasible')}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plots_dir / f"design_it{iteration:04d}_beta{beta:g}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_progress_dashboard(plots_dir: Path, records: list) -> Path:
    recs = [r for r in records if "iter" in r]
    iters = [r["iter"] for r in recs]
    betas = [r["beta"] for r in recs]
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11.5), sharex=True)

    axes[0].semilogy(iters, [max(r["Fx"], 1e-300) for r in recs], "o-",
                     label="Fx", ms=3)
    axes[0].semilogy(iters, [max(r["Fy"], 1e-300) for r in recs], "s-",
                     label="Fy", ms=3)
    axes[0].semilogy(iters, [max(r["objective"], 1e-300) for r in recs],
                     "k^-", label="F_sum", ms=3)
    axes[0].set_ylabel("FOM"); axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(iters, [r["g_solid"] for r in recs], "o-",
                 label="g_solid", ms=3)
    axes[1].plot(iters, [r["g_void"] for r in recs], "s-",
                 label="g_void", ms=3)
    axes[1].axhline(0.0, color="k", lw=0.8)
    for r in recs:
        if r.get("constraint_feasible"):
            axes[1].axvspan(r["iter"] - 0.5, r["iter"] + 0.5,
                            color="green", alpha=0.08, lw=0)
    axes[1].set_ylabel("constraint residual (<=0 feasible)")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(iters, [r.get("binarization", np.nan) for r in recs],
                 "o-", ms=3, label="binarization 1-mean(4u(1-u))")
    axes[2].plot(iters, [r.get("frac_rails", np.nan) for r in recs],
                 "s-", ms=3, label="fraction at rails (<0.01 | >0.99)")
    axes[2].set_ylim(0, 1.02); axes[2].set_ylabel("binarization")
    axes[2].legend(loc="best", fontsize=8)

    axes[3].semilogy(
        iters, [max(r.get("latent_step_rms", np.nan), 1e-300) for r in recs],
        "o-", ms=3, label="latent step RMS")
    axes[3].set_ylabel("latent change"); axes[3].set_xlabel("iteration")
    axes[3].legend(loc="best", fontsize=8)

    # beta stage boundaries
    for k in range(1, len(recs)):
        if betas[k] != betas[k - 1]:
            for ax in axes:
                ax.axvline(iters[k] - 0.5, color="purple", ls="--", lw=0.9)
            axes[0].text(iters[k], axes[0].get_ylim()[1],
                         f" b={betas[k]:g}", color="purple", fontsize=8,
                         va="top")
    if recs:
        axes[0].set_title(
            f"iter {iters[-1]}  beta={betas[-1]:g}  "
            f"F_sum={recs[-1]['objective']:.3e}  "
            f"feasible={recs[-1].get('constraint_feasible')}")
    fig.tight_layout()
    path = plots_dir / "progress.png"
    tmp = plots_dir / "progress.tmp.png"
    fig.savefig(tmp, dpi=130)
    plt.close(fig)
    tmp.replace(path)
    return path


def save_iteration_plots(run_root: Path, rho_unique: np.ndarray,
                         iteration: int, beta: float, records: list) -> list:
    """Write the per-iteration snapshot + rolling dashboard; returns paths."""
    plots_dir = Path(run_root) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    latest = records[-1] if records else {}
    paths = [save_design_snapshot(plots_dir, rho_unique, iteration, beta,
                                  latest)]
    paths.append(save_progress_dashboard(plots_dir, records))
    return paths
