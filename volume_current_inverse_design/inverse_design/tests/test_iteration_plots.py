"""FDTD-free tests for the per-iteration visualisation module.

The plots are diagnostics: they must (a) report binarization honestly and
(b) never require anything beyond numpy+matplotlib, so a plotting failure can
never take down a multi-hour production run (the runner additionally wraps the
call in try/except).
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from iteration_plots import (  # noqa: E402
    density_metrics,
    save_design_snapshot,
    save_iteration_plots,
    save_progress_dashboard,
)


def _record(i, beta, **over):
    rec = {"iter": i, "beta": beta, "Fx": 1e-6 * i, "Fy": 2e-6 * i,
           "objective": 3e-6 * i, "g_solid": -1e-6, "g_void": 1e-4,
           "constraint_feasible": False, "latent_step_rms": 1e-3,
           "binarization": 0.5, "frac_rails": 0.3, "solid_fraction": 0.4}
    rec.update(over)
    return rec


def test_density_metrics_binarization_bounds():
    gray = density_metrics(np.full((8, 8), 0.5))
    assert gray["binarization"] == pytest.approx(0.0)
    assert gray["grayness"] == pytest.approx(1.0)
    binary = density_metrics(np.random.default_rng(0).integers(0, 2, (8, 8)).astype(float))
    assert binary["binarization"] == pytest.approx(1.0)
    assert binary["frac_below_0.01"] + binary["frac_above_0.99"] == pytest.approx(1.0)


def test_save_iteration_plots_writes_snapshot_and_dashboard(tmp_path):
    rho = np.clip(np.random.default_rng(1).normal(0.5, 0.3, (24, 24)), 0, 1)
    records = [_record(1, 2.0), _record(2, 2.0), _record(3, 4.0)]
    paths = save_iteration_plots(tmp_path, rho, 3, 4.0, records)
    assert all(p.exists() for p in paths)
    assert (tmp_path / "plots" / "progress.png").exists()
    assert (tmp_path / "plots" / "design_it0003_beta4.png").exists()


def test_dashboard_tolerates_missing_optional_fields(tmp_path):
    # early records (or old history files) may lack the newer metric fields
    plots = tmp_path / "plots"
    plots.mkdir()
    records = [{"iter": 1, "beta": 2.0, "Fx": 1e-9, "Fy": 1e-9,
                "objective": 2e-9, "g_solid": 0.1, "g_void": 0.1,
                "constraint_feasible": False}]
    assert save_progress_dashboard(plots, records).exists()


def test_snapshot_records_gray_metrics_in_title(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    rho = np.full((16, 16), 0.5)
    path = save_design_snapshot(
        plots, rho, 7, 8.0, _record(7, 8.0, **density_metrics(rho)))
    assert path.exists()
