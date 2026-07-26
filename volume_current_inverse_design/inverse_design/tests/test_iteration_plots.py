"""Per-iteration plot generation must work headless and never need Lumerical."""

import numpy as np

from iteration_plots import (
    density_metrics,
    save_iteration_plots,
)


def _records(n=5):
    rng = np.random.default_rng(0)
    records = []
    for i in range(1, n + 1):
        u = np.clip(0.5 + 0.2 * rng.standard_normal((24, 24)), 0, 1)
        rec = {
            "iter": i, "beta": 2.0 if i < 4 else 4.0,
            "Fx": 1e-6 * i, "Fy": 5e-7 * i, "objective": 1.5e-6 * i,
            "g_solid": 0.02 - 0.005 * i, "g_void": 0.02 - 0.004 * i,
            "constraint_feasible": i >= 4,
            "latent_step_rms": 1e-3 / i,
            **density_metrics(u),
        }
        rec["frac_rails"] = rec["frac_below_0.01"] + rec["frac_above_0.99"]
        records.append(rec)
    return records


def test_density_metrics_binary_and_gray():
    binary = np.zeros((10, 10)); binary[:5] = 1.0
    m = density_metrics(binary)
    assert m["binarization"] == 1.0 and m["grayness"] == 0.0
    assert m["solid_fraction"] == 0.5
    gray = np.full((10, 10), 0.5)
    m2 = density_metrics(gray)
    assert m2["binarization"] == 0.0 and m2["grayness"] == 1.0


def test_save_iteration_plots_writes_files(tmp_path):
    records = _records()
    u = np.clip(0.5 + 0.2 * np.random.default_rng(1).standard_normal((24, 24)),
                0, 1)
    paths = save_iteration_plots(tmp_path, u, iteration=5, beta=4.0,
                                 records=records)
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 5000, p
    assert (tmp_path / "plots" / "design_it0005_beta4.png").exists()
    assert (tmp_path / "plots" / "progress.png").exists()


def test_dashboard_survives_single_record(tmp_path):
    records = _records(1)
    u = np.full((24, 24), 0.5)
    paths = save_iteration_plots(tmp_path, u, iteration=1, beta=2.0,
                                 records=records)
    assert all(p.exists() for p in paths)


def test_runner_wires_plots_and_metrics():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "run_constrained_inverse_design.py").read_text()
    assert "save_iteration_plots(" in src
    assert "density_metrics(" in src
    # plotting failures must be swallowed, not crash a production run
    assert "[plots] skipped" in src
    # runtime config must be part of the hashed contract
    for key in ('"adjoint_component_mode"', '"sim_time_s"',
                '"auto_shutoff_min"', '"bulk_mesh_mode"'):
        position = src.find(key)
        hash_position = src.find('contract["config_hash"] = _config_hash(contract)')
        assert 0 < position < hash_position, key
