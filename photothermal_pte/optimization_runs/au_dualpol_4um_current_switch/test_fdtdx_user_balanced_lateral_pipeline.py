from __future__ import annotations

from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    reference_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    solver_mask,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_user_balanced_lateral_refinement import (
    material_spec,
)


HERE = Path(__file__).resolve().parent


def test_exact_l500_geometry_is_replicated_not_reinterpreted() -> None:
    mask = np.asarray(
        reference_mask("l_shape_4um_with_500nm_arms"), dtype=np.uint8
    )
    expanded = solver_mask(mask, material_spec())
    assert mask.shape == (80, 80)
    assert expanded.shape == (160, 160)
    assert np.array_equal(expanded, np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1))
    assert np.count_nonzero(expanded) == 4 * np.count_nonzero(mask)


def test_safe_wrappers_forward_lateral_cli_options() -> None:
    source_wrapper = (HERE / "run_fdtdx_user_balanced_z_source_gpu.sh").read_text()
    material_wrapper = (HERE / "run_fdtdx_user_balanced_material_gpu.sh").read_text()
    assert '"$@"' in source_wrapper
    assert '"$@"' in material_wrapper
    assert "--lateral-factor" in (HERE / "fdtdx_user_balanced_z_source_only.py").read_text()
    assert "--design-flake-xy-factor" in (
        HERE / "fdtdx_user_balanced_exact_binary.py"
    ).read_text()


def test_lateral_source_pair_generator_is_fail_closed() -> None:
    source = (HERE / "fdtdx_user_balanced_lateral_source_pair.py").read_text()
    assert "per_polarization_normalization_forbidden" not in source
    assert "build_pair(" in source
    assert '"one_pair_selects_production_mesh": False' in source
    assert "return 0 if payload[\"ready\"] else 2" in source
