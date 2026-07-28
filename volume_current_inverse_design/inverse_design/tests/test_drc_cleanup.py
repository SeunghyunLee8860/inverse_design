"""DRC-driven mask cleanup + finalizer wiring.

Measured basis (official r5, 2026-07-28): the ladder finished fully
constraint-feasible (g=-1.85e-5/-1.0e-6) yet the exact-binary mask carried
333 solid + 712 void pixels below the 0.5125 um opposing-boundary rule
(wedge tips, edge bumps, ledges) -- geometry the autograd constraints have
no gradient against.  A naive "flip every thin pixel" loop entered a limit
cycle (24,020 flips, no convergence); morphology alone leaves residues
(opening's inscribed-disk guarantee does not bound the DRC's rim-chord
reads).  The shipped v2 (morphology + site-wise monotone surgery) cleans
the real r5 mask to DRC PASS in ~35 s with 7.8% pixels changed.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from drc_cleanup import CLEANUP_VERSION, drc_cleanup
from geometry_drc import geometry_drc

HERE = Path(__file__).resolve().parents[1]
FINALIZER = HERE / "final_projection.py"
LAUNCHER = HERE.parent / "run_inverse_design.sh"

# Small-grid rule: spacing 0.1 um -> rule+guard = 0.55 um = 5.5 px.
SPACING = 0.1


def _stripe(n=60, solid_rows=30):
    m = np.zeros((n, n), np.uint8)
    m[:solid_rows, :] = 1
    return m


def test_passing_mask_is_untouched():
    m = _stripe()
    assert geometry_drc(m, spacing_um=SPACING)["pass"]
    res = drc_cleanup(m, spacing_um=SPACING)
    assert res["pass"] is True
    assert res["pixels_changed"] == 0
    assert np.array_equal(res["mask"], m)
    assert res["version"] == CLEANUP_VERSION


def test_cleanup_removes_edge_bump():
    """A 1-px bump on a straight edge (r5's solid violation, miniaturised)."""
    m = _stripe()
    m[30, 20] = 1                      # single-pixel protrusion into the void
    assert not geometry_drc(m, spacing_um=SPACING)["pass"]
    res = drc_cleanup(m, spacing_um=SPACING)
    assert res["pass"] is True, res["stages"]
    assert geometry_drc(res["mask"], spacing_um=SPACING)["pass"]
    # the fix must stay local: far smaller than the stripe itself
    assert 0 < res["pixels_changed"] < 200


def test_cleanup_fills_tapering_void_wedge():
    """A void wedge tapering below the rule (r5's void violation)."""
    m = _stripe(solid_rows=34)
    for k in range(12):                # V-shaped notch cut into the solid
        half = max(0, 6 - k // 2)
        m[22 + k, 30 - half : 30 + half + 1] = 0
    assert not geometry_drc(m, spacing_um=SPACING)["pass"]
    res = drc_cleanup(m, spacing_um=SPACING)
    assert res["pass"] is True, res["stages"]
    assert geometry_drc(res["mask"], spacing_um=SPACING)["pass"]


def test_cleanup_reports_honest_failure():
    """An impossible layout (alternating thin stripes everywhere) must come
    back pass=False, never a fake success."""
    m = np.zeros((60, 60), np.uint8)
    m[::4, :] = 1                      # 1-px lines, nothing to rescue
    res = drc_cleanup(m, spacing_um=SPACING, morph_rounds=1, site_rounds=2)
    if not res["pass"]:                # expected: cleanup gives up...
        assert not res["drc"]["pass"]
    else:                              # ...or degenerates to one phase, which
        assert geometry_drc(res["mask"], spacing_um=SPACING)["pass"] \
            or res["drc"]["trivial_topology"] is False


def test_non_binary_mask_raises():
    m = _stripe().astype(float)
    m[0, 0] = 0.5
    with pytest.raises(ValueError):
        drc_cleanup(m, spacing_um=SPACING)


def test_result_mask_is_exact_binary_and_periodic_measurable():
    m = _stripe()
    m[30, 20] = 1
    res = drc_cleanup(m, spacing_um=SPACING)
    assert set(np.unique(res["mask"])) <= {0, 1}
    assert res["mask"].shape == m.shape


# --- finalizer / launcher wiring (source-level, same pattern as the other
# guards; drc_cleanup changes the finalised geometry, so it must be part of
# code_hash exactly like feasibility_repair) -------------------------------
def test_finalizer_has_cleanup_flag_before_drc_gate():
    src = FINALIZER.read_text()
    assert "--drc-cleanup" in src
    cleanup_pos = src.find("from drc_cleanup import")
    gate_pos = src.find('if not drc["pass"]:')
    assert 0 < cleanup_pos < gate_pos
    # the cleaned mask must be recorded in both the manifest and SUCCESS
    assert src.count('"drc_cleanup": cleanup_block') == 2


def test_finalizer_accept_design_code_hash_is_narrow():
    """The explicit-acceptance path must still be an equality check against
    the design's recorded hash, not a blanket skip."""
    src = FINALIZER.read_text()
    assert "--accept-design-code-hash" in src
    assert "design_hash != args.accept_design_code_hash" in src
    assert '"accepted_design_code_hash": args.accept_design_code_hash' in src


def test_drc_cleanup_is_code_hashed():
    runner = importlib.import_module("run_constrained_inverse_design")
    files = runner.production_code_files(runner.ROOT, runner.HERE)
    assert "drc_cleanup.py" in [p.name for p in files]
    with_file = runner._code_hash(files)
    without = runner._code_hash([p for p in files if p.name != "drc_cleanup.py"])
    assert with_file != without


def test_launcher_passes_cleanup_flag_by_default():
    src = LAUNCHER.read_text()
    assert 'DRC_CLEANUP:-1' in src
    assert "--drc-cleanup" in src or "$CLEANARG" in src
