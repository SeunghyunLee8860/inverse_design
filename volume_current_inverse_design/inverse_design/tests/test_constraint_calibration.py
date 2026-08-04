"""Constraint-vs-DRC boundary calibration (review #6).

Pins the CURRENT behaviour at the 425/475/525 nm boundary WITHOUT tuning tol/p
(that must be calibrated against a Lumerical run, not guessed):

  * DRC decision is deterministic and authoritative -> asserted strictly.
  * the optimisation constraint is a steering helper -> its decision is recorded
    and pinned so any future change is visible.

Known gap (documented, not silently accepted): a clean 19-cell (475 nm) bar is
already below the Zhou penalty floor, so `g_solid <= 0` (constraint-feasible)
even though DRC fails it.  Both 475 and 525 nm sit at the penalty floor, so
lowering tol or raising p cannot separate them without a wider filter / a
different formulation -- calibration is deferred to a real run.  The final DRC
gate blocks the sub-target design regardless, so no false SUCCESS is possible.
Runtime knobs recorded for that calibration: VC_TOL_SOLID, VC_TOL_VOID,
VC_CONSTRAINT_PNORM (see IMPLEMENTATION_STATUS.md).
"""

import numpy as np

from periodic_constrained_mapping import PeriodicConstrainedMapping
from geometric_constraints import LengthScaleConstraints
from geometry_drc import geometry_drc

BETA = 64.0
H = 0.025


def _bar(w, at=110):
    lat = np.zeros((240, 240))
    lat[:, at:at + w] = 1.0
    return lat.reshape(-1)


def _case(con, m, w):
    lat = _bar(w)
    g_solid = float(con.solid_penalty(lat, BETA)) - con.tol_solid
    mask = (np.asarray(m.filter_unique(lat)) >= 0.5).astype(np.uint8)
    drc = geometry_drc(mask, H, 0.5, 0.5)
    return g_solid, drc["pass"]


def test_drc_boundary_is_authoritative_and_deterministic():
    m = PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    con = LengthScaleConstraints(m)
    g425, drc425 = _case(con, m, 17)   # 425 nm
    g475, drc475 = _case(con, m, 19)   # 475 nm
    g525, drc525 = _case(con, m, 21)   # 525 nm

    # DRC: authoritative, must be exactly this
    assert drc425 is False
    assert drc475 is False            # 500 nm rule is conservative -> needs >= 525 nm
    assert drc525 is True

    # constraint: current pinned behaviour (helper only; DRC gates the rest)
    assert g425 > 0.0                 # 425 nm infeasible (penalised)
    assert g475 <= 0.0                # 475 nm at penalty floor -> constraint-feasible (known gap)
    assert g525 <= 0.0
    # penalty must be monotone non-increasing with width
    assert g425 > g475 >= g525


def test_constraint_knobs_exposed():
    # the calibration knobs must remain env-overridable for a real run
    import os
    m = PeriodicConstrainedMapping(241, 241, 13, period_um=6.0, filter_radius_um=0.5)
    os.environ["VC_TOL_SOLID"] = "2e-3"
    con = LengthScaleConstraints(m)
    assert con.tol_solid == 2e-3
    del os.environ["VC_TOL_SOLID"]
    con2 = LengthScaleConstraints(m, tol_solid=5e-4)
    assert con2.tol_solid == 5e-4
