"""Differentiable perimeter-contact optimization for TaIrTe4 PTE devices."""

from .perimeter import PerimeterDiscretization, PerimeterParameters
from .optimization import SignedSLSQPResult, run_signed_slsqp
from .robin import DifferentiableContactModel
from .scaled import ScaledDesign, SignedBranchObjective

__all__ = [
    "DifferentiableContactModel",
    "PerimeterDiscretization",
    "PerimeterParameters",
    "ScaledDesign",
    "SignedSLSQPResult",
    "SignedBranchObjective",
    "run_signed_slsqp",
]
