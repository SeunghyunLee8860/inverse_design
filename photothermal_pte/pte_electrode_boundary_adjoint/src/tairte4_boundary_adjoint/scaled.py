from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perimeter import PerimeterParameters
from .robin import DifferentiableContactModel, RobinEvaluation


@dataclass(frozen=True)
class ScaledDesign:
    """Dimensionless optimizer coordinates x=(u0,l0,u1,l1).

    Centers u0/u1 live on the real line (the universal cover of the periodic
    perimeter) and are never box-bounded.  Lengths are fractions of P.
    """

    center_0_lifted: float
    length_0_fraction: float
    center_1_lifted: float
    length_1_fraction: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.center_0_lifted,
                self.length_0_fraction,
                self.center_1_lifted,
                self.length_1_fraction,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "ScaledDesign":
        x = np.asarray(values, dtype=float)
        if x.shape != (4,):
            raise ValueError("scaled design must have shape (4,)")
        return cls(*(float(value) for value in x))

    def to_physical(self, perimeter_m: float) -> PerimeterParameters:
        x = self.as_array()
        return PerimeterParameters.from_array(perimeter_m * x)

    def canonical(self) -> "ScaledDesign":
        """Wrap centers only for output, clustering, and hard validation."""
        return ScaledDesign(
            self.center_0_lifted % 1.0,
            self.length_0_fraction,
            self.center_1_lifted % 1.0,
            self.length_1_fraction,
        )


@dataclass(frozen=True)
class SignedBranchEvaluation:
    branch_sign: int
    scaled_design: ScaledDesign
    canonical_design: ScaledDesign
    physical_parameters: PerimeterParameters
    current_A: float
    current_scale_A: float
    signed_response: float
    minimization_objective: float
    minimization_gradient_scaled: np.ndarray
    forward: RobinEvaluation


class SignedBranchObjective:
    """Dimensionless +I/-I branch objective for a single fixed beam.

    SciPy minimizes ``-branch_sign * I / I_ref``.  Running both signs and then
    comparing hard-contact ``abs(I)`` replaces production optimization of I^2.
    """

    def __init__(
        self,
        model: DifferentiableContactModel,
        *,
        current_scale_A: float | None = None,
    ):
        self.model = model
        self.perimeter_m = model.perimeter.perimeter_m
        default_scale = float(np.linalg.norm(model.q_A, ord=1))
        self.current_scale_A = (
            default_scale if current_scale_A is None else float(current_scale_A)
        )
        if not np.isfinite(self.current_scale_A) or self.current_scale_A <= 0.0:
            raise ValueError(
                "current_scale_A must be positive; a zero-gradient beam has no "
                "electrode optimization objective"
            )

    def evaluate(
        self, values: np.ndarray, *, branch_sign: int
    ) -> SignedBranchEvaluation:
        if branch_sign not in (-1, +1):
            raise ValueError("branch_sign must be +1 or -1")
        design = ScaledDesign.from_array(values)
        physical = design.to_physical(self.perimeter_m)
        forward = self.model.evaluate(physical)
        signed_response = branch_sign * forward.current_A / self.current_scale_A
        # dp/dx=P for all four coordinates because every physical coordinate
        # was scaled by the same perimeter length.
        gradient = (
            -branch_sign
            * self.perimeter_m
            * forward.current_gradient_A_per_m
            / self.current_scale_A
        )
        return SignedBranchEvaluation(
            branch_sign=branch_sign,
            scaled_design=design,
            canonical_design=design.canonical(),
            physical_parameters=physical,
            current_A=forward.current_A,
            current_scale_A=self.current_scale_A,
            signed_response=signed_response,
            minimization_objective=-signed_response,
            minimization_gradient_scaled=gradient,
            forward=forward,
        )

    def length_bounds(
        self, minimum_length_m: float, maximum_length_m: float
    ) -> tuple[tuple[float | None, float | None], ...]:
        if not 0.0 < minimum_length_m <= maximum_length_m < self.perimeter_m:
            raise ValueError("invalid physical contact-length bounds")
        lo = minimum_length_m / self.perimeter_m
        hi = maximum_length_m / self.perimeter_m
        # Lifted centers are unbounded: no artificial 0/P box seam.
        return ((None, None), (lo, hi), (None, None), (lo, hi))
