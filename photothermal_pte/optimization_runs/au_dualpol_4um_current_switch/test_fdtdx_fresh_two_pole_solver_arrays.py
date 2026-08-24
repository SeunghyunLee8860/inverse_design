from __future__ import annotations

import copy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_material import (
    coefficient_endpoint_matrix,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_candidate_model_material import (
    candidate_material_data,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    material_stack_audit,
)


def _candidate_law(targets: dict[str, complex]) -> dict[str, object]:
    axes = {}
    for material_index, name in enumerate(("au", "a", "b", "c")):
        poles = []
        for pole_index in range(2):
            offset = 0.1 * material_index + 0.01 * pole_index
            poles.append(
                {
                    "kind": "Drude",
                    "omega_p_rad_s": 1.0e15 + 1.0e13 * material_index,
                    "gamma_rad_s": 1.0e13 + material_index * 10.0 + pole_index,
                    "c1": 1.5 + offset,
                    "c2": -0.5 - offset,
                    "c3": 0.01 + offset,
                }
            )
        axes[name] = {
            "pole_kind": "Drude",
            "candidate": {
                "found": True,
                "fit_gate_passed": True,
                "fit_relative_error": 1.0e-8,
                "poles": poles,
            },
        }
    law = {
        "version": "fdtdx-fresh-stable-two-pole-material-v1",
        "case_binding": {"realized_float32_cfl": {"time_step_s": 1.0e-18}},
        "material_binding": {
            "target_epsilon": {
                name: [value.real, value.imag] for name, value in targets.items()
            },
            "tairte4_crystal_to_solver_axis": {"b": "x", "a": "y", "c": "z"},
        },
        "material_axes": axes,
        "checks": {"synthetic_gate": True},
        "promotion": {"candidate_only": True, "optimizer_start_allowed": False},
    }
    encoded = json.dumps(
        law, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    law["material_law_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return law


class _FakePole:
    def __init__(self, damping: float, registry: dict[float, tuple[float, ...]]):
        self.coefficients = registry[damping]


class FdtdxFreshTwoPoleSolverArrayTest(unittest.TestCase):
    def test_candidate_normalization_checks_pinned_generated_coefficients(self) -> None:
        targets = {
            "au": 2.0 + 1.0j,
            "a": 3.0 + 1.1j,
            "b": 4.0 + 1.2j,
            "c": 4.0 + 1.2j,
        }
        law = _candidate_law(targets)
        registry = {
            float(pole["gamma_rad_s"]): (
                float(pole["c1"]),
                float(pole["c2"]),
                float(pole["c3"]),
            )
            for axis in law["material_axes"].values()
            for pole in axis["candidate"]["poles"]
        }
        fdtdx = ModuleType("fdtdx")
        fdtdx.__path__ = []
        fdtdx.DrudePole = lambda plasma_frequency, damping: _FakePole(
            damping, registry
        )
        dispersion = ModuleType("fdtdx.dispersion")

        def compute(poles, dt):
            del dt
            matrix = np.asarray([pole.coefficients for pole in poles])
            repeated = tuple(np.repeat(matrix[:, index, None], 3, axis=1) for index in range(3))
            return (*repeated, np.zeros((2, 3), dtype=np.float64))

        dispersion.compute_pole_coefficients_per_axis = compute
        with patch.dict(
            sys.modules, {"fdtdx": fdtdx, "fdtdx.dispersion": dispersion}
        ):
            result = candidate_material_data(
                fdtdx,
                law,
                dt_s=1.0e-18,
                omega_rad_s=2.0e15,
                epsilon_au=targets["au"],
                epsilon_ta={name: targets[name] for name in ("a", "b", "c")},
            )
            self.assertEqual(len(result["poles"]["au"]), 2)
            self.assertEqual(np.asarray(result["coefficient_endpoints"]["a"]).shape, (2, 3))

            tampered = copy.deepcopy(law)
            tampered["material_axes"]["au"]["candidate"]["poles"][0]["c3"] *= 2.0
            with self.assertRaises(RuntimeError):
                candidate_material_data(
                    fdtdx,
                    tampered,
                    dt_s=1.0e-18,
                    omega_rad_s=2.0e15,
                    epsilon_au=targets["au"],
                    epsilon_ta={name: targets[name] for name in ("a", "b", "c")},
                )

    def test_two_pole_material_stack_readback_is_exact(self) -> None:
        shape = (80, 80, 4)
        inverse = np.ones((3, *shape), dtype=np.float32)
        inverse[:, :, :, 0] = np.float32(1.0 / 12.0)
        inverse[:, :, :, 1] = np.float32(1.0 / 2.0)
        endpoints = {
            "au": ((0.11, -0.12, 0.013), (0.21, -0.22, 0.023)),
            "a": ((0.31, -0.32, 0.033), (0.41, -0.42, 0.043)),
            "b": ((0.51, -0.52, 0.053), (0.61, -0.62, 0.063)),
            "c": ((0.71, -0.72, 0.073), (0.81, -0.82, 0.083)),
        }
        coefficients = {
            name: np.zeros((2, 3, *shape), dtype=np.float32)
            for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3")
        }
        for coefficient_index, name in enumerate(coefficients):
            for pole_index in range(2):
                for component, axis in enumerate(("b", "a", "c")):
                    coefficients[name][pole_index, component, :, :, 2] = np.float32(
                        endpoints[axis][pole_index][coefficient_index]
                    )
                coefficients[name][pole_index, :, :, :, 3] = np.float32(
                    endpoints["au"][pole_index][coefficient_index]
                )
        arrays = SimpleNamespace(
            inv_permittivities=inverse,
            dispersive_c1=coefficients["dispersive_c1"],
            dispersive_c2=coefficients["dispersive_c2"],
            dispersive_c3=coefficients["dispersive_c3"],
            dispersive_c4=None,
        )
        epsilon_ta = {
            "a": 1.0 + 1.1j,
            "b": 1.0 + 1.2j,
            "c": 1.0 + 1.3j,
        }
        model = {
            "slices": {
                "fixed_silicon_substrate": (slice(0, 80), slice(0, 80), slice(0, 1)),
                "fixed_285nm_sio2": (slice(0, 80), slice(0, 80), slice(1, 2)),
                "fixed_tairte4": (slice(0, 80), slice(0, 80), slice(2, 3)),
                "au_design": (slice(0, 80), slice(0, 80), slice(3, 4)),
            },
            "coefficient_endpoints": endpoints,
            "coefficients": endpoints,
            "epsilon": {
                "silicon": 12.0 + 0.0j,
                "sio2": 2.0 + 0.0j,
                "tairte4": epsilon_ta,
                "au": 1.0 + 1.4j,
            },
            "discrete_susceptibility": {
                "a": epsilon_ta["a"] - 1.0,
                "b": epsilon_ta["b"] - 1.0,
                "c": epsilon_ta["c"] - 1.0,
                "au": 1.4j,
            },
            "absorption_loss_basis": "synthetic-two-pole-discrete-ADE",
            "fits": {name: {"fit_relative_error": 0.0} for name in endpoints},
        }
        self.assertEqual(coefficient_endpoint_matrix(model, "au").shape, (2, 3))
        audit = material_stack_audit(
            model, arrays, np.ones((80, 80), dtype=np.uint8)
        )
        self.assertTrue(audit["ready"], audit["failed_checks"])
        self.assertEqual(audit["exact_binary_au"]["num_dispersive_poles"], 2)
        readback = audit["tairte4_coefficient_readback"]["dispersive_c1"]
        self.assertTrue(
            np.allclose(readback[0]["expected_by_pole"], [0.51, 0.61])
        )

    def test_endpoint_matrix_rejects_wrong_shape_or_nonfinite(self) -> None:
        with self.assertRaises(RuntimeError):
            coefficient_endpoint_matrix(
                {"coefficient_endpoints": {"au": ((1.0, 2.0),)}}, "au"
            )
        with self.assertRaises(RuntimeError):
            coefficient_endpoint_matrix(
                {"coefficient_endpoints": {"au": ((1.0, 2.0, np.nan),)}},
                "au",
            )


if __name__ == "__main__":
    unittest.main()
