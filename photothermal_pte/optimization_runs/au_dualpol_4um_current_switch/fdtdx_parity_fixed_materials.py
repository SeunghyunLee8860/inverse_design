"""Frozen non-design material carrier coefficients on the parity time step."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ade import (
    carrier_dt_s,
    carrier_omega_rad_s,
)


HERE = Path(__file__).resolve().parent
MATERIAL_CONTRACT = HERE / "results_materials_4um" / "4um_material_contract.json"
RELATIVE_EPSILON_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class FixedLorentzCarrier:
    name: str
    target_epsilon_real: float
    target_epsilon_imag: float
    c1: float
    c2: float
    c3: float

    @property
    def target_epsilon(self) -> complex:
        return complex(self.target_epsilon_real, self.target_epsilon_imag)


TA_A = FixedLorentzCarrier(
    name="TaIrTe4_a",
    target_epsilon_real=-30.713256371885343,
    target_epsilon_imag=50.848086107787424,
    c1=1.998538851737976,
    c2=-0.9985389113426208,
    c3=0.00010124654363607988,
)
TA_B = FixedLorentzCarrier(
    name="TaIrTe4_b_and_c_closure",
    target_epsilon_real=15.900726644538812,
    target_epsilon_imag=9.289194887622557,
    c1=1.9981403350830078,
    c2=-0.9981442093849182,
    c3=6.0432641475927085e-05,
)


def load_material_targets() -> dict[str, object]:
    payload = json.loads(MATERIAL_CONTRACT.read_text(encoding="utf-8"))
    if payload.get("status") != "VALIDATED_4UM_SINGLE_FREQUENCY_MATERIAL_READBACK":
        raise RuntimeError("4-um material readback is not validated")
    if float(payload["wavelength_m"]) != 4.0e-6:
        raise RuntimeError("material readback wavelength changed")
    return payload


def realized_susceptibility(carrier: FixedLorentzCarrier) -> complex:
    theta = np.float32(carrier_omega_rad_s() * carrier_dt_s())
    z_minus = np.exp(np.complex64(-1j * theta))
    z_plus = np.exp(np.complex64(1j * theta))
    denominator = (
        z_minus
        - np.float32(carrier.c1)
        - np.float32(carrier.c2) * z_plus
    )
    return complex(np.complex64(np.float32(carrier.c3)) / denominator)


def realized_epsilon(carrier: FixedLorentzCarrier) -> complex:
    return complex(np.complex64(1.0) + np.complex64(realized_susceptibility(carrier)))


def recurrence_roots(carrier: FixedLorentzCarrier) -> np.ndarray:
    return np.roots(
        [
            1.0,
            -float(np.float32(carrier.c1)),
            -float(np.float32(carrier.c2)),
        ]
    )


def lorentz_parameters(carrier: FixedLorentzCarrier) -> dict[str, float]:
    """Invert the realized float32 recurrence to positive parameters."""

    dt = carrier_dt_s()
    c1 = float(np.float32(carrier.c1))
    c2 = float(np.float32(carrier.c2))
    c3 = float(np.float32(carrier.c3))
    denominator = 2.0 / (1.0 - c2)
    gamma_dt = 2.0 * (1.0 + c2) / (1.0 - c2)
    omega0_sq_dt2 = 2.0 - c1 * denominator
    if gamma_dt <= 0.0 or omega0_sq_dt2 <= 0.0 or c3 <= 0.0:
        raise RuntimeError(f"non-passive fixed recurrence for {carrier.name}")
    gamma = gamma_dt / dt
    omega0 = math.sqrt(omega0_sq_dt2) / dt
    coupling_sq = c3 * denominator / dt**2
    return {
        "gamma_rad_s": gamma,
        "omega0_rad_s": omega0,
        "coupling_sq_rad2_s2": coupling_sq,
        "delta_epsilon": coupling_sq / omega0**2,
    }


def coefficient_hash() -> str:
    values = np.asarray(
        [TA_A.c1, TA_A.c2, TA_A.c3, TA_B.c1, TA_B.c2, TA_B.c3],
        dtype="<f4",
    )
    digest = hashlib.sha256()
    digest.update(np.asarray([carrier_dt_s()], dtype="<f8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def fdtdx_api_audit() -> dict[str, object]:
    from fdtdx.dispersion import LorentzPole, compute_pole_coefficients

    result: dict[str, object] = {}
    passed = True
    for carrier in (TA_A, TA_B):
        params = lorentz_parameters(carrier)
        arrays = compute_pole_coefficients(
            (
                LorentzPole(
                    resonance_frequency=params["omega0_rad_s"],
                    damping=params["gamma_rad_s"],
                    delta_epsilon=params["delta_epsilon"],
                ),
            ),
            carrier_dt_s(),
        )
        observed = tuple(float(np.float32(array[0])) for array in arrays)
        expected = (
            float(np.float32(carrier.c1)),
            float(np.float32(carrier.c2)),
            float(np.float32(carrier.c3)),
            0.0,
        )
        exact = observed == expected
        passed = passed and exact
        result[carrier.name] = {
            "expected_float32_coefficients": list(expected),
            "fdtdx_float32_coefficients": list(observed),
            "exact": exact,
        }
    return {"status": "PASS" if passed else "FAIL", "materials": result}


def fixed_material_audit() -> dict[str, object]:
    targets = load_material_targets()
    material_json = targets["materials"]
    expected_a = material_json["TaIrTe4"]["a"]["epsilon"]
    expected_b = material_json["TaIrTe4"]["b"]["epsilon"]
    expected_c = material_json["TaIrTe4"]["c"]["epsilon"]
    if TA_A.target_epsilon != complex(expected_a["real"], expected_a["imag"]):
        raise RuntimeError("frozen TaIrTe4-a target differs from material JSON")
    if TA_B.target_epsilon != complex(expected_b["real"], expected_b["imag"]):
        raise RuntimeError("frozen TaIrTe4-b target differs from material JSON")
    if expected_b != expected_c:
        raise RuntimeError("TaIrTe4 c=b closure changed")

    rows: dict[str, object] = {}
    passed = True
    for carrier in (TA_A, TA_B):
        realized = realized_epsilon(carrier)
        error = abs(realized - carrier.target_epsilon) / abs(carrier.target_epsilon)
        roots = recurrence_roots(carrier)
        params = lorentz_parameters(carrier)
        checks = {
            "relative_error_below_1e_5": error < RELATIVE_EPSILON_TOLERANCE,
            "passive_realized_epsilon": realized.imag > 0.0,
            "positive_lorentz_parameters": all(value > 0.0 for value in params.values()),
            "strict_recurrence_stability": bool(np.max(np.abs(roots)) < 1.0),
            "strict_jury_margin": (
                1.0
                - float(np.float32(carrier.c2))
                - abs(float(np.float32(carrier.c1)))
                > 0.0
            ),
        }
        passed = passed and all(checks.values())
        rows[carrier.name] = {
            "checks": checks,
            "target_epsilon": [carrier.target_epsilon.real, carrier.target_epsilon.imag],
            "realized_epsilon": [realized.real, realized.imag],
            "relative_complex_epsilon_error": error,
            "roots": [[float(root.real), float(root.imag)] for root in roots],
            "maximum_root_magnitude": float(np.max(np.abs(roots))),
            "lorentz_parameters": params,
            "coefficients": asdict(carrier),
        }

    sio2 = material_json["SiO2"]["epsilon"]
    silicon = material_json["Si"]["epsilon"]
    substrate_checks = {
        "SiO2_loss_negligible": abs(float(sio2["imag"])) < 1.0e-40,
        "Si_lossless_readback": float(silicon["imag"]) == 0.0,
        "SiO2_positive_real": float(sio2["real"]) > 0.0,
        "Si_positive_real": float(silicon["real"]) > 0.0,
    }
    passed = passed and all(substrate_checks.values())
    api = fdtdx_api_audit()
    passed = passed and api["status"] == "PASS"
    return {
        "status": "PASS" if passed else "FAIL",
        "axis_order_solver_xyz": ["b", "a", "c_equals_b"],
        "dt_s": carrier_dt_s(),
        "coefficient_sha256": coefficient_hash(),
        "tairte4": rows,
        "substrates": {
            "checks": substrate_checks,
            "SiO2_epsilon": [float(sio2["real"]), float(sio2["imag"])],
            "Si_epsilon": [float(silicon["real"]), float(silicon["imag"])],
        },
        "fdtdx_api": api,
    }


def main() -> int:
    payload = fixed_material_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
