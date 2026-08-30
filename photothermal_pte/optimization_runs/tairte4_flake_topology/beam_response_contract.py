"""Immutable contract for exact-binary beam-response scans with Au contacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


WAVELENGTH_M = 10.0e-6
TARGET_POWER_W = 285.0e-6
FLAKE_BOUNDS_M = (-12.0e-6, 12.0e-6)
CONTACT_INNER_EDGE_M = 10.0e-6
AU_THICKNESS_M = 50.0e-9
AU_INDEX_AT_10UM = complex(12.1, 69.2)
AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K = 19.89e6
OPTICAL_DOMAIN_SPAN_M = 48.0e-6
SOURCE_APERTURE_SPAN_M = 34.0e-6
SOURCE_TO_PML_MINIMUM_CLEARANCE_M = 2.0e-6
RESPONSE_CONTROL_VOLUME_HALF_SPAN_M = 14.0e-6
NOMINAL_WAIST_M = 8.5e-6
SOURCE_OBJECT_WAIST_SCALE = 8.36043075475035 / 8.5
WAIST_SWEEP_UM = (4.25, 6.375, 8.5, 10.625, 12.75)
POSITION_SWEEP_UM = (-10.0, -5.0, 0.0, 5.0, 10.0)

AU_OPTICAL_REFERENCE = {
    "dataset": "Ordal et al. 1987 tabulated Au optical constants",
    "doi": "https://doi.org/10.1364/AO.26.000744",
    "data_url": (
        "https://raw.githubusercontent.com/polyanskiy/"
        "refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml"
    ),
    "wavelength_um": 10.0,
    "n": AU_INDEX_AT_10UM.real,
    "k": AU_INDEX_AT_10UM.imag,
}

AU_THERMAL_REFERENCE = {
    "dataset": "NIST sample-mounting reference table",
    "url": (
        "https://www.nist.gov/ncnr/neutron-instruments/sample-environment/"
        "sample-mounting/reference-tables"
    ),
    "temperature_K": 300.0,
    "thermal_conductivity_W_mK": 317.0,
}

AU_INTERFACE_REFERENCE = {
    "measurement": "as-deposited Au/monolayer-MoS2/sapphire total conductance",
    "doi": "https://doi.org/10.1002/admi.202000364",
    "reported_W_m2K": AU_TAIRTE4_INTERFACE_CONDUCTANCE_W_M2K,
    "use_here": (
        "explicit Au/TaIrTe4 surrogate because no direct Au/TaIrTe4 thermal "
        "boundary-conductance measurement was found"
    ),
}


@dataclass(frozen=True)
class ResponseCase:
    run: int
    contact_axis: str
    geometry_mode: str
    interface_scenario: str
    polarization: str
    density_path: Path
    density_sha256: str
    base_fsp: Path
    base_fsp_sha256: str


TOP_BOTTOM_BASE = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp"
)
LEFT_RIGHT_BASE = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/"
    "uniform_rho0p5_Ea_forward_queued/attempt_0002/tairte4_flake_forward_Ea.fsp"
)
TOP_BOTTOM_BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
LEFT_RIGHT_BASE_SHA256 = "6274627f8e84cc61a8b5925472fc131041e7662b06d77141f3b52353d3578aa6"


CASES = {
    44: ResponseCase(
        44, "y", "contact_anchored", "thermally_grown", "Ea",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_exact_500nm_cleanup_20260811/void_first_exact_binary_candidate.npz"),
        "9e309d47d52ea1d5784bcae9623343b877840cd74684efd27f46490aaa75091f",
        TOP_BOTTOM_BASE, TOP_BOTTOM_BASE_SHA256,
    ),
    45: ResponseCase(
        45, "y", "contact_anchored", "thermally_grown", "Eb",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_exact_500nm_cleanup_20260811/void_first_exact_binary_candidate.npz"),
        "2d87cd272680d061f19278d54e1b5d20efa5bc3465359c76e51fef16f4fca592",
        TOP_BOTTOM_BASE, TOP_BOTTOM_BASE_SHA256,
    ),
    47: ResponseCase(
        47, "x", "left_right_contact_anchored", "thermally_grown", "Ea",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run047_Ea_fresh_current_max/forced_exact_500nm_cleanup_lr_contract_v2/void_first_exact_binary_candidate.npz"),
        "b202117579b22d3113b5a649e4f56a0f4479016c70587b8ebf61109f1b0a6ef7",
        LEFT_RIGHT_BASE, LEFT_RIGHT_BASE_SHA256,
    ),
    48: ResponseCase(
        48, "x", "left_right_contact_anchored", "thermally_grown", "Eb",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_density.npz"),
        "b07dafd6516dc858ed99489eefe053fc42441beb67eb32f506d79067ab6ad85c",
        LEFT_RIGHT_BASE, LEFT_RIGHT_BASE_SHA256,
    ),
    55: ResponseCase(
        55, "y", "contact_anchored", "evaporated", "Ea",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run055_bounded_official_dfm_exact_repair_Ea_evaporated_v1/exact_attempt_beta128/exact_candidate_01.npz"),
        "0980cc4c23f0ccf68bc971912536dec9f38d1314a1a4671d1c2c5d0b3def2fc5",
        TOP_BOTTOM_BASE, TOP_BOTTOM_BASE_SHA256,
    ),
    56: ResponseCase(
        56, "y", "contact_anchored", "evaporated", "Eb",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run056_bounded_official_dfm_exact_repair_Eb_evaporated_v1/exact_attempt_beta128/exact_candidate_02.npz"),
        "dfbff9756260204dd164786feb89bea806d0614074ff56aaf5566b4050ae87d9",
        TOP_BOTTOM_BASE, TOP_BOTTOM_BASE_SHA256,
    ),
    57: ResponseCase(
        57, "x", "left_right_contact_anchored", "evaporated", "Ea",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run057_bounded_official_dfm_exact_repair_Ea_evaporated_v1/exact_attempt_beta128/exact_candidate_03.npz"),
        "649a993ba043f69489368117dce1fa7d9851ba9f337474dc6e9ed7a52d15aed6",
        LEFT_RIGHT_BASE, LEFT_RIGHT_BASE_SHA256,
    ),
    58: ResponseCase(
        58, "x", "left_right_contact_anchored", "evaporated", "Eb",
        Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run058_bounded_official_dfm_exact_repair_Eb_evaporated_v1/exact_attempt_beta128/exact_candidate_00.npz"),
        "af397a7cd50d6b7da1f88b44659d739072267b1b60b55d75e7f156d25db7b884",
        LEFT_RIGHT_BASE, LEFT_RIGHT_BASE_SHA256,
    ),
}


def electrode_bounds_m(contact_axis: str) -> tuple[dict[str, tuple[float, float]], ...]:
    low, high = FLAKE_BOUNDS_M
    inner = CONTACT_INNER_EDGE_M
    z = (0.0, AU_THICKNESS_M)
    if contact_axis == "x":
        return (
            {"x": (low, -inner), "y": (low, high), "z": z},
            {"x": (inner, high), "y": (low, high), "z": z},
        )
    if contact_axis == "y":
        return (
            {"x": (low, high), "y": (low, -inner), "z": z},
            {"x": (low, high), "y": (inner, high), "z": z},
        )
    raise ValueError("contact_axis must be 'x' or 'y'")


def sweep_inputs(smoke: bool = False) -> list[dict[str, float | str]]:
    if smoke:
        return [{"id": "smoke_center", "kind": "smoke", "waist_um": 8.5, "x_um": 0.0, "y_um": 0.0}]
    inputs = [
        {
            "id": f"waist_{waist:g}",
            "kind": "waist",
            "waist_um": waist,
            "x_um": 0.0,
            "y_um": 0.0,
        }
        for waist in WAIST_SWEEP_UM
    ]
    positions = [
        {
            "id": f"position_x{x:g}_y{y:g}",
            "kind": "position",
            "waist_um": NOMINAL_WAIST_M * 1e6,
            "x_um": x,
            "y_um": y,
        }
        for x in POSITION_SWEEP_UM
        for y in POSITION_SWEEP_UM
        if not (np.isclose(x, 0.0) and np.isclose(y, 0.0))
    ]
    edge_check = next(
        item for item in positions if item["id"] == "position_x-10_y-10"
    )
    return [edge_check, *inputs, *(item for item in positions if item is not edge_check)]


def position_inputs() -> list[dict[str, float | str]]:
    """Return all 25 nominal-waist scan points, including the center."""

    return [
        {
            "id": f"position_x{x:g}_y{y:g}",
            "kind": "position",
            "waist_um": NOMINAL_WAIST_M * 1e6,
            "x_um": x,
            "y_um": y,
        }
        for x in POSITION_SWEEP_UM
        for y in POSITION_SWEEP_UM
    ]


def domain_center_m(x_um: float, y_um: float) -> dict[str, float]:
    """Keep the fixed device and translated source clear of transverse PML."""

    return {"x": 0.5 * x_um * 1.0e-6, "y": 0.5 * y_um * 1.0e-6}


def source_bounds_m(x_um: float, y_um: float) -> dict[str, tuple[float, float]]:
    half = 0.5 * SOURCE_APERTURE_SPAN_M
    x = x_um * 1.0e-6
    y = y_um * 1.0e-6
    bounds = {"x": (x - half, x + half), "y": (y - half, y + half)}
    center = domain_center_m(x_um, y_um)
    half_domain = 0.5 * OPTICAL_DOMAIN_SPAN_M
    clearance = min(
        value
        for axis in "xy"
        for value in (
            bounds[axis][0] - (center[axis] - half_domain),
            (center[axis] + half_domain) - bounds[axis][1],
        )
    )
    if clearance < SOURCE_TO_PML_MINIMUM_CLEARANCE_M - 1e-18:
        raise ValueError("beam aperture violates the 2 um transverse-PML clearance")
    flake_clearance = min(
        value
        for axis in "xy"
        for value in (
            FLAKE_BOUNDS_M[0] - (center[axis] - half_domain),
            (center[axis] + half_domain) - FLAKE_BOUNDS_M[1],
        )
    )
    if flake_clearance < SOURCE_TO_PML_MINIMUM_CLEARANCE_M - 1e-18:
        raise ValueError("fixed flake violates the 2 um transverse-PML clearance")
    return bounds
