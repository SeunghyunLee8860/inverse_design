from __future__ import annotations

from dataclasses import replace

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (
    AU_MATERIAL,
    SIO2_MATERIAL,
    TAIRTE4_MATERIAL,
    sampled_material_data,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_forward import (
    DENSITY_CONTROL,
    ENDPOINT_FIELD_MONITOR,
    PABS_GROUP,
    SOURCE_NAME,
    TARGET_MONITOR,
    build_layout,
    coordinate_material_partition,
    control_volume_bounds,
    material_fit_readback,
    polarization_angle_deg,
    requested_mesh_readback_gates,
    source_calibration_contract,
    validate_source_calibration_record,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    BASELINE,
    BASELINE_SOURCE_OBJECT_W0_UM,
)


class _FakeFdtd:
    def __init__(self) -> None:
        self.fdtd: list[dict[str, object]] = []
        self.gaussians: list[dict[str, object]] = []
        self.meshes: list[dict[str, object]] = []
        self.powers: list[dict[str, object]] = []
        self.objects: list[tuple[str, dict[str, object]]] = []
        self.rectangles: list[dict[str, object]] = []
        self.imports: list[dict[str, object]] = []
        self.import_calls: list[tuple[np.ndarray, ...]] = []
        self.material_types: list[str] = []
        self.material_properties: dict[tuple[object, str], object] = {}
        self.named: dict[tuple[str, str], object] = {}

    @staticmethod
    def _add(target: list[dict[str, object]]) -> dict[str, object]:
        item: dict[str, object] = {}
        target.append(item)
        return item

    def addfdtd(self):
        return self._add(self.fdtd)

    def addgaussian(self):
        return self._add(self.gaussians)

    def addmesh(self):
        return self._add(self.meshes)

    def addpower(self):
        return self._add(self.powers)

    def addobject(self, object_type: str):
        item: dict[str, object] = {}
        self.objects.append((object_type, item))
        return item

    def addrect(self):
        return self._add(self.rectangles)

    def addimport(self, properties):
        self.imports.append(dict(properties))
        return self.imports[-1]

    def importnk2(self, *values):
        self.import_calls.append(tuple(np.asarray(value) for value in values))
        return 1

    def addmaterial(self, material_type: str):
        self.material_types.append(material_type)
        return f"material_{len(self.material_types)}"

    def setmaterial(self, material, property_name: str, value):
        self.material_properties[(material, property_name)] = value

    def setnamed(self, name: str, property_name: str, value):
        self.named[(name, property_name)] = value


def test_source_contract_binds_mesh_polarization_and_calibrated_waist() -> None:
    ea = source_calibration_contract(
        BASELINE, "Ea", source_object_w0_m=4.0e-6
    )
    eb = source_calibration_contract(
        BASELINE, "Eb", source_object_w0_m=4.0e-6
    )
    recalibrated = source_calibration_contract(
        BASELINE, "Ea", source_object_w0_m=4.01e-6
    )
    cv0 = source_calibration_contract(
        replace(BASELINE, conformal_mesh="conformal variant 0").validate(),
        "Ea",
        source_object_w0_m=4.0e-6,
    )
    assert len(
        {
            ea["source_calibration_sha256"],
            eb["source_calibration_sha256"],
            recalibrated["source_calibration_sha256"],
            cv0["source_calibration_sha256"],
        }
    ) == 4
    assert polarization_angle_deg("Ea") == 90.0
    assert polarization_angle_deg("Eb") == 0.0


def test_material_run_requires_a_matching_passed_unscaled_source_record() -> None:
    contract = source_calibration_contract(
        BASELINE, "Ea", source_object_w0_m=4.0e-6
    )
    record = {
        "status": "PASSED_EXACT_AU_4UM_SOURCE_ONLY_NUMERICAL_GATE",
        "source_calibration_sha256": contract["source_calibration_sha256"],
        "polarization": "Ea",
        "all_gates_passed": True,
        "target_plane_metrics": {"downward_Poynting_power_W": 1.0},
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
        },
    }
    assert validate_source_calibration_record(record, contract)["passed"] is True
    record["source_calibration_sha256"] = "wrong"
    assert validate_source_calibration_record(record, contract)["passed"] is False


def test_source_record_can_be_bound_to_accelerator_policy_and_gpu() -> None:
    contract = source_calibration_contract(
        BASELINE,
        "Ea",
        source_object_w0_m=BASELINE_SOURCE_OBJECT_W0_UM * 1.0e-6,
    )
    record = {
        "status": "PASSED_EXACT_AU_4UM_SOURCE_ONLY_NUMERICAL_GATE_"
        "DEVELOPMENT_GPU_NOT_B200_CERTIFIED",
        "source_calibration_sha256": contract["source_calibration_sha256"],
        "polarization": "Ea",
        "all_gates_passed": True,
        "accelerator_policy": "development",
        "B200_promotion_certified": False,
        "GPU_log_evidence": {"requested_gpu_uuid": "GPU-abc"},
        "target_plane_metrics": {"downward_Poynting_power_W": 1.0},
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
        },
    }
    valid = validate_source_calibration_record(
        record,
        contract,
        expected_accelerator_policy="development",
        expected_gpu_uuid="abc",
    )
    assert valid["passed"] is True
    invalid = validate_source_calibration_record(
        record,
        contract,
        expected_accelerator_policy="b200",
        expected_gpu_uuid="GPU-def",
    )
    assert invalid["passed"] is False
    assert invalid["gates"]["accelerator_policy_matches"] is False
    assert invalid["gates"]["physical_GPU_UUID_matches"] is False


def test_source_only_and_material_layout_share_solver_source_and_mesh() -> None:
    source = _FakeFdtd()
    material = _FakeFdtd()
    source_audit = build_layout(
        source,
        case="source_only",
        polarization="Ea",
        spec=BASELINE,
        source_object_w0_m=4.0e-6,
    )
    material_audit = build_layout(
        material,
        case="simple_L",
        polarization="Ea",
        spec=BASELINE,
        source_object_w0_m=4.0e-6,
    )
    assert source.fdtd == material.fdtd
    assert source.gaussians == material.gaussians
    assert source.meshes == material.meshes
    assert source.meshes[0]["override z mesh"] is True
    assert source.meshes[0]["dz"] == BASELINE.bulk_dz_m
    assert source_audit["source_calibration_contract"] == material_audit[
        "source_calibration_contract"
    ]
    assert source.gaussians[0]["name"] == SOURCE_NAME
    assert source.powers[0]["name"] == TARGET_MONITOR
    assert material.objects[0][0] == "pabs_adv"
    assert material.objects[0][1]["name"] == PABS_GROUP
    assert len(material_audit["flux_faces"]) == 6
    assert ENDPOINT_FIELD_MONITOR in [item["name"] for item in material.powers]
    assert material_audit["geometry"]["exact_au_geometry"][
        "geometry_sha256"
    ] == "9d543a428f89fe5ea2f6910d2d98b5f97dc870cd1aac9b928760a6b4656df411"


def test_imported_density_uses_same_solver_source_mesh_and_hashes_nodes() -> None:
    exact = _FakeFdtd()
    density = _FakeFdtd()
    exact_audit = build_layout(
        exact,
        case="full",
        polarization="Eb",
        spec=BASELINE,
        source_object_w0_m=4.0e-6,
    )
    rho = np.full(CONTRACT.design_node_shape, 0.5)
    density_audit = build_layout(
        density,
        case=DENSITY_CONTROL,
        polarization="Eb",
        spec=BASELINE,
        source_object_w0_m=4.0e-6,
        projected_density=rho,
    )
    assert exact.fdtd == density.fdtd
    assert exact.gaussians == density.gaussians
    assert exact.meshes == density.meshes
    assert len(density.imports) == 1
    assert density.import_calls[0][0].shape == (81, 81, 2)
    assert density_audit["geometry"]["density_state"]["nodal_shape_xy"] == [81, 81]
    assert exact_audit["geometry"]["exact_au_geometry"]["mask_shape_xy"] == [80, 80]


def test_imported_density_fails_closed_without_exact_nodal_state() -> None:
    for value in (None, np.zeros((80, 80))):
        with np.testing.assert_raises(ValueError):
            build_layout(
                _FakeFdtd(),
                case=DENSITY_CONTROL,
                polarization="Ea",
                spec=BASELINE,
                source_object_w0_m=4.0e-6,
                projected_density=value,
            )


def test_control_volume_is_inside_domain_below_source_and_contains_stack() -> None:
    bounds = control_volume_bounds(BASELINE)
    expected_clearance = (BASELINE.pml_layers + 1) * BASELINE.outer_dxy_m
    assert np.isclose(
        bounds["x"][0] - (-0.5 * BASELINE.lateral_span_m), expected_clearance
    )
    assert np.isclose(
        0.5 * BASELINE.lateral_span_m - bounds["x"][1], expected_clearance
    )
    assert np.isclose(bounds["z"][0] - BASELINE.z_min_m, expected_clearance)
    assert bounds["x"][0] < -0.5 * CONTRACT.flake_span_x_m
    assert bounds["x"][1] > 0.5 * CONTRACT.flake_span_x_m
    assert bounds["z"][0] > BASELINE.z_min_m
    assert bounds["z"][0] < -385e-9
    assert bounds["z"][1] > CONTRACT.design_thickness_m


class _ExactMaterialReadback:
    def __init__(self) -> None:
        data = sampled_material_data()
        self.frequency = data["frequency_hz"]
        self.target = {
            AU_MATERIAL: (data["epsilon_au"],) * 3,
            SIO2_MATERIAL: (data["epsilon_sio2"],) * 3,
            TAIRTE4_MATERIAL: (
                data["epsilon_ta_x_b"],
                data["epsilon_ta_y_a"],
                data["epsilon_ta_z_c"],
            ),
        }

    def _epsilon(self, name: str, frequency: np.ndarray, component: int):
        values = self.target[name][component - 1]
        return np.interp(frequency, self.frequency, values.real) + 1j * np.interp(
            frequency, self.frequency, values.imag
        )

    def getfdtdindex(self, name, frequency, _fmin, _fmax, component):
        return np.sqrt(self._epsilon(name, frequency, component))

    def getnumericalpermittivity(
        self, name, frequency, _fmin, _fmax, _dt, component
    ):
        return self._epsilon(name, frequency, component)


def test_material_readback_compares_every_axis_over_source_band() -> None:
    result = material_fit_readback(_ExactMaterialReadback(), dt_s=1.0e-18)
    assert result["status"].startswith("PASSED")
    assert result["frequency_count"] == 81
    assert result["max_fitted_relative_error"] < 1.0e-12
    assert result["max_finite_dt_relative_error"] < 1.0e-12
    assert set(result["materials"]) == {"Au", "SiO2", "TaIrTe4"}
    for material in result["materials"].values():
        assert set(material["axes"]) == {"x", "y", "z"}


def test_mesh_readback_checks_requested_stack_and_flake_steps() -> None:
    coordinates = {
        "x": np.arange(-10.0e-6, 10.0e-6 + 50.0e-9, 50.0e-9),
        "y": np.arange(-10.0e-6, 10.0e-6 + 50.0e-9, 50.0e-9),
        "z": np.arange(-3.0e-6, 3.0e-6 + 10.0e-9, 10.0e-9),
    }
    audit = requested_mesh_readback_gates(coordinates, BASELINE)
    assert audit["all"] is True
    coordinates["z"] = np.arange(-3.0e-6, 3.0e-6 + 25.0e-9, 25.0e-9)
    assert requested_mesh_readback_gates(coordinates, BASELINE)["all"] is False


def test_coordinate_material_partition_is_disjoint_and_tracks_exact_au() -> None:
    coordinates = {
        "x": np.asarray([-9e-6, -3.95e-6, 0.0, 3.95e-6, 9e-6]),
        "y": np.asarray([-9e-6, -3.95e-6, 0.0, 3.95e-6, 9e-6]),
        "z": np.asarray([-1e-6, -300e-9, -50e-9, 25e-9, 200e-9]),
    }
    empty = coordinate_material_partition(
        coordinates, np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    )
    full = coordinate_material_partition(
        coordinates, np.ones(CONTRACT.design_shape, dtype=np.uint8)
    )
    for partition in (empty, full):
        coverage = sum(values.astype(np.uint8) for values in partition.values())
        assert np.all(coverage == 1)
    assert not np.any(empty["Au_coordinate_partition"])
    # Three interior x coordinates times three interior y coordinates at z=25 nm.
    assert np.count_nonzero(full["Au_coordinate_partition"]) == 9
