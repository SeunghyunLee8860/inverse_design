from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
    ADJOINT_FIELD_REGION,
    DENSITY_CONTROL,
    ENDPOINT_FIELD_MONITOR,
    EXACT_BINARY_CONTROL,
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


_RUNNER_PATH = Path(__file__).with_name("25_run_lumerical_4um_exact_au_control.py")
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "au_dualpol_4um_exact_control_runner_test", _RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)

_EXACT_EVALUATOR_PATH = Path(__file__).with_name(
    "42_evaluate_lumerical_4um_exact_binary.py"
)
_EXACT_EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "au_dualpol_4um_exact_binary_evaluator_test", _EXACT_EVALUATOR_PATH
)
assert _EXACT_EVALUATOR_SPEC is not None and _EXACT_EVALUATOR_SPEC.loader is not None
_EXACT_EVALUATOR = importlib.util.module_from_spec(_EXACT_EVALUATOR_SPEC)
_EXACT_EVALUATOR_SPEC.loader.exec_module(_EXACT_EVALUATOR)


class _FakeFdtd:
    def __init__(self) -> None:
        self.fdtd: list[dict[str, object]] = []
        self.gaussians: list[dict[str, object]] = []
        self.meshes: list[dict[str, object]] = []
        self.powers: list[dict[str, object]] = []
        self.objects: list[tuple[str, dict[str, object]]] = []
        self.rectangles: list[dict[str, object]] = []
        self.imports: list[dict[str, object]] = []
        self.fieldregions: list[dict[str, object]] = []
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

    def addfieldregion(self):
        return self._add(self.fieldregions)

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
    ea = source_calibration_contract(BASELINE, "Ea", source_object_w0_m=4.0e-6)
    eb = source_calibration_contract(BASELINE, "Eb", source_object_w0_m=4.0e-6)
    recalibrated = source_calibration_contract(
        BASELINE, "Ea", source_object_w0_m=4.01e-6
    )
    cv0 = source_calibration_contract(
        replace(BASELINE, conformal_mesh="conformal variant 0").validate(),
        "Ea",
        source_object_w0_m=4.0e-6,
    )
    assert (
        len(
            {
                ea["source_calibration_sha256"],
                eb["source_calibration_sha256"],
                recalibrated["source_calibration_sha256"],
                cv0["source_calibration_sha256"],
            }
        )
        == 4
    )
    assert polarization_angle_deg("Ea") == 90.0
    assert polarization_angle_deg("Eb") == 0.0


def test_default_source_waist_reproduces_hash_bound_binary64_value() -> None:
    assert BASELINE_SOURCE_OBJECT_W0_UM * 1.0e-6 == 3.9561433030461415e-6
    fine = replace(
        BASELINE,
        label="fine_z5_bulk50_xy100_staircase_pml8_span20_z6_t1ps",
        flake_dxy_m=100.0 * 1.0e-9,
        stack_dz_m=5.0 * 1.0e-9,
        bulk_dz_m=50.0 * 1.0e-9,
        outer_dxy_m=200.0 * 1.0e-9,
        lateral_span_m=20.0 * 1.0e-6,
        z_min_m=-3.0 * 1.0e-6,
        z_max_m=3.0 * 1.0e-6,
        simulation_time_s=1.0 * 1.0e-12,
        conformal_mesh="staircase",
    ).validate()
    contract = source_calibration_contract(
        fine,
        "Ea",
        source_object_w0_m=BASELINE_SOURCE_OBJECT_W0_UM * 1.0e-6,
    )
    assert contract["source_calibration_sha256"] == (
        "5ded48d0c27d50247c373b0dff8c934d608a4a62c6d6c44828cd81ce160ea00a"
    )


def test_material_run_requires_a_matching_passed_unscaled_source_record() -> None:
    contract = source_calibration_contract(BASELINE, "Ea", source_object_w0_m=4.0e-6)
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
    assert (
        source_audit["source_calibration_contract"]
        == material_audit["source_calibration_contract"]
    )
    assert source.gaussians[0]["name"] == SOURCE_NAME
    assert source.powers[0]["name"] == TARGET_MONITOR
    assert material.objects[0][0] == "pabs_adv"
    assert material.objects[0][1]["name"] == PABS_GROUP
    assert len(material_audit["flux_faces"]) == 6
    assert ENDPOINT_FIELD_MONITOR in [item["name"] for item in material.powers]
    assert (
        material_audit["geometry"]["exact_au_geometry"]["geometry_sha256"]
        == "9d543a428f89fe5ea2f6910d2d98b5f97dc870cd1aac9b928760a6b4656df411"
    )


def test_completed_fsp_layout_audit_matches_fresh_exact_layout() -> None:
    for case in ("empty", "full", "simple_L"):
        expected = build_layout(
            _FakeFdtd(),
            case=case,
            polarization="Ea",
            spec=BASELINE,
            source_object_w0_m=4.0e-6,
        )
        args = SimpleNamespace(
            case=case,
            au_max_coefficients=6,
            au_fit_tolerance=0.0,
        )
        source_contract = source_calibration_contract(
            BASELINE, "Ea", source_object_w0_m=4.0e-6
        )
        recovered = _RUNNER._exact_layout_audit_without_mutation(
            args, BASELINE, source_contract
        )
        assert recovered == expected


def test_completed_run_wall_time_comes_only_from_engine_log(tmp_path) -> None:
    (tmp_path / "run_p0.log").write_text(
        "Overall wall time measurements in seconds: 1972.364367\n",
        encoding="utf-8",
    )
    assert _RUNNER._completed_run_wall_time_s(tmp_path) == 1972.364367


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


def test_arbitrary_exact_binary_mask_uses_dispersive_Au_rectangles() -> None:
    fdtd = _FakeFdtd()
    mask = np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    mask[20:25, 30:50] = 1
    audit = build_layout(
        fdtd,
        case=EXACT_BINARY_CONTROL,
        polarization="Ea",
        spec=BASELINE,
        source_object_w0_m=4.0e-6,
        exact_binary_mask=mask,
    )
    geometry = audit["geometry"]["exact_au_geometry"]
    assert geometry["mask_shape_xy"] == [80, 80]
    assert geometry["occupied_cell_count"] == int(np.sum(mask))
    assert len(fdtd.imports) == 0
    assert any(
        rectangle.get("material") == AU_MATERIAL for rectangle in fdtd.rectangles
    )


def test_exact_binary_mask_fails_closed_on_gray_or_wrong_shape() -> None:
    for mask in (
        np.full(CONTRACT.design_shape, 0.5),
        np.zeros(CONTRACT.design_node_shape),
    ):
        with np.testing.assert_raises(ValueError):
            build_layout(
                _FakeFdtd(),
                case=EXACT_BINARY_CONTROL,
                polarization="Ea",
                spec=BASELINE,
                source_object_w0_m=4.0e-6,
                exact_binary_mask=mask,
            )


def test_imported_density_can_freeze_one_adjoint_field_region() -> None:
    fdtd = _FakeFdtd()
    rho = np.full(CONTRACT.design_node_shape, 0.5)
    audit = build_layout(
        fdtd,
        case=DENSITY_CONTROL,
        polarization="Ea",
        spec=BASELINE,
        source_object_w0_m=4.0e-6,
        projected_density=rho,
        include_adjoint_field_region=True,
    )
    assert len(fdtd.fieldregions) == 1
    assert fdtd.fieldregions[0]["name"] == ADJOINT_FIELD_REGION
    assert fdtd.fieldregions[0]["source mode"] is False
    assert audit["adjoint_field_region"]["name"] == ADJOINT_FIELD_REGION


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

    def getnumericalpermittivity(self, name, frequency, _fmin, _fmax, _dt, component):
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


def test_exact_binary_evaluator_forwards_lateral_mesh_override() -> None:
    args = SimpleNamespace(
        binary_mask_npz=Path("mask.npz"),
        binary_mask_key="binary_mask",
        gpu_index=5,
        accelerator_policy="development",
        threads=8,
        mesh_label="fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps",
        flake_dxy_nm=50.0,
        stack_dz_nm=2.5,
        bulk_dz_nm=50.0,
        outer_dxy_nm=200.0,
    )
    command = _EXACT_EVALUATOR._forward_command(
        args=args,
        polarization="Ea",
        source_calibration=Path("source.json"),
        output=Path("output"),
    )
    options = dict(zip(command[2::2], command[3::2], strict=True))
    assert options["--mesh-label"] == args.mesh_label
    assert options["--flake-dxy-nm"] == "50.0"
    assert options["--stack-dz-nm"] == "2.5"
    assert options["--bulk-dz-nm"] == "50.0"
    assert options["--outer-dxy-nm"] == "200.0"


def test_exact_binary_evaluator_hash_binds_forward_raw_artifact(tmp_path) -> None:
    raw = tmp_path / "forward_raw.npz"
    raw.write_bytes(b"original")
    result = {"raw_artifacts": [_EXACT_EVALUATOR._artifact(raw)]}
    assert _EXACT_EVALUATOR._matching_artifact(result, "_raw.npz") == raw.resolve()

    raw.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="SHA256 changed"):
        _EXACT_EVALUATOR._matching_artifact(result, "_raw.npz")


def test_exact_binary_evaluator_accepts_only_hash_identical_relocated_raw(
    tmp_path,
) -> None:
    original = tmp_path / "original_raw.npz"
    relocated = tmp_path / "relocated_raw.npz"
    original.write_bytes(b"immutable payload")
    result = {"raw_artifacts": [_EXACT_EVALUATOR._artifact(original)]}
    relocated.write_bytes(original.read_bytes())
    original.unlink()
    assert (
        _EXACT_EVALUATOR._matching_artifact(result, "_raw.npz", override_path=relocated)
        == relocated.resolve()
    )
    relocated.write_bytes(b"tampered payload")
    with pytest.raises(RuntimeError, match="size changed|SHA256 changed"):
        _EXACT_EVALUATOR._matching_artifact(result, "_raw.npz", override_path=relocated)


def _exact_evaluator_args() -> SimpleNamespace:
    return SimpleNamespace(
        accelerator_policy="development",
        mesh_label="fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps",
        flake_dxy_nm=50.0,
        stack_dz_nm=2.5,
        bulk_dz_nm=50.0,
        outer_dxy_nm=200.0,
    )


def test_reused_forward_is_bound_to_mesh_policy_polarization_and_mask() -> None:
    args = _exact_evaluator_args()
    mask = np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    forward = {
        "all_gates_passed": True,
        "case": "exact_binary",
        "polarization": "Ea",
        "mesh_spec": _EXACT_EVALUATOR._requested_mesh_spec(args),
        "accelerator_policy": "development",
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
        },
        "layout": {
            "geometry": {
                "exact_au_geometry": {
                    "mask_payload_sha256": _EXACT_EVALUATOR.binary_mask_sha256(mask)
                }
            }
        },
    }
    gates = _EXACT_EVALUATOR._validate_forward_record(
        forward=forward,
        polarization="Ea",
        mask=mask,
        args=args,
    )
    assert all(gates.values())

    changed = mask.copy()
    changed[0, 0] = 1
    with pytest.raises(RuntimeError, match="provenance gates failed"):
        _EXACT_EVALUATOR._validate_forward_record(
            forward=forward,
            polarization="Ea",
            mask=changed,
            args=args,
        )


def test_reused_dualpol_forward_arguments_are_all_or_none() -> None:
    args = SimpleNamespace(
        ea_forward_result=None,
        ea_raw_npz=None,
        eb_forward_result=None,
        eb_raw_npz=None,
    )
    assert _EXACT_EVALUATOR._reuse_forward_requested(args) is False
    args.ea_forward_result = Path("ea.json")
    with pytest.raises(ValueError, match="requires Ea/Eb"):
        _EXACT_EVALUATOR._reuse_forward_requested(args)
    args.ea_raw_npz = Path("ea_raw.npz")
    args.eb_forward_result = Path("eb.json")
    args.eb_raw_npz = Path("eb_raw.npz")
    assert _EXACT_EVALUATOR._reuse_forward_requested(args) is True


def test_exact_evaluator_requires_one_matching_visible_physical_gpu() -> None:
    args = SimpleNamespace(gpu_index=4)
    assert (
        _EXACT_EVALUATOR._visible_cuda_device(
            args, environ={"CUDA_VISIBLE_DEVICES": "4"}
        )
        == "4"
    )
    for environment in (
        {},
        {"CUDA_VISIBLE_DEVICES": ""},
        {"CUDA_VISIBLE_DEVICES": "4,7"},
        {"CUDA_VISIBLE_DEVICES": "7"},
        {"CUDA_VISIBLE_DEVICES": "-1"},
    ):
        with pytest.raises(RuntimeError):
            _EXACT_EVALUATOR._visible_cuda_device(args, environ=environment)
