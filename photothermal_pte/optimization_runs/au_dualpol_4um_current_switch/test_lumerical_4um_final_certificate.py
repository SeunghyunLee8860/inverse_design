from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    OptimizerRuntime,
)


_SCRIPT = Path(__file__).with_name(
    "43_certify_lumerical_4um_exact_binary_lateral.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "au_dualpol_4um_final_certificate_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_CERTIFIER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CERTIFIER)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        binary_mask_npz=tmp_path / "mask.npz",
        binary_mask_key="binary_mask",
        gpu_index=5,
        accelerator_policy="development",
        threads=8,
    )


def _options(command: list[str]) -> dict[str, str]:
    return dict(zip(command[2::2], command[3::2], strict=True))


def test_final_certifier_fixes_requested_100_and_50_nm_meshes(tmp_path) -> None:
    args = _args(tmp_path)
    coarse = _options(
        _CERTIFIER._forward_command(
            args=args,
            polarization="Ea",
            source_calibration=tmp_path / "coarse_source.json",
            mesh=_CERTIFIER.COARSE_MESH,
            output=tmp_path / "coarse",
        )
    )
    fine = _options(
        _CERTIFIER._forward_command(
            args=args,
            polarization="Ea",
            source_calibration=tmp_path / "fine_source.json",
            mesh=_CERTIFIER.FINE_MESH,
            output=tmp_path / "fine",
        )
    )
    assert coarse["--mesh-label"] == _CERTIFIER.COARSE_MESH_LABEL
    assert float(coarse["--flake-dxy-nm"]) == pytest.approx(100.0)
    assert fine["--mesh-label"] == _CERTIFIER.FINE_MESH_LABEL
    assert float(fine["--flake-dxy-nm"]) == pytest.approx(50.0)
    for options in (coarse, fine):
        assert float(options["--outer-dxy-nm"]) == pytest.approx(200.0)
        assert float(options["--stack-dz-nm"]) == pytest.approx(2.5)
        assert float(options["--bulk-dz-nm"]) == pytest.approx(50.0)
        assert options["--mesh-refinement"] == "conformal variant 0"
        assert options["--pml-layers"] == "8"


def test_final_certifier_pde_uses_only_fine_exact_forwards(tmp_path) -> None:
    args = _args(tmp_path)
    forwards = {
        "fine": {
            polarization: {
                "result_path": tmp_path / f"fine_{polarization}.json",
                "raw_path": tmp_path / f"fine_{polarization}_raw.npz",
            }
            for polarization in ("Ea", "Eb")
        }
    }
    command = _CERTIFIER._fine_pde_command(
        args=args,
        mask_path=args.binary_mask_npz,
        forwards=forwards,
        output=tmp_path / "pde",
    )
    options = _options(command)
    assert options["--ea-forward-result"].endswith("fine_Ea.json")
    assert options["--eb-forward-result"].endswith("fine_Eb.json")
    assert options["--ea-raw-npz"].endswith("fine_Ea_raw.npz")
    assert options["--eb-raw-npz"].endswith("fine_Eb_raw.npz")
    assert options["--mesh-label"] == _CERTIFIER.FINE_MESH_LABEL
    assert float(options["--flake-dxy-nm"]) == pytest.approx(50.0)

    coarse = _options(
        _CERTIFIER._pde_command(
            args=args,
            mask_path=args.binary_mask_npz,
            forwards={
                **forwards,
                "coarse": {
                    polarization: {
                        "result_path": tmp_path / f"coarse_{polarization}.json",
                        "raw_path": tmp_path / f"coarse_{polarization}_raw.npz",
                    }
                    for polarization in ("Ea", "Eb")
                },
            },
            output=tmp_path / "coarse_pde",
            mesh_name="coarse",
            require_through_nm=25.0,
        )
    )
    assert coarse["--mesh-label"] == _CERTIFIER.COARSE_MESH_LABEL
    assert float(coarse["--flake-dxy-nm"]) == pytest.approx(100.0)
    assert coarse["--require-pde-through-nm"] == "25.0"


def _write_source(
    path: Path,
    *,
    polarization: str,
    mesh: object,
    gpu_uuid: str = "GPU-test",
) -> None:
    path.write_text(
        json.dumps(
            {
                "case": "source_only",
                "polarization": polarization,
                "status": "PASSED_EXACT_AU_4UM_SOURCE_ONLY_NUMERICAL_GATE",
                "all_gates_passed": True,
                "mesh_spec": mesh.audit(),
                "GPU_log_evidence": {"requested_gpu_uuid": gpu_uuid},
                "accelerator_policy": "development",
                "B200_promotion_certified": False,
                "solver_version": "2026 R1.2 build 4522",
            }
        ),
        encoding="utf-8",
    )


def _runtime(tmp_path: Path, *, fine_gpu_uuid: str = "GPU-test") -> OptimizerRuntime:
    coarse: dict[str, Path] = {}
    fine: dict[str, Path] = {}
    for polarization in ("Ea", "Eb"):
        coarse_path = tmp_path / f"coarse_{polarization}.json"
        fine_path = tmp_path / f"fine_{polarization}.json"
        _write_source(
            coarse_path,
            polarization=polarization,
            mesh=_CERTIFIER.COARSE_MESH,
        )
        _write_source(
            fine_path,
            polarization=polarization,
            mesh=_CERTIFIER.FINE_MESH,
            gpu_uuid=fine_gpu_uuid,
        )
        coarse[polarization] = coarse_path
        fine[polarization] = fine_path
    return OptimizerRuntime(
        output_root=tmp_path / "output",
        source_calibration=coarse,
        gpu_index=5,
        threads=8,
        accelerator_policy="development",
        beta=1.0,
        final_xy50_source_calibration=fine,
    )


def test_runtime_binds_all_four_source_calibrations_to_mesh_gpu_solver(
    tmp_path,
) -> None:
    audit = _runtime(tmp_path).audit()
    assert set(audit["source_calibrations"]) == {"Ea", "Eb"}
    assert set(audit["final_xy50_source_calibrations"]) == {"Ea", "Eb"}
    assert all(
        row["gates"]["flake_dxy_matches"]
        for group in (
            audit["source_calibrations"],
            audit["final_xy50_source_calibrations"],
        )
        for row in group.values()
    )


def test_runtime_rejects_cross_gpu_final_source_calibration(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="GPU UUID or solver version differs"):
        _runtime(tmp_path, fine_gpu_uuid="GPU-other").audit()


def test_continuation_cannot_bypass_lateral_certifier() -> None:
    source = Path(__file__).with_name(
        "41_optimize_lumerical_4um_dualpol_continuation.py"
    ).read_text(encoding="utf-8")
    assert "43_certify_lumerical_4um_exact_binary_lateral.py" in source
    assert "require_final_xy50_source_calibration=True" in source
    assert 'str(HERE / "42_evaluate_lumerical_4um_exact_binary.py")' not in source
    assert 'output=attempt_dir / "exact_binary_certificate"' in source
    assert "if exact_switching:" in source
    assert "relaxed design switched but the exact-binary reevaluation did not" not in source
    assert 'Path(os.environ.get("EIDL_RUN_DIR", ".")).resolve()' not in source


def test_single_mesh_evaluator_explicitly_disclaims_final_certificate() -> None:
    source = Path(__file__).with_name(
        "42_evaluate_lumerical_4um_exact_binary.py"
    ).read_text(encoding="utf-8")
    assert '"final_lateral_certificate_claimed": False' in source
    assert '"requires_script_43_100_to_50nm_optical_comparison": True' in source


def _continuation_script(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "nlopt",
        SimpleNamespace(__version__="test", LD_MMA=1),
    )
    path = Path(__file__).with_name(
        "41_optimize_lumerical_4um_dualpol_continuation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "au_dualpol_4um_continuation_driver_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_binary_mask_is_atomic_and_immutable(tmp_path, monkeypatch) -> None:
    driver = _continuation_script(monkeypatch)
    path = tmp_path / "final_mask.npz"
    first = np.zeros((80, 80), dtype=np.uint8)
    driver._save_final_binary_mask(path, first)
    driver._save_final_binary_mask(path, first.copy())
    changed = first.copy()
    changed[0, 0] = 1
    with pytest.raises(RuntimeError, match="different final binary mask"):
        driver._save_final_binary_mask(path, changed)
    assert not path.with_suffix(".npz.tmp.npz").exists()


def test_passed_manifest_recovers_hash_bound_terminal_latent(
    tmp_path, monkeypatch
) -> None:
    driver = _continuation_script(monkeypatch)
    mask_path = tmp_path / "final_mask.npz"
    mask = np.zeros((80, 80), dtype=np.uint8)
    driver._save_final_binary_mask(mask_path, mask)
    state_path = tmp_path / "stage_state.npz"
    latent = np.full((81, 81), 0.5)
    np.savez_compressed(state_path, latent_final=latent)
    manifest = {
        "status": (
            "PASSED_LUMERICAL_4UM_DUALPOL_EXACT_BINARY_AU_"
            "LATERAL_PDE_NUMERICAL_CERTIFICATE"
        ),
        "passed": True,
        "final": {
            "binary_mask": driver.artifact(mask_path),
            "exact_binary_evaluation": {
                "passed": True,
                "currents_A": {"Ea": 1.0e-9, "Eb": -2.0e-9},
                "binary_mask_payload_sha256": driver.binary_mask_sha256(mask),
            },
            "continuous_stage": {
                "state_artifact": driver.artifact(state_path),
            },
        },
    }
    recovered = driver._completed_manifest_latent(manifest)
    assert np.array_equal(recovered, latent)

    state_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="stage state artifact changed"):
        driver._completed_manifest_latent(manifest)


def _fake_pde_result(
    tmp_path: Path,
    *,
    prefix: str,
    current_scale: float,
    temperature_scale: float,
) -> dict[str, object]:
    polarizations: dict[str, object] = {}
    for polarization, sign in (("Ea", 1.0), ("Eb", -1.0)):
        evidence = tmp_path / f"{prefix}_{polarization}.npz"
        temperature = np.full((4, 4), temperature_scale)
        np.savez_compressed(evidence, ta_temperature_K_50nm=temperature)
        polarizations[polarization] = {
            "selected_PDE_resolution": "50nm",
            "reference_PDE_core_step_m": 50.0e-9,
            "PDE_mesh_convergence_evidence": _CERTIFIER._artifact(evidence),
            "PDE_resolutions": {
                "50nm": {
                    "core_step_m": 50.0e-9,
                    "current_A": sign * current_scale,
                    "ta_mean_temperature_K": temperature_scale,
                    "peak_temperature_K": temperature_scale,
                }
            },
        }
    return {"polarizations": polarizations}


def test_same_pde_grid_optical_downstream_gate_is_fail_closed(tmp_path) -> None:
    fine = _fake_pde_result(
        tmp_path,
        prefix="fine",
        current_scale=1.0,
        temperature_scale=1.0,
    )
    close = _fake_pde_result(
        tmp_path,
        prefix="coarse_close",
        current_scale=1.001,
        temperature_scale=1.001,
    )
    passed = _CERTIFIER._same_pde_grid_optical_downstream_comparison(
        coarse_result=close,
        fine_result=fine,
    )
    assert passed["passed"] is True

    far = _fake_pde_result(
        tmp_path,
        prefix="coarse_far",
        current_scale=1.01,
        temperature_scale=1.01,
    )
    failed = _CERTIFIER._same_pde_grid_optical_downstream_comparison(
        coarse_result=far,
        fine_result=fine,
    )
    assert failed["passed"] is False
    assert all(
        row["metrics"]["current_relative_change"] == pytest.approx(0.01)
        for row in failed["polarizations"].values()
    )
