import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.ansys_minimum_feature import (
    CONTRACT,
    evaluate_on_cad,
)


class FakeCad:
    def __init__(self, indicators=None, gradient=None):
        self.values = {}
        self.indicators = np.asarray(
            [0.125, 0.25] if indicators is None else indicators, dtype=float
        )
        self.gradient = gradient
        self.script = ""

    def putv(self, name, value):
        self.values[name] = np.asarray(value)

    def eval(self, script):
        self.script = script

    def getv(self, name):
        if name == "codex_dfm_indicators":
            return self.indicators
        if name == "codex_dfm_gradient":
            if self.gradient is None:
                return np.ones_like(self.values["codex_dfm_topo_rho"])
            return self.gradient
        raise KeyError(name)


def test_official_contract_matches_ansys_v261_formula():
    assert np.isclose(CONTRACT.delta_eta, 17.0 / 36.0)
    assert np.isclose(CONTRACT.eta_d, 1.0 / 36.0)
    assert np.isclose(CONTRACT.eta_e, 35.0 / 36.0)
    assert CONTRACT.penalty_scaling(12.0) == 0.0
    assert CONTRACT.penalty_scaling(16.0) == 1600.0
    assert CONTRACT.penalty_scaling(32.0) == 1.0e4


def test_official_cad_wrapper_is_inactive_through_beta_12():
    latent = np.full((7, 9), 0.5)
    cad = FakeCad()
    indicators, gradient, metadata = evaluate_on_cad(cad, latent, 8.0)
    assert np.array_equal(indicators, np.zeros(2))
    assert np.array_equal(gradient, np.zeros_like(latent))
    assert metadata["penalty_scaling"] == 0.0
    assert cad.script == ""


def test_official_cad_wrapper_preserves_shape_and_raw_sign():
    latent = np.full((7, 9), 0.5)
    expected = np.arange(latent.size, dtype=float).reshape(latent.shape) - 20.0
    cad = FakeCad(gradient=expected)
    indicators, gradient, metadata = evaluate_on_cad(cad, latent, 16.0)
    assert np.array_equal(indicators, [0.125, 0.25])
    assert np.array_equal(gradient, expected)
    assert metadata["penalty_scaling"] == 1600.0
    assert "topoparamstominfeaturesizeindicator" in cad.script
    assert "topoparamstominfeaturesizegradient" in cad.script
