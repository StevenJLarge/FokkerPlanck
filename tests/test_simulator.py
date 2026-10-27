# testing methods for the simulator class
import pytest
import numpy as np
from typing import Dict
from fokker_planck.simulator.base import DynamicSimulator1D, StaticSimulator1D, VoidSimulator
from fokker_planck.simulator.simulator import (
    BreathingSimulator,
    HarmonicEquilibrationSimulator,
    HarmonicTranslationSimulator,
    PeriodicEquilibrationSimulator,
)
from fokker_planck.types.basetypes import BoundaryCondition, SplitMethod, DiffScheme
import fokker_planck.forceFunctions as ff

# 2 equivalent sample input_config dictionaries
config_1 = {
    "D": 1.0,
    "dx": 0.01,
    "dt": 0.001,
    "x_min": -3,
    "x_max": 3
}

config_2 = {
    "D": 1.0,
    "dx": 0.01,
    "dt": 0.001,
    "x_array": np.arange(-3, 3, 0.01)
}

config_3 = {}

config_4 = {
    "D": 1.0,
    "dx": 0.01,
    "dt": 0.001,
}

config_5 = {
    "dx": 0.01,
    "dt": 0.001,
}

config_6 = {
    "D": 1.0,
    "dt": 0.001,
}

config_7 = {
    "D": 1.0,
    "dx": 0.01
}

configs = [config_1, config_2]
incorrect_configs = [config_3, config_4, config_5, config_6, config_7]
sample_valid_config = config_1

additional_specs = [
    {"boundary_cond": BoundaryCondition.Periodic},
    {"split_method": SplitMethod.Strang},
    {"diff_scheme": DiffScheme.Explicit}
]


# Tests for generic simulator properties
@pytest.mark.parametrize('config', configs)
def test_simulator_initialization_with_valid_config(config: Dict):
    # Act
    _ = VoidSimulator(config)


@pytest.mark.parametrize('config', incorrect_configs)
def test_simulator_initalization_with_invalid_config_raises(config: Dict):
    # Act / assert
    with pytest.raises(ValueError):
        _ = VoidSimulator(config)


def test_invalid_protocol_class_raises():
    # Arrange
    sim = VoidSimulator(sample_valid_config)
    tau = 1.0

    # Act / Raise
    with pytest.raises(ValueError):
        sim.build_protocol(tau, mode="INVALID")


@pytest.mark.parametrize('spec', additional_specs)
def test_pass_additional_specs_passes_through(spec: Dict):
    # Arrange
    config = sample_valid_config.copy()
    for k, v in spec.items():
        config[k] = v
        attr = k
    sim = VoidSimulator(config)

    # Act
    if attr == "boundary_cond":
        obj_attr = getattr(sim.fpe, 'BC')
    else:
        obj_attr = getattr(sim.fpe, attr)

    # Assert
    assert obj_attr == spec[attr]


def test_harmonic_equilibration_default_init_writes_integrator_probability():
    sim = HarmonicEquilibrationSimulator(sample_valid_config, k_trap=8.0)
    before = sim.fpe.prob.copy()

    sim.initialize_probability()

    assert not np.allclose(before, sim.fpe.prob)
    assert sim.fpe.prob[0] == 0
    assert sim.fpe.prob[-1] == 0
    assert np.isclose(np.sum(sim.fpe.prob * sim.fpe.dx), 1.0)


def test_periodic_equilibration_uniform_and_gaussian_initialization():
    sim = PeriodicEquilibrationSimulator(sample_valid_config, amp=1.0, n_minima=1)

    sim.initialize_probability()
    assert np.allclose(sim.fpe.prob, np.ones_like(sim.fpe.prob) / (len(sim.fpe.prob) * sim.fpe.dx))

    sim.initialize_probability(classification="gaussian", mean=0.0, var=0.25)
    assert np.isclose(sim.fpe.mean, 0.0, atol=1e-3)
    assert np.isclose(np.sum(sim.fpe.prob * sim.fpe.dx), 1.0)

    with pytest.raises(NotImplementedError):
        sim.initialize_probability(classification="unsupported")


def test_harmonic_translation_update_accepts_cfl_signature():
    config = {
        "D": 1.0,
        "dx": 0.05,
        "dt": 0.0005,
        "x_min": -2,
        "x_max": 2,
    }
    sim = HarmonicTranslationSimulator(config, trap_init=0, trap_fin=1, trap_strength=8)
    sim.initialize_probability()
    sim.update(0.0, 0.01)

    assert len(sim.fpe.work_tracker) == 1


def test_static_simulator_tracks_stride_multiples_and_copies_probability():
    class TinyStaticSimulator(StaticSimulator1D):
        def initialize_probability(self):
            self.fpe.initialize_probability(0, 0.25)

    config = {
        "D": 1.0,
        "dx": 0.05,
        "dt": 0.0005,
        "x_min": -2,
        "x_max": 2,
    }
    sim = TinyStaticSimulator(config, tracking_stride=10)
    sim.force_func = ff.harmonic_force
    sim.force_params = [8.0, 0.0]
    result = sim.run_simulation(tau=0.02)

    step_indices = [round(t / config["dt"]) for t in result.time]
    assert step_indices == list(range(0, 40, 10))
    assert not np.allclose(result.prob_tracker[0], result.prob_tracker[-1])


def test_dynamic_simulator_honors_tracking_stride():
    class TinyDynamicSimulator(DynamicSimulator1D):
        def initialize_probability(self):
            self.fpe.initialize_probability(0, 0.25)

        def update(self, protocol_bkw, protocol_fwd):
            return

    config = {
        "D": 1.0,
        "dx": 0.05,
        "dt": 0.0005,
        "x_min": -2,
        "x_max": 2,
    }
    sim = TinyDynamicSimulator(config, 0.0, 1.0, tracking_stride=5)
    result = sim.run_simulation(tau=0.02, mode="naive")

    n_steps = int(0.02 / config["dt"])
    expected_idx = list(range(0, n_steps - 1, 5))
    np.testing.assert_allclose(result.time, sim.time_array[expected_idx])
    assert not np.shares_memory(result.prob_tracker[0], sim.fpe.prob)


def test_breathing_simulator_simulation_time_uses_protocol_grid():
    from fokker_planck.simulator.simulator import BreathingSimulator

    config = {
        "D": 1.0,
        "dx": 0.05,
        "dt": 0.0005,
        "x_min": -2,
        "x_max": 2,
    }
    sim = BreathingSimulator(
        config, 0.5, 4.0, ff.harmonic_force, ff.harmonic_energy, tracking_stride=8
    )
    assert len(sim.simulation_time) == 0

    sim.build_protocol(tau=0.04, mode="naive")
    np.testing.assert_allclose(sim.simulation_time, sim.time_array[::8])


SIM_CONFIG = {
    "D": 1.0,
    "dx": 0.05,
    "dt": 0.0005,
    "x_min": -2,
    "x_max": 2,
}


def test_static_simulator_requires_force_definition():
    class IncompleteStaticSimulator(StaticSimulator1D):
        def initialize_probability(self):
            self.fpe.initialize_probability(0, 0.25)

    sim = IncompleteStaticSimulator(SIM_CONFIG)
    with pytest.raises(ValueError, match="force_params"):
        sim.run_simulation(tau=0.01)


def test_harmonic_equilibration_run_simulation_relaxes_toward_trap():
    sim = HarmonicEquilibrationSimulator(SIM_CONFIG, k_trap=8.0, trap_min=0.0)
    result = sim.run_simulation(tau=0.04)

    assert result.sim_type == "static"
    np.testing.assert_allclose(result.x_array, sim.fpe.x_array)
    assert len(result.prob_tracker) > 1
    assert np.max(result.prob_tracker[-1]) > np.max(result.prob_tracker[0])
    assert np.isclose(np.sum(result.prob_tracker[-1] * sim.fpe.dx), 1.0)


def test_naive_protocol_is_linear_in_control_parameter():
    sim = BreathingSimulator(
        SIM_CONFIG, 0.5, 4.0, ff.harmonic_force, ff.harmonic_energy
    )
    protocol, time = sim.build_protocol(tau=0.05, mode="naive")
    np.testing.assert_allclose(protocol, np.linspace(0.5, 4.0, len(time)))


def test_optimal_breathing_protocol_lags_naive_at_midpoint():
    # Friction ~ 1/k^3 is large at small k, so the optimal protocol should
    # still be below the linear interpolation at the midpoint time.
    sim = BreathingSimulator(
        SIM_CONFIG, 0.5, 4.0, ff.harmonic_force, ff.harmonic_energy
    )
    naive, time = sim.build_protocol(tau=0.05, mode="naive")
    optimal, _ = sim.build_protocol(tau=0.05, mode="optimal")
    mid = len(time) // 2
    assert optimal[mid] < naive[mid]


def test_constant_friction_optimal_protocol_matches_naive():
    sim = HarmonicTranslationSimulator(
        SIM_CONFIG, trap_init=0.0, trap_fin=1.0, trap_strength=8.0
    )
    naive, _ = sim.build_protocol(tau=0.05, mode="naive")
    optimal, _ = sim.build_protocol(tau=0.05, mode="optimal")
    np.testing.assert_allclose(optimal, naive, rtol=1e-3, atol=1e-3)


def test_optimal_protocol_requires_friction_array():
    class TinyDynamicSimulator(DynamicSimulator1D):
        def initialize_probability(self):
            self.fpe.initialize_probability(0, 0.25)

        def update(self, protocol_bkw, protocol_fwd):
            return

    sim = TinyDynamicSimulator(SIM_CONFIG, 0.0, 1.0)
    with pytest.raises(NotImplementedError):
        sim.build_protocol(tau=0.01, mode="optimal")


def test_breathing_simulator_run_resets_work_between_runs():
    sim = BreathingSimulator(
        SIM_CONFIG, 0.5, 4.0, ff.harmonic_force, ff.harmonic_energy
    )
    result = sim.run_simulation(tau=0.02, mode="naive")

    assert result.sim_type == "dynamic"
    assert result.CP is not None
    work_first = sim.total_work
    assert work_first != 0
    assert len(sim.work) > 0
    assert len(sim.power) == len(sim.work)

    sim.run_simulation(tau=0.02, mode="naive")
    assert np.isclose(sim.total_work, work_first)


if __name__ == "__main__":
    pytest.main([__file__])
