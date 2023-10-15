# testing methods for the simulator class
import pytest
import numpy as np
from typing import Dict
from fokker_planck.simulator.base import VoidSimulator
from fokker_planck.types.basetypes import BoundaryCondition, SplitMethod, DiffScheme

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


if __name__ == "__main__":
    pytest.main([__file__])
