import pytest
import numpy as np

from fokker_planck.integrator import FokkerPlanck1D
from fokker_planck.types.basetypes import BoundaryCondition

# Source of truth matrices for each boundary condition

# Hard-wall (default)
AMat_test = np.array([
    [1.32, -0.32, 0.0, 0.0],
    [-0.08, 1.16, -0.08, 0.0],
    [0.0, -0.08, 1.16, -0.08],
    [0.0, 0.0, -0.32, 1.32]
])

BMat_test = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.08, 0.84, 0.08, 0.0],
    [0.0, 0.08, 0.84, 0.08],
    [0.0, 0.0, 0.0, 1.0]
])

# Periodic
AMat_test_periodic = np.array([
    [1.16, -0.08, 0.0, -0.08],
    [-0.08, 1.16, -0.08, 0.0],
    [0.0, -0.08, 1.16, -0.08],
    [-0.08, 0.0, -0.08, 1.16]
])

BMat_test_periodic = np.array([
    [0.84, 0.08, 0.0, 0.08],
    [0.08, 0.84, 0.08, 0.0],
    [0.0, 0.08, 0.84, 0.08],
    [0.08, 0.0, 0.08, 0.84]
])

# Open
AMat_test_open = np.array([
    [1.16, -0.08, 0.0, 0.0],
    [-0.08, 1.16, -0.08, 0.0],
    [0.0, -0.08, 1.16, -0.08],
    [0.0, 0.0, -0.08, 1.16]
])

BMat_test_open = np.array([
    [0.84, 0.08, 0.0, 0.0],
    [0.08, 0.84, 0.08, 0.0],
    [0.0, 0.08, 0.84, 0.08],
    [0.0, 0.0, 0.08, 0.84]
])

testing_matrices = [
    {'BC': BoundaryCondition.HardWall, 'AMat': AMat_test, 'BMat': BMat_test},
    {'BC': BoundaryCondition.Periodic, 'AMat': AMat_test_periodic, 'BMat': BMat_test_periodic},
    {'BC': BoundaryCondition.Open, 'AMat': AMat_test_open, 'BMat': BMat_test_open}
]


def test_default_instatiation():
    # Arrange
    D = 1.0
    dt = 0.001
    dx = 0.01
    xArray = np.arange(0, 1, dx)
    # Cant instantiate this directly as it has abstract methods
    _ = FokkerPlanck1D(D, dt, dx, xArray)


def test_input_error_handling():
    # Arrange
    error_config = "ERROR"
    D = 1.0
    dt = 0.001
    dx = 0.01
    xArray = np.arange(0, 1, dx)

    # Act
    with pytest.raises(ValueError):
        _ = FokkerPlanck1D(D, dt, dx, xArray, diff_scheme=error_config)


@pytest.mark.parametrize("init_config", testing_matrices)
def test_diffusion_matrix_initialization(init_config):
    # Arrange
    dx = 0.25
    dt = 0.01
    D = 1.0
    xArray = np.arange(0, 1, dx)
    boundaryCond = init_config["BC"]
    AMat_test_local = init_config["AMat"]
    BMat_test_local = init_config["BMat"]

    # Act
    integrator = FokkerPlanck1D(D, dt, dx, xArray, boundary_cond=boundaryCond)

    error_A = np.round(AMat_test_local - integrator.AMat, 5)
    error_B = np.round(BMat_test_local - integrator.BMat, 5)

    # Assert
    assert np.sum(error_A) == 0
    assert np.sum(error_B) == 0


if __name__ == "__main__":
    pytest.main([__file__])
