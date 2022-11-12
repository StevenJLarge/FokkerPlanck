import pytest
import numpy as np

from FPE.Integrator import FPE_Integrator_1D

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
    [0.0, 0.0, 0.08, 0.84]
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


def test_default_instatiation():
    # Arrange
    D = 1.0
    dt = 0.001
    dx = 0.01
    xArray = np.arange(0, 1, dx)
    # Cant instantiate this directly as it has abstract methods
    _ = FPE_Integrator_1D(D, dt, dx, xArray)


def test_input_error_handling():
    # Arrange
    error_config = "ERROR"
    D = 1.0
    dt = 0.001
    dx = 0.01
    xArray = np.arange(0, 1, dx)

    # Act
    with pytest.raises(ValueError):
        _ = FPE_Integrator_1D(D, dt, dx, xArray, diffScheme=error_config)


def test_diffusion_matrix_initialization():
    # Arrange
    dx = 0.25
    dt = 0.01
    D = 1.0
    xArray = np.arange(0, 1, dx)

    # Act``
    integrator = FPE_Integrator_1D(D, dt, dx, xArray)

    error_A = np.round(AMat_test - integrator.AMat, 5)
    error_B = np.round(BMat_test - integrator.BMat, 5)

    # Assert
    assert np.sum(error_A) == 0
    assert np.sum(error_B) == 0
