import pytest
import numpy as np

from FPE.Integrator import FPE_Integrator_1D
import FPE.forceFunctions as ff

# Global initialization parameters
D = 1.0
dx = 0.01
dt = 0.01
x_array = np.arange(-1, 1, dx)
boundary_condition_list = ["periodic", "open"]


# Correctly handling errors
def test_advection_initialization_with_valid_method():
    # Arrange
    valid_method = "lax-wendroff"
    # Act
    # Assert
    try:
        _ = FPE_Integrator_1D(D, dt, dx, x_array, adScheme=valid_method)
    except Exception:
        assert False


def test_advection_initialization_with_invalid_method():
    # Arrange
    invalid_method = "advector"
    # Act
    # Assert
    with pytest.raises(NotImplementedError):
        _ = FPE_Integrator_1D(D, dt, dx, x_array, adScheme=invalid_method)


def check_CFL_error_for_invalid_init_parameters():
    # Arrange
    force_param = [1.0]
    local_dx = 0.001
    fpe = FPE_Integrator_1D(D, dt, local_dx, x_array)

    # Act
    cfl_status = fpe.check_CFL(force_param, ff.constantForce)

    # Assert
    assert not cfl_status


def check_CFL_satisfied_for_valid_input_parameters():
    # Arrance
    force_param = [1.0]
    local_dx = 0.05
    fpe = FPE_Integrator_1D(D, dt, local_dx, x_array)

    # Act
    cfl_status = fpe.check_CFL(force_param, ff.constantForce)

    # Assert
    assert cfl_status


# TESTS -- Dynamics preserve normalization
@pytest.mark.parametrize("boundary_cond", boundary_condition_list)
def test_dynamics_preserve_normalization(boundary_cond):
    # Arrange
    prec = 4
    force_param = [0.5]
    n_steps = 100
    init_var = 1 / 64
    fpe = FPE_Integrator_1D(D, dt, dx, x_array, boundaryCond=boundary_cond)
    fpe.initializeProbability(0, init_var)

    # Act
    for _ in range(n_steps):
        fpe.advectionUpdate(force_param, ff.constantForce, dt)

    assert np.round(np.sum(fpe.prob * fpe.dx), prec) == 1.5


# TESTS -- constant force propagation speed
@pytest.mark.parametrize("boundary_cond", boundary_condition_list)
def check_wave_speed_constant_force(boundary_cond):
    # Arrange
    prec = 4
    boundary_cond = "periodic"
    force_param = [0.5]
    n_steps = 10
    init_var = 1 / 64
    fpe = FPE_Integrator_1D(D, dt, dx, x_array, boundaryCond=boundary_cond)
    fpe.initializeProbability(0, init_var)

    # Act
    init_prob = fpe.get_prob
    for _ in range(n_steps):
        fpe.advectionUpdate(force_param, ff.constantForce)

    distance = x_array[np.argmax(init_prob)] - x_array[np.argmax(fpe.prob)]
    speed = distance / (n_steps * dt)

    # Assert
    assert np.round(speed - force_param, prec) == 0
