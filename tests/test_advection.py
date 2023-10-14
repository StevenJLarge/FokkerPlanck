import pytest
import numpy as np

from fokker_planck.integrator import FokkerPlanck1D
from fokker_planck.types.basetypes import BoundaryCondition
import fokker_planck.forceFunctions as ff

# Global initialization parameters
D = 1.0
dx = 0.01
dt = 0.01
x_array = np.arange(-1, 1, dx)
boundary_condition_list = [BoundaryCondition.Periodic, BoundaryCondition.Open]


def test_check_CFL_error_for_invalid_init_parameters():
    # Arrange
    force_param = [1.0]
    local_dx = 0.001
    fpe = FokkerPlanck1D(D, dt, local_dx, x_array)

    # Act
    cfl_status = fpe.check_CFL(force_param, ff.constant_force)

    # Assert
    assert not cfl_status


def test_check_CFL_satisfied_for_valid_input_parameters():
    # Arrance
    force_param = [1.0]
    local_dx = 0.05
    fpe = FokkerPlanck1D(D, dt, local_dx, x_array)

    # Act
    cfl_status = fpe.check_CFL(force_param, ff.constant_force)

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
    fpe = FokkerPlanck1D(D, dt, dx, x_array, boundary_cond=boundary_cond)
    fpe.initialize_probability(0, init_var)

    # Act
    for _ in range(n_steps):
        fpe.advection_update(force_param, ff.constant_force, dt)

    assert np.round(np.sum(fpe.prob * fpe.dx), prec) == 1.0


# TESTS -- constant force propagation speed
@pytest.mark.parametrize("boundary_cond", boundary_condition_list)
def test_wave_speed_constant_force(boundary_cond):
    # Arrange
    prec = 4
    boundary_cond = BoundaryCondition.Periodic
    force_param = [0.5]
    n_steps = 10
    init_var = 1 / 64
    fpe = FokkerPlanck1D(D, dt, dx, x_array, boundary_cond=boundary_cond)
    fpe.initialize_probability(0, init_var)

    # Act
    init_prob = fpe.get_prob
    for _ in range(n_steps):
        fpe.advection_update(force_param, ff.constant_force, dt)

    distance = x_array[np.argmax(fpe.prob)] - x_array[np.argmax(init_prob)]
    speed = distance / (n_steps * dt)

    # Assert
    assert np.round(speed - force_param, prec) == 0


if __name__ == '__main__':
    pytest.main([__file__])
