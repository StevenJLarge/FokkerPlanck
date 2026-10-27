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


def test_check_CFL_includes_diffusion_coefficient():
    # Advective speed is D * F, so CFL = |F| D dt / dx
    force_param = [1.0]
    local_dx = 0.05
    # |F| dt / dx = 0.2 < 1, but D * |F| dt / dx = 2 > 1
    fpe = FokkerPlanck1D(10.0, dt, local_dx, x_array)

    cfl_status = fpe.check_CFL(force_param, ff.constant_force)

    assert not cfl_status
    assert np.isclose(fpe.CFL, 10.0 * 1.0 * dt / local_dx)


# TESTS -- mass conservation under advection (no renormalization)
conserving_boundary_conditions = [
    BoundaryCondition.Periodic,
    BoundaryCondition.HardWall,
]


def _total_mass(fpe: FokkerPlanck1D) -> float:
    return float(np.sum(fpe.prob * fpe.dx))


@pytest.mark.parametrize("boundary_cond", conserving_boundary_conditions)
def test_dynamics_preserve_normalization(boundary_cond):
    # Arrange
    force_param = [0.5]
    n_steps = 100
    init_var = 1 / 64
    fpe = FokkerPlanck1D(D, dt, dx, x_array, boundary_cond=boundary_cond)
    fpe.initialize_probability(0, init_var)

    # Act
    for _ in range(n_steps):
        fpe.advection_update(force_param, ff.constant_force, dt)

    assert np.isclose(_total_mass(fpe), 1.0, atol=1e-4)


def test_hard_wall_advection_conserves_mass_when_forced_into_wall():
    """No-flux walls must conserve probability in the LW update itself.

    This must not go through integrate_step, which can hide a leak by
    renormalizing.
    """
    local_dx = 0.02
    local_dt = 0.01
    local_x = np.arange(-1, 1, local_dx)
    fpe = FokkerPlanck1D(
        D, local_dt, local_dx, local_x, boundary_cond=BoundaryCondition.HardWall
    )
    fpe.initialize_probability(0.5, 0.05)
    mass0 = _total_mass(fpe)

    for _ in range(20):
        fpe.advection_update([2.0], ff.constant_force, local_dt)

    assert np.isclose(_total_mass(fpe), mass0, atol=1e-12)
    assert np.isclose(_total_mass(fpe), 1.0, atol=1e-12)


def test_open_integrate_step_does_not_renormalize_outflow():
    local_dx = 0.02
    local_dt = 0.005
    local_x = np.arange(-1, 1, local_dx)
    fpe = FokkerPlanck1D(
        D, local_dt, local_dx, local_x, boundary_cond=BoundaryCondition.Open
    )
    fpe.initialize_probability(0.6, 0.04)
    mass0 = _total_mass(fpe)

    for _ in range(40):
        fpe.integrate_step([1.0], ff.constant_force)

    mass1 = _total_mass(fpe)
    assert mass1 < mass0 - 1e-3
    assert mass1 < 1.0 - 1e-3


def test_conserving_integrate_step_keeps_unit_mass():
    local_dx = 0.02
    local_dt = 0.005
    local_x = np.arange(-1, 1, local_dx)
    fpe = FokkerPlanck1D(
        D, local_dt, local_dx, local_x, boundary_cond=BoundaryCondition.HardWall
    )
    fpe.initialize_probability(0.5, 0.05)

    for _ in range(40):
        fpe.integrate_step([1.0], ff.constant_force)

    assert np.isclose(_total_mass(fpe), 1.0, atol=1e-10)


# TESTS -- constant force propagation speed
@pytest.mark.parametrize("boundary_cond", [
    BoundaryCondition.Periodic,
    BoundaryCondition.Open,
])
def test_wave_speed_constant_force(boundary_cond):
    force_param = [0.5]
    n_steps = 10
    init_var = 1 / 64
    fpe = FokkerPlanck1D(D, dt, dx, x_array, boundary_cond=boundary_cond)
    fpe.initialize_probability(0, init_var)

    init_argmax = np.argmax(fpe.prob)
    for _ in range(n_steps):
        fpe.advection_update(force_param, ff.constant_force, dt)

    distance = x_array[np.argmax(fpe.prob)] - x_array[init_argmax]
    speed = distance / (n_steps * dt)

    assert np.isclose(speed, D * force_param[0], atol=dx / (n_steps * dt))


def test_wave_speed_scales_with_diffusion_coefficient():
    local_dt = 0.005
    force_param = [0.5]
    D_local = 2.0
    n_steps = 10
    fpe = FokkerPlanck1D(
        D_local, local_dt, dx, x_array, boundary_cond=BoundaryCondition.Periodic
    )
    fpe.initialize_probability(0, 1 / 64)

    init_argmax = np.argmax(fpe.prob)
    for _ in range(n_steps):
        fpe.advection_update(force_param, ff.constant_force, local_dt)

    speed = (
        (x_array[np.argmax(fpe.prob)] - x_array[init_argmax])
        / (n_steps * local_dt)
    )
    assert np.isclose(speed, D_local * force_param[0], atol=dx / (n_steps * local_dt))


def test_zero_force_advection_does_not_translate_packet():
    fpe = FokkerPlanck1D(
        D, dt, dx, x_array, boundary_cond=BoundaryCondition.Periodic
    )
    fpe.initialize_probability(0.0, 1 / 64)
    mean0 = fpe.mean

    for _ in range(20):
        fpe.advection_update([0.0], ff.no_force, dt)

    assert np.isclose(fpe.mean, mean0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__])
