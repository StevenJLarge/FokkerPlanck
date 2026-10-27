import pytest
import numpy as np

from fokker_planck.integrator import FokkerPlanck1D
from fokker_planck.types.basetypes import BoundaryCondition
import fokker_planck.forceFunctions as ff

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

    with pytest.raises(ValueError):
        _ = FokkerPlanck1D(D, dt, dx, xArray, boundary_cond=error_config)

    with pytest.raises(ValueError):
        _ = FokkerPlanck1D(D, dt, dx, xArray, split_method=error_config)


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


def test_reset_clears_work_trackers():
    dx = 0.05
    dt = 0.0005
    x_array = np.arange(-2, 2, dx)
    fpe = FokkerPlanck1D(1.0, dt, dx, x_array)
    fpe.initialize_probability(0, 0.25)
    fpe.work_step([1.0, 0.0], [2.0, 0.0], ff.harmonic_force, ff.harmonic_energy)

    assert fpe.work_accumulator != 0
    assert len(fpe.work_tracker) == 1
    assert len(fpe.power_tracker) == 1

    fpe.reset()

    assert fpe.work_accumulator == 0
    assert fpe.work_tracker == []
    assert fpe.power_tracker == []
    assert fpe.flux_tracker == 0


def test_initialize_user_probability_without_params():
    dx = 0.05
    x_array = np.arange(-2, 2, dx)
    fpe = FokkerPlanck1D(1.0, 0.0005, dx, x_array)
    fpe.initialize_user_probability(lambda x: np.exp(-(x ** 2)))

    assert np.isclose(np.sum(fpe.prob * fpe.dx), 1.0)
    assert np.isclose(fpe.x_array[np.argmax(fpe.prob)], 0.0, atol=dx)


def test_initialize_user_probability_with_params():
    dx = 0.05
    x_array = np.arange(-2, 2, dx)
    fpe = FokkerPlanck1D(1.0, 0.0005, dx, x_array)

    def gaussian(x, mean, var):
        return np.exp(-0.5 * ((x - mean) ** 2) / var)

    fpe.initialize_user_probability(gaussian, params=(-0.5, 0.2))

    assert np.isclose(np.sum(fpe.prob * fpe.dx), 1.0)
    assert np.isclose(fpe.mean, -0.5, atol=1e-2)


def test_initialized_gaussian_moments():
    dx = 0.02
    x_array = np.arange(-4, 4, dx)
    fpe = FokkerPlanck1D(1.0, 0.001, dx, x_array)
    fpe.initialize_probability(0.3, 0.25)

    assert np.isclose(np.sum(fpe.prob * fpe.dx), 1.0)
    assert np.isclose(fpe.mean, 0.3, atol=1e-3)
    assert np.isclose(fpe.variance, 0.25, atol=1e-3)
    np.testing.assert_allclose(fpe.get_prob, fpe.prob)


def test_reset_to_gaussian_when_moments_provided():
    dx = 0.05
    x_array = np.arange(-3, 3, dx)
    fpe = FokkerPlanck1D(1.0, 0.001, dx, x_array)
    fpe.initialize_probability(0.0, 1.0)
    fpe.reset(mean=-0.4, variance=0.2)

    assert np.isclose(fpe.mean, -0.4, atol=1e-2)
    assert np.isclose(fpe.variance, 0.2, atol=1e-2)


def test_work_step_is_sekimoto_energy_change_at_fixed_probability():
    dx = 0.05
    dt = 0.0005
    x_array = np.arange(-2, 2, dx)
    fpe = FokkerPlanck1D(1.0, dt, dx, x_array)
    fpe.initialize_probability(0.0, 0.25)
    prob = fpe.prob.copy()

    params_pre = [1.0, 0.0]
    params_post = [2.0, 0.0]
    expected_dw = (
        np.sum(
            (ff.harmonic_energy(x_array, params_post)
             - ff.harmonic_energy(x_array, params_pre))
            * prob
        )
        * dx
    )

    fpe.work_step(params_pre, params_post, ff.harmonic_force, ff.harmonic_energy)

    assert np.isclose(fpe.work_accumulator, expected_dw)
    assert np.isclose(fpe.work_tracker[-1], expected_dw)
    assert np.isclose(fpe.power_tracker[-1], expected_dw / dt)
    assert not np.allclose(fpe.prob, prob)


def test_flux_step_uses_physical_gradient_spacing():
    dx = 0.05
    x_array = np.arange(-2, 2, dx)
    D = 1.5
    fpe = FokkerPlanck1D(D, 0.0005, dx, x_array)
    fpe.initialize_probability(0.0, 0.2)
    force_params = [0.4]
    prob = fpe.prob.copy()

    expected_flux = (
        D * ff.constant_force(x_array, force_params) * prob
        - D * np.gradient(prob, dx)
    )
    expected_net = np.sum(expected_flux) * dx

    fpe.flux_step(force_params, ff.constant_force)

    np.testing.assert_allclose(fpe.flux, expected_flux)
    assert np.isclose(fpe.flux_tracker, expected_net)


if __name__ == "__main__":
    pytest.main([__file__])
