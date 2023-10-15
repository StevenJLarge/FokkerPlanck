import pytest
import numpy as np

from fokker_planck.integrator import FokkerPlanck1D
from fokker_planck.types.basetypes import BoundaryCondition
import fokker_planck.forceFunctions as ff

# Testing suite: because the majority of the raw initialization and error
# handling is handled in the diffusion and advection suites, respectively,
# here we focus on physical correctness and source-of-truth calculation accuracy

dx = 0.01
dt = 0.00025
x_array = np.arange(-3, 2, dx)

spring_const = [8, 16, 32]
diff_coeff = [0.25, 0.5, 1.0]


@pytest.mark.parametrize('k_trap', spring_const)
def test_correct_equilibrium_in_harmonic_potential(k_trap):
    # Arrange
    D = 1.0
    fpe = FokkerPlanck1D(D, dt, dx, x_array, boundary_cond=BoundaryCondition.HardWall)
    init_var = 1 / k_trap
    fpe.initialize_probability(0, init_var)
    init_prob = fpe.prob.copy()
    eq_theory = np.exp(-ff.harmonic_energy(x_array, ([k_trap, 0])))
    eq_theory = eq_theory / np.sum(eq_theory * dx)

    # Act
    for _ in range(100):
        fpe.integrate_step(([k_trap, 0]), ff.harmonic_force)

    # Assert -- on average errors do no exceed 0.0001 per element
    assert sum(np.abs(eq_theory - init_prob)) < (1e-4 * len(x_array))
    assert sum(np.abs(eq_theory - fpe.prob)) < (1e-4 * len(x_array))


@pytest.mark.parametrize('k_trap', spring_const)
@pytest.mark.parametrize('D_input', diff_coeff)
def test_correct_relaxation_of_mean_harmonic_system(k_trap, D_input):
    # Arrange
    # Use local dx and x_array here so that CFL is satisfied for all parameters
    dx_local = 0.05
    x_array_local = np.arange(-2, 1, dx_local)
    fpe = FokkerPlanck1D(D_input, dt, dx_local, x_array_local, boundary_cond=BoundaryCondition.HardWall)
    eq_var = 1 / k_trap
    init_mean = -1
    fpe.initialize_probability(init_mean, eq_var)
    n_steps = 100

    time_tracker = np.zeros(n_steps)
    mean_tracker = []
    current_time = 0

    # Act
    for i in range(n_steps):
        fpe.integrate_step(([k_trap, 0]), ff.harmonic_force)
        mean_tracker.append(fpe.mean.copy())
        time_tracker[i] = current_time
        current_time += fpe.dt

    theory_mean = -1 * np.exp(-D_input * k_trap * time_tracker)

    # Assert
    assert (np.abs(theory_mean - mean_tracker) < 1e-2).all()


@pytest.mark.parametrize('k_trap', spring_const)
@pytest.mark.parametrize('D_input', diff_coeff)
def test_variance_relaxation_harmonic(k_trap, D_input):
    # Arrange
    # Use local dx and x_array here so that CFL is satisfied for all parameters
    dx_local = 0.05
    x_array_local = np.arange(-1.5, 1.5, dx_local)
    fpe = FokkerPlanck1D(D_input, dt, dx_local, x_array_local, boundary_cond=BoundaryCondition.HardWall)
    eq_var = 1 / k_trap
    init_var = 2 * eq_var
    eq_mean = 0
    fpe.initialize_probability(eq_mean, init_var)
    n_steps = 100

    time_tracker = np.zeros(n_steps)
    variance_tracker = []
    current_time = 0

    # Act
    for i in range(n_steps):
        fpe.integrate_step(([k_trap, 0]), ff.harmonic_force)
        variance_tracker.append(fpe.variance.copy())
        time_tracker[i] = current_time
        current_time += fpe.dt

    theory_variance = eq_var + eq_var * np.exp(-np.array(time_tracker) * k_trap * 2 * D_input)

    # Assert
    assert (np.abs(theory_variance - variance_tracker) < 1e-2).all()


if __name__ == "__main__":
    pytest.main([__file__])
