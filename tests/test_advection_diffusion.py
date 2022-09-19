import pytest
import numpy as np

from FPE.Integrator import FPE_Integrator_1D
import FPE.forceFunctions as ff


# Tests - does it relax to correct equilibrium distribution?
#       - Does it reach correct steady-state in constatnt velocity harmonic
#       - Check relaxation of mean
#       - Check relaxation of variance
#       - Check excess power relation?

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
    fpe = FPE_Integrator_1D(D, dt, dx, x_array, boundaryCond="hard-wall")
    init_var = 1 / k_trap
    fpe.initializeProbability(0, init_var)
    init_prob = fpe.prob.copy()
    eq_theory = np.exp(-ff.harmonicEnergy(x_array, ([k_trap, 0])))
    eq_theory = eq_theory / np.sum(eq_theory * dx)

    # Act
    for _ in range(100):
        fpe.integrate_step(([k_trap, 0]), ff.harmonicForce)

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
    fpe = FPE_Integrator_1D(D_input, dt, dx_local, x_array_local, boundaryCond="hard-wall")
    eq_var = 1 / k_trap
    init_mean = -1
    fpe.initializeProbability(init_mean, eq_var)
    n_steps = 100

    time_tracker = np.zeros(n_steps)
    mean_tracker = []
    current_time = 0

    # Act
    for i in range(n_steps):
        fpe.integrate_step(([k_trap, 0]), ff.harmonicForce)
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
    fpe = FPE_Integrator_1D(D_input, dt, dx_local, x_array_local, boundaryCond="hard-wall")
    eq_var = 1 / k_trap
    init_var = 2 * eq_var
    eq_mean = 0
    fpe.initializeProbability(eq_mean, init_var)
    n_steps = 100

    time_tracker = np.zeros(n_steps)
    variance_tracker = []
    current_time = 0

    # Act
    for i in range(n_steps):
        fpe.integrate_step(([k_trap, 0]), ff.harmonicForce)
        variance_tracker.append(fpe.variance.copy())
        time_tracker[i] = current_time
        current_time += fpe.dt

    theory_variance = eq_var + eq_var * np.exp(-np.array(time_tracker) * k_trap * 2 * D_input)

    # Assert
    assert (np.abs(theory_variance - variance_tracker) < 1e-2).all()


if __name__ == "__main__":
    pytest.main([__file__])
