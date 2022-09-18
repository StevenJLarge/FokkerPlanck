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
diff_coeff = [0.25, 0.5, 1.0, 2.0]


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
    for _ in range(50):
        fpe.integrate_step(([k_trap, 0]), ff.harmonicForce)

    # Assert
    assert sum(np.abs(eq_theory - init_prob)) < (1e-4 * len(x_array))
    assert sum(np.abs(eq_theory - fpe.prob)) < (1e-4 * len(x_array))


if __name__ == "__main__":
    pytest.main([__file__])
