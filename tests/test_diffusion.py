import pytest
import numpy as np
from FPE import Integrator
from FPE import config
import statsmodels.api as sm

diffusion_coeffs = [1, 2, 4]
integrator_schemes = config.diffSchemes
boundary_conditions = config.boundaryConditions

D = 1.0
dt = 0.01
dx = 0.01
x_array = np.arange(-3, 3, dx)
x_array_large = np.arange(-5, 5, dx)
init_var = 1 / 128


# Check correct instantiation of constDiff parameter
def test_constDiff_parameter_initialized_correctly():
    # Arrange / Act
    fpe_cd = Integrator.FPE_Integrator_1D(D, dt, dx, x_array, constDiff=True)
    fpe_no_cd = Integrator.FPE_Integrator_1D(D, dt, dx, x_array, constDiff=False)

    # Assert
    assert fpe_cd.constDiff is True
    assert fpe_no_cd.constDiff is False


# Test normalization preservation
@pytest.mark.parametrize('BC', boundary_conditions)
def test_normalization_preservation(BC):
    # Arrange
    n_steps = 10
    fpe = Integrator.FPE_Integrator_1D(D, dt, dx, x_array, boundaryCond=BC)
    fpe.initializeProbability(0, init_var)

    # Act
    init_norm = np.sum(fpe.get_prob * dx)

    for _ in range(n_steps):
        fpe.diffusionUpdate()

    final_norm = np.sum(fpe.get_prob * dx)
    # Assert
    assert np.isclose(init_norm, 1, atol=0.001)
    assert np.isclose(final_norm, 1, atol=0.001)


# Test initialization of different diffusion schemes
@pytest.mark.parametrize('scheme', integrator_schemes)
def test_initialization_of_explicit_implicit_schemes(scheme):
    # Arrange / Act
    fpe = Integrator.FPE_Integrator_1D(D, dt, dx, x_array, diffScheme=scheme)

    # Assert
    assert fpe.diffScheme == scheme


def test_default_diffusion_scheme_initialization():
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _ = Integrator.FPE_Integrator_1D(D, dt, dx, x_array, diffScheme="UNSUPPORTED")


# Test Diffusion relation
@pytest.mark.parametrize('D', diffusion_coeffs)
def test_diffusion_relation(D):

    def variance(density, x_vals):
        dx = x_vals[1] - x_vals[0]
        mean = np.sum(x_vals * density * dx)
        var = np.sum(((x_vals - mean) ** 2) * density * dx)
        return var

    # Arrange
    total_time = 2.0
    n_steps = int((total_time / dt) / (2 * D))
    error_tolerance = 0.1
    time = 0
    fpe = Integrator.FPE_Integrator_1D(D, dt, dx, x_array_large, boundaryCond="open")
    fpe.initializeProbability(0, init_var)
    var_tracker = [variance(fpe.get_prob, fpe.xArray)]
    time_tracker = [time]
    theory_slope = 2 * D

    # Act
    for _ in range(n_steps):
        time += dt
        fpe.diffusionUpdate()
        var_tracker.append(variance(fpe.prob, fpe.xArray))
        time_tracker.append(time)

    Y = np.array(var_tracker) + init_var
    X = np.array(time_tracker)

    diff_model = sm.OLS(Y, X)
    result = diff_model.fit()

    empirical_slope = result.params[0]

    # Assert
    assert np.abs(empirical_slope - theory_slope) < error_tolerance
