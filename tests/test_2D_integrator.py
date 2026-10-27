import numpy as np
import pytest

from fokker_planck.integrator_2D import FokkerPlanck2D, FPE_integrator_2D
from fokker_planck.types.basetypes import BoundaryCondition, SplitMethod
import fokker_planck.forceFunctions as ff


def _grid(dx=0.2, span=2.0):
    axis = np.arange(-span, span, dx)
    return axis, axis


def _mass(fpe: FokkerPlanck2D) -> float:
    return float(np.sum(fpe.prob) * fpe.dx * fpe.dy)


def test_alias_and_instantiation():
    x, y = _grid()
    fpe = FokkerPlanck2D(1.0, 0.001, 0.2, 0.2, x, y)
    assert isinstance(fpe, FPE_integrator_2D)
    assert fpe.dimension == 2
    assert np.isclose(_mass(fpe), 1.0)


def test_initialize_gaussian_moments():
    x, y = _grid(dx=0.1, span=3.0)
    fpe = FokkerPlanck2D(1.0, 0.001, 0.1, 0.1, x, y)
    fpe.initialize_probability(mean=(-0.3, 0.4), covariance=(0.2, 0.35))
    mx, my = fpe.mean
    cov = fpe.covariance
    assert np.isclose(_mass(fpe), 1.0)
    assert np.isclose(mx, -0.3, atol=2e-2)
    assert np.isclose(my, 0.4, atol=2e-2)
    assert np.isclose(cov[0, 0], 0.2, atol=3e-2)
    assert np.isclose(cov[1, 1], 0.35, atol=3e-2)


@pytest.mark.parametrize("bc", [
    BoundaryCondition.Periodic,
    BoundaryCondition.HardWall,
])
def test_diffusion_conserves_mass(bc):
    x, y = _grid()
    fpe = FokkerPlanck2D(
        1.0, 0.01, 0.2, 0.2, x, y, boundary_cond=bc
    )
    fpe.initialize_probability((0.0, 0.0), (0.15, 0.15))
    mass0 = _mass(fpe)
    for _ in range(15):
        fpe.diffusion_update()
    assert np.isclose(_mass(fpe), mass0, atol=1e-8)


def test_diffusion_variance_grows_as_2Dt():
    dx = 0.2
    x, y = _grid(dx=dx, span=4.0)
    D = 1.0
    dt = 0.01
    fpe = FokkerPlanck2D(
        D, dt, dx, dx, x, y, boundary_cond=BoundaryCondition.Periodic
    )
    fpe.initialize_probability((0.0, 0.0), (0.12, 0.12))
    var0 = fpe.covariance[0, 0]
    n_steps = 20
    for _ in range(n_steps):
        fpe.diffusion_update()
    expected = var0 + 2 * D * n_steps * dt
    assert np.isclose(fpe.covariance[0, 0], expected, rtol=0.15)
    assert np.isclose(fpe.covariance[1, 1], expected, rtol=0.15)


def test_open_diffusion_can_lose_mass():
    x, y = _grid(dx=0.15, span=1.0)
    fpe = FokkerPlanck2D(
        1.0, 0.01, 0.15, 0.15, x, y,
        boundary_cond=BoundaryCondition.Open
    )
    fpe.initialize_probability((0.0, 0.0), (0.3, 0.3))
    mass0 = _mass(fpe)
    for _ in range(40):
        fpe.diffusion_update()
    assert _mass(fpe) < mass0 - 1e-3


@pytest.mark.parametrize("bc", [
    BoundaryCondition.Periodic,
    BoundaryCondition.HardWall,
])
def test_advection_conserves_mass(bc):
    x, y = _grid()
    fpe = FokkerPlanck2D(
        1.0, 0.005, 0.2, 0.2, x, y, boundary_cond=bc
    )
    fpe.initialize_probability((0.0, 0.0), (0.15, 0.15))
    mass0 = _mass(fpe)
    params = (0.4, 0.0)
    forces = (ff.constant_force_2D_x, ff.constant_force_2D_y)
    for _ in range(20):
        fpe.advection_update(params, forces, fpe.dt)
    assert np.isclose(_mass(fpe), mass0, atol=1e-10)


def test_hard_wall_advection_conserves_when_driven_into_wall():
    x, y = _grid(dx=0.1, span=1.5)
    fpe = FokkerPlanck2D(
        1.0, 0.005, 0.1, 0.1, x, y,
        boundary_cond=BoundaryCondition.HardWall
    )
    fpe.initialize_probability((0.8, 0.0), (0.08, 0.08))
    mass0 = _mass(fpe)
    params = (1.2, 0.0)
    forces = (ff.constant_force_2D_x, ff.no_force_2D)
    for _ in range(25):
        fpe.advection_update(params, forces, fpe.dt)
    assert np.isclose(_mass(fpe), mass0, atol=1e-10)


def test_constant_force_translates_in_x_only():
    x, y = _grid(dx=0.1, span=2.0)
    D, dt = 1.0, 0.005
    fpe = FokkerPlanck2D(
        D, dt, 0.1, 0.1, x, y, boundary_cond=BoundaryCondition.Periodic
    )
    fpe.initialize_probability((0.0, 0.0), (0.12, 0.12))
    mean0 = fpe.mean.copy()
    fx = 0.5
    n_steps = 12
    for _ in range(n_steps):
        fpe.advection_update((fx, 0.0), (ff.constant_force_2D_x, ff.constant_force_2D_y), dt)
    mean1 = fpe.mean
    expected_dx = D * fx * n_steps * dt
    assert np.isclose(mean1[0] - mean0[0], expected_dx, atol=0.05)
    assert np.isclose(mean1[1], mean0[1], atol=0.02)


def test_open_integrate_step_does_not_renormalize():
    x, y = _grid(dx=0.1, span=1.2)
    fpe = FokkerPlanck2D(
        1.0, 0.005, 0.1, 0.1, x, y,
        boundary_cond=BoundaryCondition.Open
    )
    fpe.initialize_probability((0.5, 0.0), (0.08, 0.08))
    mass0 = _mass(fpe)
    for _ in range(30):
        fpe.integrate_step((0.8, 0.0), ff.constant_force_2D_x, ff.no_force_2D)
    assert _mass(fpe) < mass0 - 1e-3


def test_cfl_includes_d_and_both_axes():
    x, y = _grid()
    D, dt, dx = 2.0, 0.05, 0.2
    fpe = FokkerPlanck2D(D, dt, dx, dx, x, y)
    ok = fpe.check_CFL((5.0, 0.0), ff.constant_force_2D_x, ff.constant_force_2D_y)
    assert np.isclose(fpe.CFL, D * 5.0 * dt / dx)
    assert not ok


def test_2d_harmonic_mean_relaxation():
    dx = 0.15
    x, y = _grid(dx=dx, span=2.0)
    D, dt, k = 1.0, 0.0005, 8.0
    params = ([k, 0.0], [k, 0.0])
    fpe = FokkerPlanck2D(
        D, dt, dx, dx, x, y, boundary_cond=BoundaryCondition.HardWall
    )
    fpe.initialize_probability((-0.5, 0.4), (1.0 / k, 1.0 / k))
    n_steps = 80
    for _ in range(n_steps):
        fpe.integrate_step(
            params, ff.harmonic_force_2D_x, ff.harmonic_force_2D_y
        )
    t = n_steps * dt
    mx, my = fpe.mean
    assert np.isclose(mx, -0.5 * np.exp(-D * k * t), atol=0.04)
    assert np.isclose(my, 0.4 * np.exp(-D * k * t), atol=0.04)
    assert np.isclose(_mass(fpe), 1.0, atol=1e-8)


@pytest.mark.parametrize("split_method", [
    SplitMethod.Lie,
    SplitMethod.Strang,
    SplitMethod.SymStrang,
])
def test_split_methods_conserve_and_relax(split_method):
    x, y = _grid(dx=0.2, span=2.0)
    params = ([8.0, 0.0], [8.0, 0.0])
    fpe = FokkerPlanck2D(
        1.0, 0.0005, 0.2, 0.2, x, y,
        split_method=split_method,
        boundary_cond=BoundaryCondition.HardWall,
    )
    fpe.initialize_probability((-0.6, 0.5), (0.2, 0.2))
    mean0 = fpe.mean.copy()
    mass0 = _mass(fpe)
    for _ in range(50):
        fpe.integrate_step(params, ff.harmonic_force_2D_x, ff.harmonic_force_2D_y)
    mean1 = fpe.mean
    assert np.isclose(_mass(fpe), mass0, atol=1e-8)
    assert abs(mean1[0]) < abs(mean0[0])
    assert abs(mean1[1]) < abs(mean0[1])


def test_work_step_sekimoto():
    x, y = _grid(dx=0.2, span=2.0)
    fpe = FokkerPlanck2D(1.0, 0.0005, 0.2, 0.2, x, y)
    fpe.initialize_probability((0.0, 0.0), (0.25, 0.25))
    pre = ([1.0, 0.0], [1.0, 0.0])
    post = ([2.0, 0.0], [1.0, 0.0])
    X, Y = np.meshgrid(fpe.x_array, fpe.y_array, indexing="ij")
    grid = fpe.prob_grid.copy()
    vol = fpe.dx * fpe.dy
    expected = (
        np.sum(
            (ff.harmonic_energy_2D(X, Y, post) - ff.harmonic_energy_2D(X, Y, pre))
            * grid
        )
        * vol
    )
    fpe.work_step(pre, post, ff.harmonic_force_2D_x, ff.harmonic_force_2D_y, ff.harmonic_energy_2D)
    assert np.isclose(fpe.work_accumulator, expected)


def test_flux_step_uses_physical_spacing():
    x, y = _grid()
    D = 1.5
    fpe = FokkerPlanck2D(D, 0.001, 0.2, 0.2, x, y)
    fpe.initialize_probability((0.0, 0.0), (0.2, 0.2))
    params = (0.3, 0.0)
    grid = fpe.prob_grid.copy()
    fx, fy = fpe._force_fields(ff.constant_force_2D_x, ff.no_force_2D, params)
    ddx, ddy = np.gradient(grid, fpe.dx, fpe.dy)
    expected_x = D * fx * grid - D * ddx
    fpe.flux_step(params, ff.constant_force_2D_x, ff.no_force_2D)
    np.testing.assert_allclose(fpe.flux_x, expected_x)


def test_reset_clears_work():
    x, y = _grid()
    fpe = FokkerPlanck2D(1.0, 0.0005, 0.2, 0.2, x, y)
    fpe.initialize_probability((0.0, 0.0), (0.2, 0.2))
    pre = ([1.0, 0.0], [1.0, 0.0])
    post = ([1.5, 0.0], [1.0, 0.0])
    fpe.work_step(pre, post, ff.harmonic_force_2D_x, ff.harmonic_force_2D_y, ff.harmonic_energy_2D)
    fpe.reset()
    assert fpe.work_accumulator == 0
    assert fpe.work_tracker == []
