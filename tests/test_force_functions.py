import numpy as np
import pytest

import fokker_planck.forceFunctions as ff


def _assert_force_matches_minus_energy_gradient(force_fn, energy_fn, params, x):
    dx = x[1] - x[0]
    force = np.asarray(force_fn(x, params), dtype=float)
    force_from_energy = -np.gradient(energy_fn(x, params), dx)
    np.testing.assert_allclose(
        force[15:-15],
        force_from_energy[15:-15],
        atol=5e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "force_fn, energy_fn, params, x",
    [
        (ff.harmonic_force, ff.harmonic_energy, [4.0, 0.2], np.linspace(-1.5, 1.5, 801)),
        (ff.bistable_force, ff.bistable_energy, [1.0, 1.0], np.linspace(-1.5, 1.5, 801)),
        (ff.ABEL_force, ff.ABEL_energy, [1.0, 0.8, 0.3], np.linspace(-1.5, 1.5, 801)),
        (ff.periodic_force, ff.periodic_energy, [1.2, 2.0, 0.1], np.linspace(0.0, 1.0, 801)),
    ],
)
def test_force_is_negative_energy_gradient(force_fn, energy_fn, params, x):
    _assert_force_matches_minus_energy_gradient(force_fn, energy_fn, params, x)


def test_no_force_is_identically_zero():
    x = np.linspace(-1.0, 1.0, 21)
    assert ff.no_force(x, []) == 0


def test_constant_force_is_independent_of_position():
    x = np.linspace(-1.0, 1.0, 21)
    assert ff.constant_force(x, [2.5]) == 2.5


def test_harmonic_const_velocity_adds_galilean_term():
    x = np.linspace(-1.0, 1.0, 51)
    k, center, velocity, D = 4.0, 0.1, 0.8, 2.0
    params = [k, center, velocity, D]
    force = ff.harmonic_force_const_velocity(x, params)
    expected = ff.harmonic_force(x, [k, center]) - velocity / D
    np.testing.assert_allclose(force, expected)


def test_periodic_force_const_velocity_matches_periodic_force_plus_galilean_term():
    x = np.linspace(0.0, 1.0, 401)
    barrier, n_minima, phase = 1.5, 2.0, 0.1
    velocity, D = 0.4, 2.0
    params = [barrier, n_minima, phase, velocity, D]

    force = ff.periodic_force_const_velocity(x, params)
    expected = ff.periodic_force(x, params[:3]) - velocity / D

    np.testing.assert_allclose(force, expected)


def test_periodic_force_const_velocity_conservative_part_matches_energy_gradient():
    x = np.linspace(0.0, 1.0, 2001)
    dx = x[1] - x[0]
    params_potential = [1.2, 3.0, 0.05]
    velocity, D = 0.3, 1.5

    energy = ff.periodic_energy(x, params_potential)
    force_from_energy = -np.gradient(energy, dx)
    force = ff.periodic_force_const_velocity(
        x, [*params_potential, velocity, D]
    )

    np.testing.assert_allclose(
        force[20:-20],
        force_from_energy[20:-20] - velocity / D,
        atol=5e-3,
    )
