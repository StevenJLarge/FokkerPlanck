# This file contains a set of helper/utility routines for improving the
# notebook calculations
import numpy as np
from typing import Iterable, Optional


def calc_harmonic_work_theory(
    velocity: float, trap_strength: float, time: Iterable,
    beta: Optional[float] = 1.0, D: Optional[float] = 1.0
) -> Iterable:
    prefactor = (velocity ** 2) / (beta * D)
    numer = (1 - np.exp(-beta * D * trap_strength * np.array(time)))
    denom = beta * D * trap_strength

    return prefactor * (np.array(time) - (numer / denom))


def calc_harmonic_power_theory(
    velocity: float, trap_strength: float, time: Iterable,
    beta: Optional[float] = 1.0, D: Optional[float] = 1.0
) -> Iterable:
    prefactor = (velocity ** 2) / (beta * D)
    argument = 1 - np.exp(-beta * D * trap_strength * np.array(time))
    return prefactor * argument
