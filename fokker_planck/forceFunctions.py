# this python module contains all of the force functions for the FPE integrator
#
# Steven Large
# August 26th 2019

# import os
import numpy as np
import sys
from typing import Tuple, Union, Iterable

InputParameters = Union[Tuple, Iterable]
PositionData = Union[float, np.ndarray]

ForceData = Union[float, np.ndarray]
EnergyData = Union[float, np.ndarray]


def no_force(
    position: PositionData, params: InputParameters
) -> ForceData:
    """Null force function, returns zero always

    Args:
        position (float ot numpy array): System position
        params (tuple): Parameters

    Returns:
        ForceData: zero force
    """
    return 0


def constant_force(
    position: PositionData, params: InputParameters
) -> ForceData:
    """Constant force function, force parameters are of form [kValue]
    where kValue is the force.

    Potential energy is -p_0 * position

    p_0 = Constant force value

    Args:
        position (PositionData): system position
        params (ForceParameters): constant force value

    Returns:
        ForceData: force at each position point
    """
    return params[0]


def harmonic_force(
    position: PositionData, params: InputParameters
) -> ForceData:
    """Harmonic potential force, with potential of the form

    E(x) = p_0 / 2 * (position - p_1)^2

    p_0 = spring constant
    p_1 = potential energy minimum position

    Args:
        position (PositionData): system position
        params (ForceParameters): poatential parameters

    Returns:
        ForceData: forces
    """
    return -params[0] * (position - params[1])


def harmonic_energy(
    position: PositionData, params: InputParameters
) -> EnergyData:
    """Harmonic energy function

    E(x) = p_0 * (x - p_1)^2

    p_0 = spring constant
    p_1 = potential energy minimum location

    Args:
        position (PositionData): system position
        params (InputParameters): Energy parameters

    Returns:
        EnergyData: Harmonic energy at the input position
    """
    return 0.5 * params[0] * ((position - params[1])**2)


def harmonic_force_const_velocity(
    position: PositionData, params: InputParameters
) -> ForceData:
    """Effecive force function for a harmonic trap with its minimum moving at
    a constant velocity. In this potential it is assmed that the x-coordinate
    reference frame situates the trap minimum at x=0 (the refernece frame co-moves
    with the harmonic potential). (beta = 1/kT is assumed to be unity)

    The force function is of the form:

    F(x) = -p_0 * x - p_1 / p_2

    p_0 = Spring constant
    p_1 = Trap center (assumed to be zero for forces)
    p_2 = Trap velocity
    p_3 = Diffusion coefficient

    Args:
        position (PositionData): system posision
        params (InputParameters): force function parameters

    Returns:
        ForceData: Force at each input position
    """
    return -params[0] * (position - params[1]) - params[2] / params[3]


def harmonic_energy_const_velocity(
    position: PositionData, params: InputParameters
) -> EnergyData:
    """Energy function corresponding to a centered harmonic potential, but with
    a parameter signature that matches the harmonic force constant-velocity
    force function

    Args:
        position (PositionData): system posision
        params (InputParameters): force function parameters

    Returns:
        EnergyData: Energy at each input position
    """
    return 0.5 * params[0] * (position - params[1]) ** 2


def periodic_force(
    position: PositionData, params: InputParameters
) -> ForceData:
    """Force produced by a sinusoindal, periodic potential.  Force function is
    of the form

    F(x) = - (pi * p_0 / p_1) * sin((2 * pi / p_1) * x - p_2 )

    p_0: Barrier height in sinusoidal potential
    p_1: Number of minima/maxima per rotation
    p_2: Phase shift (in rotations)

    Args:
        position (PositionData): System position (in rotations: 0 - 1
            corresponds to a full rotation)
        params (InputParameters): Input force parameters

    Returns:
        ForceData: Force as a function of the input rotation value(s)
    """
    force = (
        -(np.pi * params[0] * params[1])
        * np.sin((2 * np.pi * params[1]) * (position - params[2]))
    )
    return force


def periodic_energy(
    position: PositionData, params: InputParameters
) -> EnergyData:
    """Energy of a periodic potential energy function, taking the form

    E(x) = -(p_0 / 2) * cos(2 * pi * p_1 * (x - p_2))

    p_0: Height of energy barrier of potential (in units of beta = 1 / kT)
    p_1: Number of minima and maxima in a single rotation
    p_2: Phase shift, in rotations

    Args:
        position (PositionData): System position
        params (InputParameters): Energy function parameters

    Returns:
        EnergyData: energy as a function of rotation
    """
    energy = -0.5 * params[0] * np.cos((2 * np.pi * params[1]) * (position - params[2]))
    return energy


def periodic_force_const_velocity(
    position: PositionData, params: InputParameters
) -> ForceData:
    # TODO There should probably be a diffusion coefficient in here?
    """Force applied to a periodic system when the potential is rotating at a
    constant velocity. Force function is of the form:

    F(x) = - p_3 - pi * p_0 * p_1 * sin(2 * pi * p_1 * (x - p_2))

    p_0: Height of energy barrier of potential (in units of beta = 1 / kT)
    p_1: Number of minima and maxima in a single rotation
    p_2: Phase shift, in rotations
    p_3: Velocity of rotation, in rotations / unit time

    Args:
        position (PositionData): System position
        params (InputParameters): Input force parameters

    Returns:
        ForceData: Force at each of the input position(s)
    """
    return -params[2] - np.pi * params[0] * np.sin(2 * np.pi * (position - params[1]))


def bistable_force(
    position: PositionData, params: InputParameters
) -> ForceData:
    """Simple potential energy function for a bistable potential of the form.
    The energy potential is bistable when p_0 > 0, p_1 > 0 so the parameters
    are restricted to be in that range

    F(x) = p_0 * x - p_1 * x^3

    p_0: strength of parabolic energy contribution
    p_1: strength of quartic energy contribution

    Args:
        position (PositionData): System position
        params (InputParameters): Input force parameters

    Returns:
        ForceData: Force as a function of position
    """
    return params[0] * (position) - params[1] * (position**3)


def bistable_energy(
    position: PositionData, params: InputParameters
) -> EnergyData:
    """Energy function for simple bistable potential of the form:

    E(x) = -p_0/2 * x^2 + p_1/4 * x^4

    p_0: Strength of quadratic potential
    p_1: Strength of quartic potential

    Args:
        position (PositionData): System postiion
        params (InputParameters): Input energy parameters

    Returns:
        EnergyData: Energy as a function of position
    """
    return -0.5 * params[0] * (position ** 2) + 0.25 * params[1] * (position ** 4)


def ABEL_force(position: PositionData, params: InputParameters) -> ForceData:
    """Force produced by a bistable ABEL trap from Ref[1]. This
    function is parameterized so that there will be potential energy minima
    at x = +- 1. The energy function is of the form

    E(x) = 4 p_0 * (1/4 * x^4 - p_1/2 * x^2 - p_2 * p_3 * x)

    p_0: Height of energy barrier separating the two minima (when p_1 = 1)
    p_1: Control parameter dictating the height of the separating barrier
        (p_1 is in [0, 1])
    p_2: Control parameter indicating the strength of a linear perturbation to
        the potential, biasing the system into one of the two wells (p_2 is in [0, 1])
    p_3 (Optional): Parameter indicating the degree of erasure, default is 0.5
        (full erasure) but can be adjusted to larger values as well. This parameter
        is not considered a dynamic control parameter, but simply defines the
        scale of the linear term energetic bias.

    [1] Y. Jun, M. Gavrilov & J. Bechhoefer "High-precision test of Landauer's
    Principle in a Feedback Trap", Phys. Rev. Lett., 2014, 113, 190601

    Args:
        position (PositionData): Position of the system
        params (InputParameters): Energy parameters

    Returns:
        ForceData: Force as a function of the input positions

    """
    if len(params) > 3:
        A = params[3]
    else:
        A = 0.5
    return -4 * params[0] * (position ** 3 - params[1] * position - A * params[2])


def ABEL_energy(position: PositionData, params: InputParameters) -> EnergyData:
    """Potential energy function of a bistable ABEL trap from Ref[1]. This
    function is parameterized so that there will be potential energy minima
    at x = +- 1. The energy function is of the form

    E(x) = 4 p_0 * (1/4 * x^4 - p_1/2 * x^2 - p_2 * p_3 * x)

    p_0: Height of energy barrier separating the two minima (when p_1 = 1)
    p_1: Control parameter dictating the height of the separating barrier
        (p_1 is in [0, 1])
    p_2: Control parameter indicating the strength of a linear perturbation to
        the potential, biasing the system into one of the two wells (p_2 is in [0, 1])
    p_3 (Optional): Parameter indicating the degree of erasure, default is 0.5
        (full erasure) but can be adjusted to larger values as well. This parameter
        is not considered a dynamic control parameter, but simply defines the
        scale of the linear term energetic bias.

    [1] Y. Jun, M. Gavrilov & J. Bechhoefer "High-precision test of Landauer's
    Principle in a Feedback Trap", Phys. Rev. Lett., 2014, 113, 190601

    Args:
        position (PositionData): Position of the system
        params (InputParameters): Energy parameters

    Returns:
        EnergyData: Energy as a function of the input positions
    """
    if len(params) > 3:
        A = params[3]
    else:
        A = 0.5

    return 4 * params[0] * (0.25 * (position ** 4) - 0.5 * params[1] * (position ** 2) - A * params[2] * position)


def ABEL_force_v2(position, params):
    '''
    ABEL force v2 (current version, circa 2019), force parameters: [a,xm]
    potential energy:   0.5*a*(x+xm)^2    IF  x < -xp
                        -0.5*(2*eDag/(xm^2 - 2*eDag/a))x^2 + eDag   IF -xp < x < xp
                        0.5*a*(x-xm)^2    IF  x > xp
    '''
    pass


def ABEL_energy_v2(position, params):
    pass


'''
Hairpin forces, force parameters?
'''

'''
KSS Model potential (eriodic version of it...)
'''


'''
# ANCHOR ------ MULTIDIMENSIONAL POTENTIALS ------
'''


def const_force(
    x: PositionData, y: PositionData, params: InputParameters
) -> ForceData:
    """Constant force funtion for 2-dimensional potential energy function,
    assumes that the constant foce values is the same in both X and Y
    directions. Force is if the function

    F(x, y) = p_0

    Args:
        x (PositionData): x-coordinate data
        y (PositionData): y-coordinate data
        params (InputParameters): input parameters

    Returns:
        ForceData: Constant force value
    """
    return params[0]


# ANCHOR Harmonic potential
def harmonic_energy_2D(
    x: PositionData, y: PositionData, params: InputParameters
) -> EnergyData:
    """2-dimensional harmonic energy function of the form:

    E(x, y) = p_00 / 2 * (x - p_01)^2 + p_10 / 2 * (y - p_11)^2

    Parameters are arranged in an array, where the first index determines the
    dimension they act on (x or y) and the second index indicates the parameter
    within that direction

    p_00: Spring constant in x-coordinate
    p_01: Trap center in x-coordinate
    p_10: Spring constant in y-coordinate
    p_11: Trap center in y-coordinate

    Args:
        x (PositionData): System x-position values
        y (PositionData): SyStem y-position values
        params (InputParameters): Input energy parameters

    Returns:
        EnergyData: energy as a function of the input positions
    """
    return 0.5 * (params[0][0] * ((x - params[0][1]) ** 2) + params[1][0] * ((y - params[1][1]) ** 2))


def harmonic_force_z(
    z: PositionData, params: InputParameters
) -> ForceData:
    """Force in the z-direction (where z is x or y) due to 2-D harmonic
    potential energy function. For this function, we only need parameters from
    the z-direction. The energy function is of
    the form

    E(x, y) = p_00/2 * (x - p_01)^2 + p_10/2 * (x - p_11)^2

    So that (for z = x)

    F_x(x, y) = -par_x E(x, y) = -p_00 * (x - p_01)

    p_0: Spring constant in z-direction
    p_1: Trap center in z-direction

    Args:
        z (PositionData): z-coordinate position
        params (InputParameters): Parameters for z-direction dynamics

    Returns:
        ForceData: Force along the z-direction for input z data
    """
    return -params[0] * (z - params[1])


# ANCHOR Periodic potential
def periodic_energy_2D(
    x: PositionData, y: PositionData, params: InputParameters
) -> EnergyData:
    """2-D periodic energy function, given by the equation

    E(x, y) = p_00 / 2 * cos(2 * pi * p_01 * x) + p_10 / 2 * cos(2 * pi * p_11 * y)

    p_00: Energy barrier height along x-direction
    p_01: Number of potential energy minima and maxima per rotation in x
    p_10: Energy barrier height along the y-direfction
    P_11: Number of potential energy minima and maxima per rotation in y

    Args:
        x (PositionData): x position coordimates
        y (PositionData): y position coordinates
        params (InputParameters): Parameters defining the X-Y potential energy function

    Returns:
        EnergyData: Potential energy as a function of the x,y coordaintes input
    """
    E_x = 0.5 * params[0][0] * np.cos(2 * np.pi * params[0][1] * x)
    E_y = 0.5 * params[1][0] * np.cos(2 * np.pi * params[1][1] * y)
    return E_x + E_y


def gen_periodic_landscape_2D(x_array, y_array, params):
    energy = periodic_energy_2D(x_array, y_array, params)
    force = np.gradient(energy)
    force_x, force_y = force[0], force[1]
    return energy, force_x, force_y


def periodic_force_2D(x_val, y_val, params, flag="z"):
    if flag == "x":
        return np.pi * params[0] * np.sin(2 * np.pi * x_val)
    if flag == "y":
        return np.pi * params[1] * np.sin(2 * np.pi * y_val)
    else:
        print("Error: unidentified force flag value, exiting program...")
        sys.exit()
