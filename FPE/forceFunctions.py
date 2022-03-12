# this python module contains all of the force functions for the FPE integrator
#
# Steven Large
# August 26th 2019

# import os
import numpy as np
import sys


def noForce(position, params):
    return 0


def constantForce(position, params):
    '''
    Constant force, force parameters: [kVal]
    potential energy: -kVal*position
    '''
    force = params[0]
    return force


def constantEnergy(position, params):
    energy = -params[0] * position
    return energy


def harmonicForce(position, params, timeIndex=None):
    '''
    Harmonic force, force parameters: [kVal,center]
    potential energy: 0.5*kVal*(position - center)**2
    Effective force for constant velocity trap motion has parameters [kVal,CPVel,D,beta]
    '''
    if(timeIndex is None):
        force = -params[0] * (position - params[1])
    else:
        force = -params[0] * (position - params[1][timeIndex])

    return force


def harmonicEnergy(position, params, timeIndex=None):
    if(timeIndex is None):
        energy = 0.5*params[0]*((position - params[1])**2)
    else:
        energy = 0.5*params[0]*((position - params[1][timeIndex])**2)

    return energy


def harmonicForce_constVel(position, params):
    return -params[0] * position - params[1] / (params[2] * params[3])


def harmonicEnergy_constVel(position, params, trapCenter=0):
    return 0.5 * params[0] * ((position - params[4]) ** 2)
    # return 0.5 * params[0] * ((position - trapCenter)**2)


def periodicForce(position, params, timeIndex=None):
    '''
    Periodic force, force parameters: [amp,nMin,minPos]
    potential energy: -0.5*amp*cos((2*np.pi/params[1])*(position - params[2]))
    '''
    if(timeIndex is None):
        force = (
            -(np.pi * params[0] / float(params[1]))
            * np.sin(
                (2 * np.pi / float(params[1])) * (position - params[2])
            )
        )
    else:
        force = (
            -(np.pi * params[0] / float(params[1]))
            * np.sin(
                (2 * np.pi / float(params[1])) * (position - params[2][timeIndex])
            )
        )
    return force


def periodicEnergy(position, params, timeIndex=None):
    if(timeIndex is None):
        energy = -0.5 * params[0] * np.cos((2 * np.pi/float(params[1])) * (position - params[2]))
    else:
        energy = -0.5 * params[0] * np.cos((2 * np.pi / float(params[1])) * (position - params[2][timeIndex]))
    return energy


def periodicForce_constVel(position, params):
    return -params[2] - np.pi * params[0] * np.sin(2 * np.pi * (position - params[1]))


def periodicEnergy_constVel(position, params):
    return -0.5 * params[0] * np.cos(2 * np.pi * (position - params[1]))


def bistableForce(position, params):
    '''
    Bistable potential (simple form), force parameters: [A,B]
    potential energy: -0.5*A*x^2 + 0.25*B*x^4
    '''
    return params[0] * (position) - params[1] * (position**3)


def bistableEnergy(position, params):
    return -0.5 * params[0] * (position**2) + 0.25 * params[1] * (position**4)


def ABELForce(position, params):
    '''
    ABEL force one (legacy version), force parameters: [eDag,A,CP1,CP2]
    potential energy: 4.0*eDag*(-0.5*CP1*x^2 + 0.25*x^4 - A*CP2*x)
    '''
    return params[0] * (position) - params[1] * (position**3) + params[2]


def ABELEnergy(position, params):
    return -0.5 * params[0] * (position**2) + 0.25 * params[1] * (position**4) - params[2] * position


def ABELForce_v2(position, params):
    '''
    ABEL force v2 (current version, circa 2019), force parameters: [a,xm]
    potential energy:   0.5*a*(x+xm)^2    IF  x < -xp
                        -0.5*(2*eDag/(xm^2 - 2*eDag/a))x^2 + eDag   IF -xp < x < xp
                        0.5*a*(x-xm)^2    IF  x > xp
    '''
    pass


def ABELEnergy_v2(position, params):
    pass


'''
Hairpin forces, force parameters
'''


'''
    ------ MULTIDIMENSIONAL POTENTIALS ------
'''


def constForce(xVals, yVals, params):
    return params[0]


# ANCHOR Harmonic potential
def harmonicEnergy_2D(xVals, yVals, params):
    """
    Potential energy: 0.5*(params[0][0]*x^2 + params[1][0]*y^2)
    """
    return 0.5 * (params[0][0] * (xVals**2) + params[1][0] * (yVals**2))


def harmonicForce_x(xVals, yVals, params_x):
    """
    This returns the force in the x direction for the harmonicEnergy_2D
    """
    return -params_x[0] * xVals


def harmonicForce_y(xVals, yVals, params_y):
    """
    This returns the force in the y direction for the harmonicEnergy_2D
    """
    return -params_y[0] * yVals


# ANCHOR Periodic potential
def periodicEnergy_2D(xVals, yVals, params):
    """
    Potential energy: 0.5*params[0]*cos(2*pi*x) + 0.5*params[1]*cos(2*pi*y)
    """
    return 0.5 * params[0] * np.cos(2 * np.pi * xVals) + 0.5 * params[1] * np.cos(2 * np.pi * yVals)


def genPeriodicLandscape_2D(xArray, yArray, params):
    energy = periodicEnergy_2D(xArray, yArray, params)
    force = np.gradient(energy)
    force_x, force_y = force[0], force[1]
    return energy, force_x, force_y


def periodicForce_2D(xVal, yVal, params, flag="z"):
    if flag == "x":
        return np.pi * params[0] * np.sin(2 * np.pi * xVal)
    if flag == "y":
        return np.pi * params[1] * np.sin(2 * np.pi * yVal)
    else:
        print("Error: unidentified force flag value, exiting program...")
        sys.exit()
