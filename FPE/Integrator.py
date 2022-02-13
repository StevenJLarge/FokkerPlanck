'''
Filename: FPE_integrator_v4.py

This python script contains the routines used to solve the Fokker-Planck
equation numerically via a split-integrator scheme with multiple methods

Author:         Steven Large
Created:        August 25th 2019
Last Modified:  January 3rd 2020

Software:       python 3.7.x (compatible with python 2.x.x and 3.x.x)
'''

from typing import Callable, Optional, Tuple
import numpy as np
import scipy.sparse
import time
# import forceFunctions as ff
from FPE.base import BaseIntegrator


class FPE_Integrator_1D(BaseIntegrator):

    def __init__(
        self, D: float, dt: float, dx: float, xArray: float,
        diffScheme: Optional[str] = 'crank-nicolson',
        adScheme: Optional[str] = 'lax-wendroff',
        boundaryCond: Optional[str] = 'hard-wall',
        splitMethod: Optional[str] = 'strang',
        output: Optional[bool] = True,
        constDiff: Optional[bool] = True
    ):
        super().__init__(
            D, dt, diffScheme, adScheme, boundaryCond, splitMethod, output,
            constDiff
        )
        self.dx = dx
        # self.dt = dt

        self.N = len(xArray)
        self.prob = np.ones(self.N) / (self.N * self.dx)
        self.xArray = xArray

        self.initDiffusionMatrix()

    def reset(self):
        pass

    def initializePhysicalTrackers(self):
        # Work and power tracking arrays
        self.workAccumulator = 0
        self.workTracker = []
        self.powerTracker = []
        self.timeTracker = []

        # Total (integrated) flux tracker
        self.flux = np.zeros(len(self.xArray))
        self.fluxTracker = 0

    def initializeProbability(self, mean: float, var: float):
        # ANCHOR port this to parent class
        self.prob = np.exp(-(0.5 / var) * ((self.xArray - mean)**2))
        self.prob = self.prob / (sum(self.prob) * self.dx)

    def initDiffusionMatrix(self):
        if(self.output is True):
            print("\n\nInitializing diffusion term integration matrix...\n")
        # Set parameters for diffusion matrix iteration
        super()._setDiffusionScheme()

        if(self.output is True):
            print("\t\tInitializing integration matrices for diffusion\n")
        alpha = self.D * self.dt / (self.dx * self.dx)
        self.AMat = np.zeros((self.N, self.N))
        self.BMat = np.zeros((self.N, self.N))

        # Initialize boundary columns based on self.BC
        self._initializeBoundaryTerms(alpha)

        # Initialize bulk matrix terms
        for rowIndex in range(1, self.N - 1):
            self.AMat[rowIndex, :] = [
                1 + 2 * alpha * self.expImp if col == rowIndex
                else -self.expImp * alpha if col == (rowIndex - 1)
                else -self.expImp * alpha if col == (rowIndex + 1)
                else 0 for col in range(self.N)
            ]

            self.BMat[rowIndex, :] = [
                1 - 2 * alpha * (1 - self.expImp) if col == rowIndex
                else alpha * (1 - self.expImp) if col == (rowIndex - 1)
                else alpha * (1 - self.expImp) if col == (rowIndex + 1)
                else 0 for col in range(self.N)
            ]

        # Calculate 'C' matrix for matmul operation when diffusion is constant
        self.CMat = np.matmul(np.linalg.inv(self.AMat), self.BMat)

        # Test if sparse-matrix iteration steps are faster than normal matrix
        # multiplication
        self.testSparse()

    def _matrixBoundary_A(self, alpha: float, idx: int):

        if(self.BC == "periodic"):
            self.AMat[idx, :] = [
                1 + 2 * alpha * self.expImp if col == idx
                else -self.expImp * alpha if col == (idx + 1) % self.N
                else -self.expImp * alpha if col == (idx - 1) % self.N
                else 0 for col in range(self.N)
            ]

        elif(self.BC == "open"):
            self.AMat[idx, :] = [
                1 + 2 * alpha * self.expImp if col == idx
                else -self.expImp * alpha if col == abs(idx - 1)
                else 0 for col in range(self.N)
            ]

        else:
            self.AMat[idx, :] = [
                1 + 2 * alpha if col == idx
                else -2 * alpha if col == abs(idx - 1)
                else 0 for col in range(self.N)
            ]

    def _matrixBoundary_B(self, alpha: float, idx: int):
        if(self.BC == "periodic"):
            self.BMat[idx, :] = [
                1 - 2 * alpha * (1 - self.expImp) if col == idx
                else alpha * (1 - self.expImp) if col == (idx + 1) % self.N
                else alpha * (1 - self.expImp) if col == (idx - 1) % self.N
                else 0 for col in range(self. N)
            ]

        elif(self.BC == "open"):
            self.BMat[idx, :] = [
                1 - 2 * alpha * (1 - self.expImp) if col == idx
                else alpha * (1 - self.expImp) if col == abs(idx - 1)
                else 0 for col in range(self.N)
            ]

        else:
            self.BMat[idx, :] = [
                1 if col == idx
                else 0 for col in range(self.N)
            ]

    def _initializeBoundaryTerms(self, alpha: float):
        # Left-side boundary
        self._matrixBoundary_A(alpha, 0)
        self._matrixBoundary_B(alpha, 0)

        # Right-side boundary
        self._matrixBoundary_A(alpha, self.N - 1)
        self._matrixBoundary_B(alpha, self.N - 1)

    def work_step(
        self, forceParams: Tuple, forceParams_new: Tuple,
        forceFunction: Callable, energyFunction: Callable
    ):
        currEnergy = (
            sum(energyFunction(self.xArray, forceParams) * self.prob) * self.dx
        )
        newEnergy = (
            sum(energyFunction(self.xArray, forceParams_new) * self.prob) * self.dx
        )
        self.integrate_step(forceParams_new, forceFunction)
        self.workAccumulator += newEnergy - currEnergy
        self.workTracker.append(self.workAccumulator)
        self.powerTracker.append((newEnergy - currEnergy) / self.dt)

    def flux_step(self, forceParams: Tuple, forceFunction: Callable):
        self.flux = (
            self.D * forceFunction(self.xArray, forceParams) * self.prob
            - self.D * np.gradient(self.prob)
        )
        self.fluxTracker = sum(self.flux) * self.dx
        self.integrate_step(forceParams, forceFunction)

    def check_CFL(self, forceParams: Tuple, forceFunction: Callable):
        maxForce = 0
        for index in range(len(self.xArray)):
            tempForce = np.abs(forceFunction(self.xArray[index], forceParams))
            if(tempForce > maxForce):
                maxForce = tempForce

        self.CFL = maxForce * self.dt / self.dx
        if(self.CFL > 1):
            print("\n\n\t\tStability warning, invalid CFL --> " + str(self.CFL) + "\n\n")
            returnVal = False
        else:
            print("\n\n\t\tCFL criterion satisfied, CFL --> " + str(self.CFL) + "\n\n")
            returnVal = True

        return returnVal

    def _getFluxDiff_LaxWendroff(
        self, forceFunction: Callable, forceParams: Tuple, deltaT: float,
        idx: int
    ) -> float:
        fluxFw = (
            (self.D * deltaT / self.dx)
            * forceFunction(self.xArray[(idx + 1) % len(self.xArray)], forceParams)
            * self.prob[(idx + 1) % len(self.xArray)]
        )
        fluxRev = (
            (self.D * deltaT / self.dx)
            * forceFunction(self.xArray[idx], forceParams)
            * self.prob[idx]
        )

        return fluxFw - fluxRev

    def _calcBoundaryFlux_laxWendroff(
        self, forceParams: Tuple, forceFunction: Callable, deltaT: float
    ) -> Tuple[np.ndarray, np.ndarray]:

        halfProb = np.zeros(len(self.prob) + 1)
        halfFlux = np.zeros(len(self.prob) + 1)

        # Bulk terms
        for i in range(len(self.prob) - 1):

            halfProb[i + 1] = (
                0.5 * (self.prob[i + 1] + self.prob[i])
                # BUG There was an inconsistency between the differet iBCs for
                # this, HW had a factor of 1/2 in front of the flux difference,
                # check this
                - self._getFluxDiff_LaxWendroff(forceFunction, forceParams, deltaT, i)
            )
            # NOTE Also have factor of 1/2 here (see below)
            halfFlux[i + 1] = (
                0.5 * forceFunction(self.xArray[i] + 0.5 * self.dx, forceParams) * halfProb[i + 1]
            )

        # For HW noting else needs to be done
        # For periodic need to specify boundaries (self.N-1 index)
        if self.BC == 'periodic':
            halfProb[0] = (
                0.5 * (self.prob[0] + self.prob[-1])
                - self._getFluxDiff_LaxWendroff(
                    forceFunction, forceParams, deltaT, len(self.prob) - 1
                )
            )
            # NOTE Same factor of 1/2 here as well.
            # Seems to make the calculations work...
            halfFlux[0] = (
                0.5 * forceFunction(self.xArray[0] - 0.5 * self.dx, forceParams) * halfProb[0]
            )
            halfFlux[-1] = halfFlux[0]

        # NOTE For open BCs, there are currently no modifications to the
        # specification of coundary terms. I think this is incorrect?
        # elif self.BC == "open":
        return halfProb, halfFlux

    def laxWendroff(
        self, forceParams: Tuple, forceFunction: Callable, deltaT: float
    ):
        newProb = np.zeros(len(self.prob))

        halfProb, halfFlux = self._calcBoundaryFlux_laxWendroff(
            forceParams, forceFunction, deltaT
        )

        for index in range(len(self.prob)):
            newProb[index] = (
                self.prob[index]
                - (self.D * deltaT / self.dx)
                * (halfFlux[index+1] - halfFlux[index])
            )

        self.prob = newProb

    def lax(self, forceParams: Tuple, forceFunction: Callable, deltaT: float):
        alpha = deltaT / (2 * self.dx)
        newProb = np.zeros(len(self.prob))

        for index in range(len(self.prob) - 2):
            newProb[index+1] = (
                0.5 * (self.prob[index+1] - self.prob[index-1])
                - alpha * (
                    forceFunction(self.xArray[index+2], forceParams) * self.prob[index+2]
                    - forceFunction(self.xArray[index]*self.prob[index])
                )
            )

        # ANCHOR IMPLEMENT BCs HERE CORRECTLY....
        newProb[0] = 0
        newProb[-1] = 0

        self.prob = newProb


class FPE_integrator_2D:

    def __init__(
        self, D: float, dt: float, dx: float, dy: float, xArray: float,
        yArray: float, diffScheme: Optional[str] = 'crank-nicolson',
        adScheme: Optional[str] = 'lax-wendroff',
        boundaryCond: Optional[str] = 'hard-wall',
        splitMethod: Optional[str] = 'strang',
        output: Optional[bool] = True, constDiff: Optional[bool] = True
    ):
        self.D = D
        self.dx = dx
        self.dy = dy
        self.dt = dt

        self.diffScheme = diffScheme.lower()
        self.adScheme = adScheme.lower()
        self.BC = boundaryCond.lower()
        self.splitMethod = splitMethod.lower()

        self.output = output
        self.Nx = len(xArray)
        self.Ny = len(yArray)
        self.N = self.Nx*self.Ny

        # self.prob = np.ones((self.N,self.N))/(sum(self.Nx*self.dx)*sum(self.Ny*self.dy))
        self.prob = np.ones(self.Nx * self.Ny) / (self.Nx * self.Ny * self.dx * self.dy)
        self.xArray = xArray
        self.yArray = yArray

        self.constDiff = constDiff
        self.sparTest = False

        # ANCHOR new additions: work and power tracking arrays (as well as time)
        self.workAccumulator = 0
        self.workTracker = []
        self.powerTracker = []
        self.timeTracker = []

        # ANCHOR new additions of total (integrated) flux tracker
        self.xFlux = np.zeros(len(xArray))
        self.yFlux = np.zeros(len(yArray))
        self.xFluxTracker = 0
        self.yFluxTracker = 0

        self.initDiffusionMatrix()

    def initDiffusionMatrix(self):
        if(self.output is True):
            print("\n\nInitializing diffusion term integration matrix...\n")

        if(self.diffScheme.lower() == "crank-nicolson"):
            if(self.output is True):
                print("\t\tUsing Crank-Nicolson integration scheme...")
            self.expImp = 0.5
        else:
            if(self.output is True):
                print("\t\tIntegration scheme not recognized, using default setting (Crank-Nicolson)...")
            self.expImp = 0.5

        if(self.output is True):
            print("\t\tInitializing integration matrices for diffusion\n")

        alpha = self.D * self.dt / (self.dx * self.dx)
        self.AMat = np.zeros((self.N, self.N))
        self.BMat = np.zeros((self.N, self.N))

        for rowIndex in range(self.N):

            if rowIndex == 0:
                if(self.BC == 'hard-wall' or self.BC == 'hw'):
                    # TODO Implement hard-wall boundary conditions
                    pass
                # print("\t\tUsing hard-wall boundary conditions...")
                # self.AMat[rowIndex,:] = [1+2*alpha if col==0 else -2*alpha if col==1 else 0 for col in range(self.N)]
                # self.BMat[rowIndex,:] = [1 if col==0 else 0 for col in range(self.N)]
                elif(self.BC == 'periodic' or self.BC == 'p'):
                    print("\t\tusing periodic boundary conditions...")
                # self.AMat[rowIndex,:] = [1+2*alpha*self.expImp if col==0 else -self.expImp*alpha
                # if col==1 else -self.expImp*alpha if col==(self.N-1) else 0 for col in range(self.N)]
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == 0
                        else -self.expImp * alpha if (
                            col == 1
                            or col == (self.N - 1)
                            or col == self.Nx
                            or col == (self.N - self.Nx)
                        )
                        else 0 for col in range(self.N)
                    ]
                # self.BMat[rowIndex,:] = [1-2*alpha*(1-self.expImp) if col==0 else alpha*(1-self.expImp)
                # if col==1 else alpha*(1-self.expImp) if col==(self.N-1) else 0 for col in range(self.N)]
                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == 0
                        else alpha * (1 - self.expImp) if (
                            col == 1
                            or col == (self.N - 1)
                            or col == self.Nx
                            or col == (self.N - self.Nx)
                        )
                        else 0 for col in range(self.N)
                    ]

                elif(self.BC == 'open' or self.BC == 'o'):
                    print("\t\tusing open domain boundary conditions...")
                    # self.AMat[rowIndex,:] = [1+2*alpha*self.expImp if col==0 else -self.expImp*alpha if col==1
                    # else 0 for col in range(self.N)]
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == 0
                        else -self.expImp * alpha if (
                            col == 1
                            or col == self.Nx
                        )
                        else 0 for col in range(self.N)
                    ]
                    # self.BMat[rowIndex,:] = [1-2*alpha*(1-self.expImp) if col==0 else alpha*(1-self.expImp)
                    # if col==1 else 0 for col in range(self.N)]
                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == 0
                        else alpha * (1 - self.expImp) if (
                            col == 1
                            or col == self.Nx
                        )
                        else 0 for col in range(self.N)
                    ]
                else:
                    # print("\t\tboundary condition not recognized, using default (hard-wall)...")
                    print("\t\tboundary condition not recognized, using default (open)...")
                    # self.AMat[rowIndex,:] = [1+2*alpha if col==0 else -2*alpha if col==1
                    # else 0 for col in range(self.N)]
                    # self.BMat[rowIndex,:] = [1 if col==0 else 0 for col in range(self.N)]
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == 0
                        else -self.expImp * alpha if (
                            col == 1
                            or col == self.Nx
                        )
                        else 0 for col in range(self.N)
                    ]

                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == 0
                        else alpha * (1 - self.expImp) if (
                            col == 1
                            or col == self.Nx
                        )
                        else 0 for col in range(self.N)
                    ]

            elif rowIndex == (self.N - 1):
                if(self.BC == 'hard-wall' or self.BC == 'hw'):
                    # TODO Implement ard-wall BCs here as well
                    pass
                # self.AMat[rowIndex,:] = [1+2*alpha if col==(self.N-1) else -2*alpha if col==(self.N-2) else 0
                # for col in range(self.N)]
                # self.BMat[rowIndex,:] = [1 if col==(self.N-1) else 0 for col in range(self.N)]
                elif(self.BC == 'periodic' or self.BC == 'p'):
                    # self.AMat[rowIndex,:] = [1+2*alpha*self.expImp if col==(self.N-1) else -self.expImp*alpha if
                    # col==(self.N-2) else -self.expImp*alpha if col==0  else 0 for col in range(self.N)]
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == (self.N - 1)
                        else -self.expImp * alpha if (
                            col == (self.N - 2)
                            or col == 0
                            or col == (self.N - (self.Nx + 1))
                            or col == (self.Nx - 1)
                        )
                        else 0 for col in range(self.N)
                    ]
                # self.BMat[rowIndex,:] = [1-2*alpha*(1-self.expImp) if col==(self.N-1) else alpha*(1-self.expImp)
                # if col==(self.N-2) else alpha*(1-self.expImp) if col==0 else 0 for col in range(self.N)]

                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == (self.N - 1)
                        else alpha * (1 - self.expImp) if (
                            col == (self.N - 2)
                            or col == 0
                            or col == (self.N - (self.Nx + 1))
                            or col == (self.Nx - 1)
                        )
                        else 0 for col in range(self.N)
                    ]

                elif(self.BC == 'open' or self.BC == 'o'):
                    # self.AMat[rowIndex,:] = [1+2*alpha*self.expImp if col==(self.N-1) else -self.expImp*alpha
                    # if col==(self.N-2) else 0 for col in range(self.N)]
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == (self.N - 1)
                        else -self.expImp * alpha if (
                            col == (self.N - 2)
                            or col == (self.N - (self.Nx + 1))
                        )
                        else 0 for col in range(self.N)
                    ]
                    # self.BMat[rowIndex,:] = [1-2*alpha*(1-self.expImp) if col==(self.N-1)
                    # else alpha*(1-self.expImp) if col==(self.N-2) else 0 for col in range(self.N)]
                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == (self.N - 1)
                        else alpha * (1 - self.expImp) if (
                            col == (self.N - 2)
                            or col == (self.N - (self.Nx + 1))
                        )
                        else 0 for col in range(self.N)
                    ]

                else:
                    # self.AMat[rowIndex,:] = [1+2*alpha if col==(self.N-1) else -2*alpha if col==(self.N-1)
                    # else 0 for col in range(self.N)]
                    # self.BMat[rowIndex,:] = [1 if col==(self.N-1) else 0 for col in range(self.N)]
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == (self.N - 1)
                        else -self.expImp * alpha if (
                            col == (self.N - 2)
                            or col == (self.N - (self.Nx + 1))
                        )
                        else 0 for col in range(self.N)
                    ]

                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == (self.N - 1)
                        else alpha * (1 - self.expImp) if (
                            col == (self.N - 2)
                            or col == (self.N - (self.Nx + 1))
                        )
                        else 0 for col in range(self.N)
                    ]

            else:
                # Here we need additional cases, depending on the rowIndex relative to the size of the second dimension
                # Current code is for open BCs with caveat for periodic BCs
                if(rowIndex < self.Nx):
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == rowIndex
                        else -self.expImp * alpha if col == (rowIndex - 1)
                        else -self.expImp * alpha if col == (rowIndex + 1)
                        else -self.expImp * alpha if col == (rowIndex + self.Nx)
                        else 0 for col in range(self.N)
                    ]

                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == rowIndex
                        else self.expImp * alpha if col == (rowIndex - 1)
                        else self.expImp * alpha if col == (rowIndex + 1)
                        else self.expImp * alpha if col == (rowIndex + self.Nx)
                        else 0 for col in range(self.N)
                    ]

                    if(self.BC == 'periodic' or self.BC == 'p'):
                        self.AMat[rowIndex, rowIndex - self.Nx] = -self.expImp * alpha
                        self.BMat[rowIndex, rowIndex - self.Nx] = self.expImp * alpha

                elif(rowIndex < (self.N - self.Nx)):
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == rowIndex
                        else -self.expImp * alpha if col == (rowIndex - 1)
                        else -self.expImp * alpha if col == (rowIndex + 1)
                        else -self.expImp * alpha if col == (rowIndex - self.Nx)
                        else -self.expImp * alpha if col == (rowIndex + self.Nx)
                        else 0 for col in range(self.N)
                    ]

                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == rowIndex
                        else self.expImp * alpha if col == (rowIndex - 1)
                        else self.expImp * alpha if col == (rowIndex + 1)
                        else self.expImp * alpha if col == (rowIndex - self.Nx)
                        else self.expImp * alpha if col == (rowIndex + self.Nx)
                        else 0 for col in range(self.N)
                    ]
                    # if(self.BC=='periodic' or self.BC=='p'):
                    #    self.AMat[rowIndex,(rowIndex-self.Nx)] = -self.expImp*alpha
                    #    self.BMat[rowIndex,(rowIndex-self.Nx)] = self.expImp*alpha
                else:
                    self.AMat[rowIndex, :] = [
                        1 + 4 * alpha * self.expImp if col == rowIndex
                        else -self.expImp * alpha if col == (rowIndex - 1)
                        else -self.expImp * alpha if col == (rowIndex + 1)
                        else -self.expImp * alpha if col == (rowIndex - self.Nx)
                        else 0 for col in range(self.N)
                    ]

                    self.BMat[rowIndex, :] = [
                        1 - 4 * alpha * (1 - self.expImp) if col == rowIndex
                        else self.expImp * alpha if col == (rowIndex - 1)
                        else self.expImp * alpha if col == (rowIndex + 1)
                        else self.expImp * alpha if col == (rowIndex - self.Nx)
                        else 0 for col in range(self.N)
                    ]

                    if(self.BC == 'periodic' or self.BC == 'p'):
                        self.AMat[rowIndex, rowIndex - (self.N - self.Nx)] = -self.expImp * alpha
                        self.BMat[rowIndex, rowIndex - (self.N - self.Nx)] = self.expImp * alpha

                # self.AMat[rowIndex,:] = [1+2*alpha*self.expImp if col==rowIndex
                # else -self.expImp*alpha if col==(rowIndex-1) else -self.expImp*alpha if col==(rowIndex+1)
                # else 0 for col in range(self.N)]
                # self.BMat[rowIndex,:] = [1-2*alpha*(1-self.expImp) if col==rowIndex
                # else alpha*(1-self.expImp) if col==(rowIndex-1) else alpha*(1-self.expImp) if col==(rowIndex+1)
                # else 0 for col in range(self.N)]

        print("AMat :\n" + str(self.AMat))
        print("\n\nBMat :\n" + str(self.BMat))

        self.CMat = np.matmul(np.linalg.inv(self.AMat), self.BMat)
        self.testSparse()

    def testSparse(self):
        sBMat = scipy.sparse.csr_matrix(self.BMat)
        sAMat = scipy.sparse.csr_matrix(self.AMat)
        sCMat = scipy.sparse.csr_matrix(self.CMat)

        if(self.constDiff is True):
            startSparse_inv = time.time()
            _ = sCMat.dot(self.prob)
            endSparse_inv = time.time()
            timeSparse = endSparse_inv - startSparse_inv

            startReg_inv = time.time()
            _ = np.linalg.solve(self.CMat, self.prob)
            endReg_inv = time.time()
            timeReg = endReg_inv - startReg_inv

            if(timeSparse < timeReg):
                if(self.output is True):
                    print("\t\tSparse matrix methods preferred...")
                self.sparTest = True
                self.AMat = sAMat
                self.BMat = sBMat
                self.CMat = sCMat
            else:
                if(self.output is True):
                    print("\t\tDense matrix methods preferred...")
                self.sparTest = False

        else:
            startSparse_full = time.time()
            sbVec = sBMat.dot(self.prob)
            _ = scipy.sparse.linalg.spsolve(sAMat, sbVec)
            endSparse_full = time.time()
            timeSparse = endSparse_full - startSparse_full

            startReg_full = time.time()
            bVec = np.matmul(self.BMat, self.prob)
            _ = np.linalg.solve(self.AMat, bVec)
            endReg_full = time.time()
            timeReg = endReg_full - startReg_full

            if(timeSparse < timeReg):
                if(self.output is True):
                    print("\t\tSparse matrix methods preferred...")
                self.sparTest = True
            else:
                if(self.output is True):
                    print("\t\tDense matrix methods preferred...")
                self.sparTest = False

    def integrate_step(
        self, forceParams: Tuple, forceFunction_x: Callable,
        forceFunction_y: Callable
    ):
        if(self.splitMethod == 'lie'):
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, self.dt
            )
            self.diffusionUpdate()

        elif(self.splitMethod == 'strang'):
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, 0.5 * self.dt
            )
            self.diffusionUpdate()
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, 0.5 * self.dt
            )

        elif(self.splitMethod == 'swss'):
            initProb = self.prob
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, self.dt
            )
            self.diffusionUpdate()
            prob_1 = self.prob
            self.prob = initProb
            self.diffusionUpdate()
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, self.dt
            )
            prob_2 = self.prob
            self.prob = 0.5*(prob_1 + prob_2)

        else:
            # Symmetric Strang splitting is the default choice
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, 0.5 * self.dt
            )
            self.diffusionUpdate()
            self.advectionUpdate(
                forceParams, forceFunction_x, forceFunction_y, 0.5 * self.dt
            )

    def diffusionUpdate(self):
        if(self.sparTest is True):
            if(self.constDiff is True):
                self.prob = self.CMat.dot(self.prob)
            else:
                bVec = self.BMat.dot(self.prob)
                self.prob = scipy.sparse.linalg.spsolve(self.AMat, bVec)
        else:
            if(self.constDiff is True):
                self.prob = np.matmul(self.CMat, self.prob)
            else:
                bVec = np.matmul(self.BMat, self.prob)
                self.prob = np.linalg.solve(self.AMat, bVec)

    def advectionUpdate(
        self, forceParams: Tuple, forceFunc_x: Callable, forceFunc_y: Callable,
        deltaT: float
    ):
        # if(self.adScheme=='lax-wendroff' or self.adScheme=='lw'):
        #    self.laxWendroff(forceParams,forceFunction,deltaT)
        # if(self.adScheme=='lax' or self.adScheme=='l'):
        #    self.lax(forceParams,forceFunction,deltaT)
        # if(self.adScheme=='lax-ds' or self.adScheme=='lds'):
        #    self.lax_dimSplit(forceParams,forceFunction,deltaT)
        # else:
        #   self.laxWendroff(forceParams,forceFunction,deltaT)
        #   self.lax_dimSplit(forceParams,forceFunction,deltaT)
        self.laxWendroff_lieSplit(forceFunc_x, forceFunc_y, forceParams, deltaT)

    def lax(self, forceParams: Tuple, forceFunction: Callable, deltaT: float):
        # TODO TEST LAX METHOD
        """
        This function updates the 2D probability density function using the 2-D lax-step method
        current implementation only uses PBCs
        """
        alpha = deltaT / (2 * self.dx)
        prob_mat = np.reshape(self.prob, (self.Nx, self.Ny))
        newProb = np.zeros_like(prob_mat)

        # First resolve 'bulk coordinates', '-2' counter means we want to evaluate boudary conditions separately
        for i in range(self.Nx - 2):
            for j in range(self.Ny - 2):
                # First calculate local average of NN probabilities
                newProb[i+1, j+1] = 0.25 * (
                    prob_mat[i+2, j+1] + prob_mat[i, j+1] + prob_mat[i+1, j+2] + prob_mat[i+1, j]
                )
                # Then calculate the forces
                newProb[i+1, j+1] -= alpha * (
                    forceFunction(self.xArray[i+2], self.yArray[j+1], forceParams, flag="x") * prob_mat[i+2, j+1] -
                    forceFunction(self.xArray[i], self.yArray[j+1], forceParams, flag="x") * prob_mat[i, j+1] +
                    forceFunction(self.xArray[i+1], self.yArray[j+2], forceParams, flag="y") * prob_mat[i+1, j+2] -
                    forceFunction(self.xArray[i+1], self.yArray[j], forceParams, flag="y") * prob_mat[i+1, j]
                )

        # Implement PBCs on boundary nodes
        # NOTE In this form, the 'corner boundaries' (where both x and y have simultaneous boudaries) are
        # double-counted, but this should only be ~2 lines per call
        for j in range(self.Ny):
            # Boundary conditions on 'x = 0' nodes
            newProb[0, j] = 0.25 * (
                prob_mat[1, j] + prob_mat[-1, j] + prob_mat[0, (j - 1) % self.Ny] + prob_mat[0, (j + 1) % self.Ny]
            )
            newProb[0, j] -= alpha * (
                forceFunction(self.xArray[1], self.yArray[j], forceParams, flag="x")
                * prob_mat[1, j]
                - forceFunction(self.xArray[-1], self.yArray[j], forceParams, flag="x")
                * prob_mat[-1, j]
                + forceFunction(self.xArray[0], self.yArray[(j + 1) % self.Ny], forceParams, flag="y")
                * prob_mat[0, (j + 1) % self.Ny]
                - forceFunction(self.xArray[0], self.yArray[(j - 1) % self.Ny], forceParams, flag="y")
                * prob_mat[0, (j - 1) % self.Ny]
            )

            # Boundary conditions on 'x = Nx' nodes
            newProb[-1, j] = 0.25 * (
                prob_mat[0, j] + prob_mat[-2, j] + prob_mat[-1, (j - 1) % self.Ny] + prob_mat[-1, (j + 1) % self.Ny]
            )
            newProb[-1, j] -= alpha * (
                forceFunction(self.xArray[0], self.yArray[j], forceParams, flag="x")
                * prob_mat[0, j]
                - forceFunction(self.xArray[-2], self.yArray[j], forceParams, flag="x")
                * prob_mat[-2, j]
                + forceFunction(self.xArray[-1], self.yArray[(j + 1) % self.Ny], forceParams, flag="y")
                * prob_mat[-1, (j + 1) % self.Ny]
                - forceFunction(self.xArray[-1], self.yArray[(j - 1) % self.Ny], forceParams, flag="y")
                * prob_mat[-1, (j - 1) % self.Ny]
            )

        for i in range(self.Nx):
            # Boundary conditions on 'y = 0' nodes
            newProb[i, 0] = 0.25 * (
                prob_mat[i, 1] + prob_mat[i, -1] + prob_mat[(i + 1) % self.Nx, 0] + prob_mat[(i - 1) % self.Nx, 0]
            )
            newProb[i, 0] -= alpha * (
                forceFunction(self.xArray[i], self.yArray[1], forceParams, flag="y")
                * prob_mat[i, 1]
                - forceFunction(self.xArray[i], self.yArray[-1], forceParams, flag="y")
                * prob_mat[i, -1]
                + forceFunction(self.xArray[(i + 1) % self.Nx], self.yArray[0], forceParams, flag="x")
                * prob_mat[(i + 1) % self.Nx, 0]
                - forceFunction(self.xArray[(i - 1) % self.Nx], self.yArray[0], forceParams, flag="x")
                * prob_mat[(i - 1) % self.Nx, 0]
            )

            # Boundary conditions on 'y = Ny' nodes
            newProb[i, -1] = 0.25 * (
                prob_mat[i, 0] + prob_mat[i, -2] + prob_mat[(i + 1) % self.Nx, -1] + prob_mat[(i - 1) % self.Nx, -1]
            )
            newProb[i, -1] -= alpha * (
                forceFunction(self.xArray[i], self.yArray[0], forceParams, flag="y")
                * prob_mat[i, 0]
                - forceFunction(self.xArray[i], self.yArray[-2], forceParams, flag="y")
                * prob_mat[i, -2]
                + forceFunction(self.xArray[(i + 1) % self.Nx], self.yArray[-1], forceParams, flag="x")
                * prob_mat[(i + 1) % self.Nx, -1]
                - forceFunction(self.yArray[(i - 1) % self.Nx], self.yArray[-1], forceParams, flag="x")
                * prob_mat[(i - 1) % self.Nx, -1]
            )

        # Flatten matrix probability and reform it into a single vector
        self.prob = np.reshape(newProb, (self.Nx * self.Ny))

    def lax_lieSplit(
        self, forceFunc_x: Callable, forceFunc_y: Callable, forceParams: Tuple,
        deltaT: float
    ):
        """
        This function implements a Lax update of the 2D probability distribution based on a Lie dimensional splitting
        """
        # First cast the probabbility to a matrix shape
        prob_mat = np.reshape(self.prob, (self.Nx, self.Ny))
        newProb = np.zeros_like(prob_mat)

        # tempdt = self.dt
        self.dt = deltaT

        par_x = forceParams[0]
        par_y = forceParams[1]

        # Now, update the X dimension using the Lax scheme
        for i in range(self.Nx - 1):
            for j in range(self.Ny):
                newProb[i+1, j] = (
                    0.5 * (prob_mat[i + 2, j] + prob_mat[i, j])
                    - (self.dt / (2 * self.dx)) * (
                        forceFunc_x(self.xArray[i + 2], self.yArray[j], par_x)
                        * prob_mat[i + 2, j]
                        - forceFunc_x(self.xArray[i], self.yArray[j], par_x)
                        * prob_mat[i, j]
                    )
                )

        # TODO: Implement hard wall and open BCS here (and for y-coordinate below)
        # Resolve periodic boundary conditions along the x=0 and x=N boundaries
        for j in range(self.Ny):
            newProb[0, j] = (
                0.5 * (prob_mat[1, j] + prob_mat[-1, j])
                - (self.dt / (2 * self.dx)) * (
                    forceFunc_x(self.xArray[1], self.yArray[j], par_x)
                    * prob_mat[1, j]
                    - forceFunc_x(self.xArray[-1], self.yArray[j], par_x)
                    * prob_mat[-1, j]
                )
            )
            newProb[-1, j] = (
                0.5 * (prob_mat[0, j] + prob_mat[-2, j])
                - (self.dt / (2 * self.dx)) * (
                    forceFunc_x(self.xArray[0], self.yArray[j], par_x)
                    * prob_mat[0, j]
                    - forceFunc_x(self.xArray[-2], self.yArray[j], par_x)
                    * prob_mat[-2, j]
                )
            )

        # Now, update the y-coordinate using the Lax scheme
        for i in range(self.Nx):
            for j in range(self.Ny - 1):
                prob_mat[i, j+1] = (
                    0.5 * (newProb[i, j+2] + newProb[i, j])
                    - (self.dt / (2 * self.dy)) * (
                        forceFunc_y(self.xArray[i], self.yArray[j+2], par_y)
                        * newProb[i, j+2]
                        - forceFunc_y(self.xArray[i], self.yArray[j], par_y)
                        * newProb[i, j]
                    )
                )

        # Resolve periodic boundary conditions along the y=0 and y=N boundaries
        for i in range(self.Ny):
            prob_mat[i, 0] = (
                0.5 * (newProb[i, 1] + newProb[i, -1])
                - (self.dt / (2 * self.dy)) * (
                    forceFunc_y(self.xArray[i], self.yArray[1], par_y)
                    * newProb[i, 1]
                    - forceFunc_y(self.xArray[i], self.yArray[-1], par_y)
                    * self.newProb[i, -1]
                )
            )

            prob_mat[i, -1] = (
                0.5 * (newProb[i, 0] + newProb[i, -2])
                - (self.dt / (2 * self.dy)) * (
                    forceFunc_y(self.xArray[i], self.yArray[0], par_y)
                    * newProb[i, 0]
                    - forceFunc_y(self.xArray[i], self.yArray[-2], par_y)
                    * self.newProb[i, -2]
                )
            )

        # Update the probability vector
        self.prob = np.reshape(prob_mat, self.Nx * self.Ny)

        # tempdt = self.dt
        self.dt = deltaT

    def laxWendroff_lieSplit(
        self, forceFunc_x: Callable, forceFunc_y: Callable, forceParams: Tuple,
        deltaT: float
    ):
        """
        This function performs an update on the advection term based on the
        Lax-Wendroff scheme for a 2D-diffusion, here the forceFunc_x and and
        ForceFunc_y are functions quantifying the force given a 2-D coordinate
        (x,y) and forceparams contains the individual parameters
        [[xParams],[yParams]] for the x and y force functions
        """

        # first shape probability into a matrix and declare two temporary matrices
        prob_mat = np.reshape(self.prob, (self.Nx, self.Ny))
        halfStep = np.zeros_like(prob_mat)
        probStar = np.zeros_like(prob_mat)

        tempdt = self.dt
        self.dt = deltaT

        par_x = forceParams[0]
        par_y = forceParams[1]

        # First calculate the x-update

        # Calculate half-step probabilities
        # NOTE: Updated modular calculation in arrays, should incorporate PBCS into this line
        # for i in range(self.Nx-1)
        for i in range(self.Nx):
            for j in range(self.Ny):
                halfStep[i, j] = (
                    0.5 * (prob_mat[(i + 1) % self.Nx, j] + prob_mat[i, j])
                    - (self.dt / (2 * self.dx)) * (
                        forceFunc_x(self.xArray[(i + 1) % self.Nx], self.yArray[j], par_x)
                        * prob_mat[(i + 1) % self.Nx, j]
                        - forceFunc_x(self.xArray[i], self.yArray[j], par_x)
                        * prob_mat[i, j]
                    )
                )

        # TODO: Implement hard wall and open boundary conditions for this
        # Resolve periodic boundaries for the half-step probabilities along the x=Nx boundary
        # for j in range(self.Ny):
        #    halfStep[-1,j] = 0.5*(prob_mat[0,j] + prob_mat[-1,j]) -
        # (self.dt/(2*self.dx))*(forceFunc_x(self.xArray[0],self.yArray[j],par_x)*prob_mat[0,j] -
        # forceFunc_x(self.xArray[-1],self.yArray[j],par_x)*prob_mat[-1,j])

        # Now update the probabilities in the x-direction based on the half-step fluxes
        # for i in range(self.Nx-1):
        for i in range(self.Nx):
            for j in range(self.Ny):
                probStar[(i + 1) % self.Nx, j] = (
                    prob_mat[(i + 1) % self.Nx, j]
                    - (self.dt / self.dx) * (
                        forceFunc_x(self.xArray[(i + 1) % self.Nx] + 0.5 * self.dx, self.yArray[j], par_x)
                        * halfStep[(i + 1) % self.Nx, j]
                        - forceFunc_x(self.xArray[i] + 0.5 * self.dx, self.yArray[j], par_x) * halfStep[i, j]
                    )
                )

        # And resolve the periodic boundary conditions on the x-updates
        # for j in range(self.Ny):
        #    probStar[0,j] = prob_mat[0,j] - (self.dt/self.dx)*(forceFunc_x(self.xArray[0]+
        # 0.5*self.dx,self.yArray[j],par_x)*halfStep[0,j] - forceFunc_x(self.xArray[-1]+0.5*self.dx,
        # self.yArray[j],par_x)*halfStep[-1,j])

        # Now perform the Y-updates on the probStar distribution
        # first calculate the half-step probabilities
        for i in range(self.Nx):
            # for j in range(self.Ny-1):
            for j in range(self.Ny):
                halfStep[i, j] = (
                    0.5 * (probStar[i, (j + 1) % self.Ny] + probStar[i, j])
                    - (self.dt / (2 * self.dy)) * (
                        forceFunc_y(self.xArray[i], self.yArray[(j + 1) % self.Ny], par_y)
                        * probStar[i, (j + 1) % self.Ny]
                        - forceFunc_y(self.xArray[i], self.yArray[j], par_y)
                        * probStar[i, j]
                    )
                )

        # Resolve the periodic boundaries for the half-step updates based on the y=Ny boundary
        # for i in range(self.Nx):
        #    halfStep[i,-1] = 0.5*(probStar[i,0] + probStar[i,-1]) - (self.dt/(2*self.dy))*
        # (forceFunc_y(self.xArray[i],self.yArray[0],par_y)*probStar[i,0] -
        # forceFunc_y(self.xArray[i],self.yArray[-1],par_y)*probStar[i,-1])

        # Now update the probability distribution based on the half-step fluxes
        for i in range(self.Nx):
            # for j in range(self.Ny-1):
            for j in range(self.Ny):
                prob_mat[i, (j + 1) % self.Ny] = (
                    probStar[i, (j + 1) % self.Ny]
                    - (self.dt / self.dy) * (
                        forceFunc_y(self.xArray[i], self.yArray[(j + 1) % self.Ny] + 0.5 * self.dy, par_y)
                        * halfStep[i, (j + 1) % self.Ny]
                        - forceFunc_y(self.xArray[i], self.yArray[j] + 0.5 * self.dy, par_y)
                        * halfStep[i, j]
                    )
                )

        # And resolve the boundary conditions for the full y-updates
        # for i in range(self.Nx):
        #    prob_mat[i,0] = probStar[i,0] - (self.dt/self.dy)*(forceFunc_y(self.xArray[i],
        # self.yArray[0]+0.5*self.dy,par_y)*halfStep[i,0] - forceFunc_y(self.xArray[i],
        # self.yArray[-1]+0.5*self.dy,par_y)*halfStep[i,-1])

        self.dt = tempdt
        # Update self.prob vector with result
        self.prob = np.reshape(prob_mat, self.Nx * self.Ny)
