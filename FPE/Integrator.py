'''
Filename: Integrator.py

This python script contains the routines used to solve the Fokker-Planck
equation numerically via a split-integrator scheme with a 2-step Lax Wendroff
method for the forcing term and a semi-implicit Crank-Nicolson scheme for the
diffusion matrix

Author:         Steven Large
Created:        August 25th 2019

Version: 2.0.0

Software:       python 3.x
'''

from typing import Callable, Optional, Tuple
import numpy as np
import scipy.sparse
import time
from FPE.base import BaseIntegrator


class FPE_Integrator_1D(BaseIntegrator):
    """1-dimensional Fokker-Planck integrator class, inherits from
    BaseIntegrator class
    """

    def __init__(
        self, D: float, dt: float, dx: float, xArray: np.ndarray,
        diffScheme: Optional[str] = 'crank-nicolson',
        adScheme: Optional[str] = 'lax-wendroff',
        boundaryCond: Optional[str] = 'hard-wall',
        splitMethod: Optional[str] = 'strang',
        output: Optional[bool] = False,
        constDiff: Optional[bool] = True
    ):
        """Constructor for 1-dimensional integrator object.

        Args:
            D (float): Diffusion coefficient for dynamics
            dt (float): Discrete time step size
            dx (float): Discrete spatial step size
            xArray (np.ndarray): Numpy array representing the discrete x-values
                of the system domain. 
            diffScheme (Optional[str], optional): Diffusion scheme used for
                integration. Defaults to 'crank-nicolson'.
            adScheme (Optional[str], optional): Advection scheme used for
                integration. Defaults to 'lax-wendroff'.
            boundaryCond (Optional[str], optional): Boundary condition
                specification. Defaults to 'hard-wall'.
            splitMethod (Optional[str], optional): Operator splitting method
                used. Defaults to 'strang'.
            output (Optional[bool], optional): Flag for whether or not output
                is logged to console. Defaults to False.
            constDiff (Optional[bool], optional): Whether or not the diffusion
                term is constant in the integration course (D is unchanging).
                Defaults to True.
        """
        super().__init__(
            D, dt, diffScheme, adScheme, boundaryCond, splitMethod, output,
            constDiff
        )
        self.dx = dx

        self.N = len(xArray)
        self.prob = np.ones(self.N) / (self.N * self.dx)
        self.xArray = xArray

        self.initDiffusionMatrix()

    @property
    def dimension(self):
        return 1

    @property
    def mean(self):
        return np.sum(self.prob * self.xArray * self.dx)

    @property
    def variance(self):
        return np.sum(((self.xArray - self.mean) ** 2) * self.prob * self.dx)

    def reset(
        self, variance: Optional[float] = None, mean: Optional[float] = None
    ):
        """Routine to reset probability vector to unbiform, and re-initialize
        physical trackers, if BOTH variance and mean are provided, then the
        probability will be reinitialized to a Gaussian distribution with the
        input variance ane mean
        
        Args:
            variance (Optional[float], optional): Input varaince for Gaussian
                distribution. Defaults to None.
            mean (Optional[float], optional): Input mean for Gaussian
                distribution. Defaults to None.
        """
        if variance is not None and mean is not None:
            self.initializeProbability(mean, variance)
        else:
            self.prob = np.ones(self.N) / (self.N * dx)
        self.initializePhysicalTrackers()

    def initializePhysicalTrackers(self):
        """Routine to (re)initialize physical quantity trackers.
        """
        # Work and power tracking arrays
        self.workAccumulator = 0
        self.workTracker = []
        self.powerTracker = []
        self.timeTracker = []

        self.flux = np.zeros(len(self.xArray))
        # Total (integrated) flux tracker
        self.fluxTracker = 0

    def initializeProbability(self, mean: float, var: float):
        """Initialize 1-D Gaussian probability density

        Args:
            mean (float): mean of distribution
            var (float): variance of distribution
        """
        self.prob = np.exp(-(0.5 / var) * ((self.xArray - mean)**2))
        self.prob = self.prob / (sum(self.prob) * self.dx)

    def initializeUserProbability(
        self, func: Callable, params: Optional[Tuple] = None
    ):
        """Initlialize a user-provided probability density, based on input
        function `func` that has the signature func(x, *params)

        Args:
            func (Callable): Function with the signature (x, *params) that
                returns a probability as a function of x
            params (Optional[Tuple], optional): Tuple of parameters for the
                input probability function. Defaults to None.
        """
        self.prob = func(self.xArray, *params)
        self.prob = self.prob / (sum(self.prob) * self.dx)

    def initDiffusionMatrix(self):
        """Routine to initialize the A and B diffusion matrices for diffusion
        integration
        """
        if(self.output):
            print("\n\nInitializing diffusion term integration matrix...\n")
        # Set parameters for diffusion matrix iteration
        super()._setDiffusionScheme()

        if(self.output):
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

    def _initializeBoundaryTerms(self, alpha: float):
        """Initialize boundary terms for diffusion matrices

        Args:
            alpha (float): _description_
        """
        # Left-side boundary
        self._matrixBoundary_A(alpha, 0)
        self._matrixBoundary_B(alpha, 0)

        # Right-side boundary
        self._matrixBoundary_A(alpha, self.N - 1)
        self._matrixBoundary_B(alpha, self.N - 1)

    def _matrixBoundary_A(self, alpha: float, idx: int):
        """Routine to set boudary-related terms in the diffusion matrix

        Args:
            alpha (float): coefficient for diffusion matrix terms, see
                documentation for definition
            idx (int): x-array index where

        Raises:
            ValueError: raised when self.BC parameter is invalid / not supported
        """
        # Periodic boundary condition resolution
        if(self.BC == "periodic"):
            self.AMat[idx, :] = [
                1 + 2 * alpha * self.expImp if col == idx
                else -self.expImp * alpha if col == (idx + 1) % self.N
                else -self.expImp * alpha if col == (idx - 1) % self.N
                else 0 for col in range(self.N)
            ]

        # Open boundary condition resolution
        elif(self.BC == "open"):
            self.AMat[idx, :] = [
                1 + 2 * alpha * self.expImp if col == idx
                else -self.expImp * alpha if col == abs(idx - 1)
                else 0 for col in range(self.N)
            ]

        # Hard-wall boundary condition resolution
        elif(self.BC == "hard-wall"):
            self.AMat[idx, :] = [
                1 + 2 * alpha if col == idx
                else -2 * alpha if col == abs(idx - 1)
                else 0 for col in range(self.N)
            ]

        else:
            raise ValueError(
                f"Invalid boundary condition: {self.BC}, cannot resolve "
                "diffusion matrix A"
            )

    def _matrixBoundary_B(self, alpha: float, idx: int):
        """Determines / sets the parameters of matrix B on the boundaries

        Args:
            alpha (float): coefficient for diffusion matrix terms, see
                documentation for definition
            idx (int): x-array index where

        Raises:
            ValueError: raised when self.BC parameter is invalid / not supported
        """
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

        elif(self.BC == "hard-wall"):
            self.BMat[idx, :] = [
                1 if col == idx
                else 0 for col in range(self.N)
            ]

        else:
            raise ValueError(
                f"Invalid Boudary condition {self.BC}, cannot resolve"
                "diffusion matrix B"
            )

    def work_step(
        self, forceParams_pre: Tuple, forceParams_post: Tuple,
        forceFunction: Callable, energyFunction: Callable
    ):
        """Wrapper routine for integration step to calculate the work done due
        to changes in force function parameters. In this scheme, we effectively
        implement the Sekimoto definition of work, which splits work and heat
        into two steps: work occurs when system control parameters change,
        and dissipation follows afterwards. Thus, work is defined using an Ito
        discretization (changes occur at the start of the time step) while
        heat dissiaption occurs afterwards.  Thus, the force parameters here
        are used to calculate work, and then the system responds to the values
        of the force based on `forceParams_post`.

        Args:
            forceParams_pre (Tuple): Force parameters from before (pre) update
            forceParams_post (Tuple): Force parameters form after (post) update
            forceFunction (Callable): Force function
            energyFunction (Callable): Energy function that the forceFunction
                is derived from 
        """

        # Calculate average energy before and after update to force parameters
        currEnergy = (
            sum(energyFunction(self.xArray, forceParams_pre) * self.prob) * self.dx
        )
        newEnergy = (
            sum(energyFunction(self.xArray, forceParams_post) * self.prob) * self.dx
        )

        self.integrate_step(forceParams_post, forceFunction)

        # Work done per step is given by the change in average energy through
        # force function update
        self.workAccumulator += newEnergy - currEnergy
        self.workTracker.append(self.workAccumulator)
        # Power is the change in work, divided by the time over which the change
        # occurred (which is dt here)
        self.powerTracker.append((newEnergy - currEnergy) / self.dt)

    def flux_step(self, forceParams: Tuple, forceFunction: Callable):
        """Similar to work_step(...) this routine wraps the integrate_step in
        the parent class with calcualtions for the probability flux. Here
        the flux J is defined in the alternate version of the FPE:

            partial_t p = - partial_x J

        And so the flux can be calculated relatively simply by te current
        values of the probability, as well as the force parameters

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        # flux as a function of position
        self.flux = (
            self.D * forceFunction(self.xArray, forceParams) * self.prob
            - self.D * np.gradient(self.prob)
        )
        # Calculate integrated (net) flux over current configuration
        self.fluxTracker = sum(self.flux) * self.dx
        self.integrate_step(forceParams, forceFunction)

    def laxWendroff(
        self, forceParams: Tuple, forceFunction: Callable, deltaT: float
    ):
        """Implementation of lax-wendroff method for 1-D system

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
            deltaT (float): Size of time step (which is not necessarily equal
                to the self.dt parameter because of the operator splittings)
        """
        # Initialize empty array for new probability
        new_prob = np.zeros(len(self.prob))

        # Get half-time-step flux terms
        halfFlux = self._calcFlux_laxWendroff(
            forceParams, forceFunction, deltaT
        )

        # Update probability with Lax-step using half-time-step flux terms
        for index in range(len(self.prob)):
            new_prob[index] = (
                self.prob[index]
                - (self.D * deltaT / self.dx)
                * (halfFlux[index+1] - halfFlux[index])
            )

        # set probability to be update prob
        self.prob = new_prob

    def _calcFlux_laxWendroff(
        self, forceParams: Tuple, forceFunction: Callable, deltaT: float
    ) -> np.ndarray:
        """Routine to calculate the half-time-step flux terms for the
        lax-wendroff advection update scheme

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
            deltaT (float): size of temporal discretization (dt) for
                integration step

        Returns:
            np.ndarray: half-time-step flux terms, at positions x_min = -1/2 dx
            up to x_max = (N + 1/2) dx
        """

        # initialize half-step probability and half-step flux terms
        half_prob = np.zeros(len(self.prob) + 1)
        half_flux = np.zeros(len(self.prob) + 1)

        # Bulk terms
        for i in range(len(self.prob) - 1):

            half_prob[i + 1] = (
                0.5 * (self.prob[i + 1] + self.prob[i])
                - self._getFluxDiff_LaxWendroff(forceFunction, forceParams, deltaT, i)
            )
            half_flux[i + 1] = (
                forceFunction(self.xArray[i] + 0.5 * self.dx, forceParams) * half_prob[i + 1]
            )

        # Boundary terms
        if self.BC == 'periodic':
            half_prob[0] = (
                0.5 * (self.prob[0] + self.prob[-1])
                - self._getFluxDiff_LaxWendroff(
                    forceFunction, forceParams, deltaT, len(self.prob) - 1
                )
            )

            half_prob[-1] = (
                0.5 * (self.prob[-1] + self.prob[0])
                - self._getFluxDiff_LaxWendroff(forceFunction, forceParams, deltaT, 0)
            )
 
            half_flux[0] = (
                forceFunction(self.xArray[0] - 0.5 * self.dx, forceParams) * half_prob[0]
            )
            half_flux[-1] = half_flux[0]

        elif self.BC == "open":

            fluxFw = (
                (self.D * deltaT / (2 * self.dx))
                * forceFunction(self.xArray[0], forceParams)
                * self.prob[0]
            )
            fluxRev = (
                (self.D * deltaT / (2 * self.dx))
                * forceFunction(self.xArray[-1], forceParams)
                * self.prob[-1]
            )
            half_prob[0] = 0.5 * self.prob[0] - fluxFw
            half_prob[-1] = 0.5 * self.prob[-1] + fluxRev
            half_flux[0] = forceFunction(self.xArray[0] - 0.5 * self.dx, forceParams) * half_prob[0]
            half_flux[-1] = forceFunction(self.xArray[-1] + 0.5 * self.dx, forceParams) * half_prob[-1]

        else:
            # Hard wall boundaries
            half_prob[0] = 0.5 * self.prob[0]
            half_prob[-1] = 0.5 * self.prob[-1]
            half_flux[0] = forceFunction(self.xArray[0] - 0.5 * self.dx, forceParams) * half_prob[0]
            half_flux[-1] = forceFunction(self.xArray[0] - 0.5 * self.dx, forceParams) * half_prob[-1]

        return half_flux

    def _getFluxDiff_LaxWendroff(
        self, forceFunction: Callable, forceParams: Tuple, deltaT: float,
        idx: int
    ) -> float:
        """Calculate the difference between forward and reverse flux terms at
        half-step points within the LW method

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
            deltaT (float): Discrete time step size for integration step
            idx (int): Index for half-step update (which is i - 1/2 in terms of
                the original spatial mesh)

        Returns:
            float: Flux difference between i-1/2 and i+1/2 in spatial
                discretization
        """
        fluxFw = (
            (self.D * deltaT / (2 * self.dx))
            * forceFunction(self.xArray[(idx + 1) % len(self.xArray)], forceParams)
            * self.prob[(idx + 1) % len(self.xArray)]
        )
        fluxRev = (
            (self.D * deltaT / (2 * self.dx))
            * forceFunction(self.xArray[idx], forceParams)
            * self.prob[idx]
        )

        return fluxFw - fluxRev


# ANCHOR 2D Integrator
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
        # sCMat = scipy.sparse.csr_matrix(self.CMat)

        if(self.constDiff is True):
            # NOTE after benchmarking, for constant diffusion coefficients
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


if __name__ == "__main__":
    D = 1
    dx = 0.01
    dt = 0.01
    x_array = np.arange(-1, 1, dx)
    fpe = FPE_Integrator_1D(D, dt, dx, x_array)

    print("-- DONE --")
