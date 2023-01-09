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
    def dimension(self) -> int:
        return 1

    @property
    def mean(self) -> float:
        return np.sum(self.prob * self.xArray * self.dx)

    @property
    def variance(self) -> float:
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
            self.prob = np.ones(self.N) / (self.N * self.dx)
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
        self._setDiffusionScheme()

        if(self.output):
            print("\t\tInitializing integration matrices for diffusion\n")

        alpha = self.D * self.dt / (self.dx * self.dx)

        self.AMat = (
            np.diag(1 + 2 * alpha * self.expImp * np.ones(self.N))
            - np.diag(alpha * self.expImp * np.ones(self.N - 1), k=1)
            - np.diag(alpha * self.expImp * np.ones(self.N - 1), k=-1)
        )

        self.BMat = (
            np.diag(1 - 2 * alpha * (1 - self.expImp) * np.ones(self.N))
            + alpha * np.diag((1 - self.expImp) * np.ones(self.N - 1), k=1)
            + alpha * np.diag((1 - self.expImp) * np.ones(self.N - 1), k=-1)
        )

        # Initialize boundary columns based on self.BC
        self._initializeBoundaryTerms(alpha)

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
        if self.BC == 'periodic':
            self.AMat[idx, (idx + 1) % self.N] = -self.expImp * alpha
            self.AMat[idx, (idx - 1) % self.N] = -self.expImp * alpha

        # NOTE double check this boundary resolution..
        elif self.BC == 'hard-wall':
            self.AMat[idx, idx] = 1 + 2 * alpha
            self.AMat[idx, abs(idx - 1)] = -2 * alpha

        elif self.BC == "open":
            pass

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
        if self.BC == 'periodic':
            self.BMat[idx, (idx + 1) % self.N] = alpha * (1 - self.expImp)
            self.BMat[idx, (idx - 1) % self.N] = alpha * (1 - self.expImp)

        # NOTE double check this boundary resolution..
        elif self.BC == 'hard-wall':
            self.BMat[idx, idx] = 1
            self.BMat[idx, abs(idx - 1)] = 0

        elif self.BC == "open":
            pass

        else:
            raise ValueError(
                f"Invalid boundary condition: {self.BC}, cannot resolve "
                "diffusion matrix A"
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


if __name__ == "__main__":
    D = 1
    dx = 0.01
    dt = 0.01
    x_array = np.arange(-1, 1, dx)
    fpe = FPE_Integrator_1D(D, dt, dx, x_array)

    print("-- DONE --")
