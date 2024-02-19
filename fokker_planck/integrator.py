'''
Filename: Integrator.py

This python script contains the routines used to solve the Fokker-Planck
equation numerically via a split-integrator scheme with a 2-step Lax Wendroff
method for the forcing term and a semi-implicit Crank-Nicolson scheme for the
diffusion matrix

Author:         Steven Large
Created:        August 25th 2019

Software:       python 3.x
'''
from typing import Callable, Optional, Tuple
import numpy as np
from fokker_planck.base import Integrator
from fokker_planck.types.basetypes import (
    DiffScheme, BoundaryCondition, SplitMethod
)


class FokkerPlanck1D(Integrator):
    """1-dimensional Fokker-Planck integrator class, inherits from
    BaseIntegrator class
    """
    def __init__(
        self, D: float, dt: float, dx: float, x_array: np.ndarray,
        diff_scheme: DiffScheme = DiffScheme.CrankNicolson,
        boundary_cond: BoundaryCondition = BoundaryCondition.HardWall,
        split_method: SplitMethod = SplitMethod.Strang,
        output: bool = False,
        const_diffusion: bool = True
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
            D, dt, diff_scheme, boundary_cond, split_method, output,
            const_diffusion
        )
        self.dx = dx

        self.N = len(x_array)
        self.prob = np.ones(self.N) / (self.N * self.dx)
        self.x_array = x_array

        self.init_diffusion_matrix()

    @property
    def dimension(self) -> int:
        return 1

    @property
    def mean(self) -> float:
        return np.sum(self.prob * self.x_array * self.dx)

    @property
    def variance(self) -> float:
        return np.sum(((self.x_array - self.mean) ** 2) * self.prob * self.dx)

    def normalize_prob(self):
        """
        Normalize the probability distribution.

        This method normalizes the probability distribution by dividing each
        element by the sum of all elements multiplied by the step size.

        Parameters:
            None

        Returns:
            None
        """
        self.prob = self.prob / (sum(self.prob) * self.dx)

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
            self.initialize_probability(mean, variance)
        else:
            self.prob = np.ones(self.N) / (self.N * self.dx)
        self.init_physical_trackers()

    def init_physical_trackers(self):
        """Routine to (re)initialize physical quantity trackers.
        """
        # Work and power tracking arrays
        self.workAccumulator = 0
        self.workTracker = []
        self.powerTracker = []
        self.timeTracker = []

        self.flux = np.zeros(len(self.x_array))
        # Total (integrated) flux tracker
        self.fluxTracker = 0

    def initialize_probability(self, mean: float, var: float):
        """Initialize 1-D Gaussian probability density

        Args:
            mean (float): mean of distribution
            var (float): variance of distribution
        """
        self.prob = np.exp(-(0.5 / var) * ((self.x_array - mean)**2))
        self.prob = self.prob / (sum(self.prob) * self.dx)

    def initialize_user_probability(
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
        self.prob = func(self.x_array, *params)
        self.prob = self.prob / (sum(self.prob) * self.dx)

    def check_CFL(self, force_params: Tuple, force_function: Callable) -> bool:
        """Routine to check whether or not Courant-Friedrichs-Lewy (CFL)
        criterion is satisfied for dynamics, given input force

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position

        Returns:
            bool: Flag as to whether or not the force function, parameters, and
                discretization evel satisfy the CFL
        """
        max_force = np.max(np.abs(force_function(self.x_array, force_params)))

        # set CFL and check if it is less than or greater than Unity
        self.CFL = max_force * self.dt / self.dx
        if (self.CFL > 1):
            if self.output:
                print("\t\tStability warning, invalid CFL --> " + str(self.CFL) + "\n\n")
            return False

        if self.output:
            print("\t\tCFL criterion satisfied, CFL --> " + str(self.CFL) + "\n\n")
        return True

    def init_diffusion_matrix(self):
            """Routine to initialize the A and B diffusion matrices for diffusion
            integration
            
            This method initializes the A and B diffusion matrices used for diffusion integration.
            It sets the parameters for diffusion matrix iteration and calculates the values of the matrices.
            The A matrix is calculated using the formula:
            A = diag(1 + 2 * alpha * exp_imp * ones(N)) - diag(alpha * exp_imp * ones(N - 1), k=1) - diag(alpha * exp_imp * ones(N - 1), k=-1)
            The B matrix is calculated using the formula:
            B = diag(1 - 2 * alpha * (1 - exp_imp) * ones(N)) + alpha * diag((1 - exp_imp) * ones(N - 1), k=1) + alpha * diag((1 - exp_imp) * ones(N - 1), k=-1)
            The boundary columns of the matrices are initialized based on the boundary conditions.
            The C matrix is calculated as the matrix product of the inverse of A and B.
            The method also tests if sparse-matrix iteration steps are faster than normal matrix multiplication.
            """
            # Set parameters for diffusion matrix iteration
            self._setDiffusionScheme()

            alpha = self.D * self.dt / (self.dx * self.dx)

            self.AMat = (
                np.diag(1 + 2 * alpha * self.exp_imp * np.ones(self.N))
                - np.diag(alpha * self.exp_imp * np.ones(self.N - 1), k=1)
                - np.diag(alpha * self.exp_imp * np.ones(self.N - 1), k=-1)
            )

            self.BMat = (
                np.diag(1 - 2 * alpha * (1 - self.exp_imp) * np.ones(self.N))
                + alpha * np.diag((1 - self.exp_imp) * np.ones(self.N - 1), k=1)
                + alpha * np.diag((1 - self.exp_imp) * np.ones(self.N - 1), k=-1)
            )

            # Initialize boundary columns based on self.BC
            self._initialize_boundary_terms(alpha)

            self.CMat = np.matmul(np.linalg.inv(self.AMat), self.BMat)

            # Test if sparse-matrix iteration steps are faster than normal matrix
            # multiplication
            self.test_sparse()

    def _initialize_boundary_terms(self, alpha: float):
        """Initialize boundary terms for diffusion matrices

        Args:
            alpha (float): _description_
        """
        # Left-side boundary
        self._matrix_boundary_A(alpha, 0)
        self._matrix_boundary_B(alpha, 0)

        # Right-side boundary
        self._matrix_boundary_A(alpha, self.N - 1)
        self._matrix_boundary_B(alpha, self.N - 1)

    def _matrix_boundary_A(self, alpha: float, idx: int):
        """Routine to set boudary-related terms in the diffusion matrix

        Args:
            alpha (float): coefficient for diffusion matrix terms, see
                documentation for definition
            idx (int): x-array index where

        Raises:
            ValueError: raised when self.BC parameter is invalid / not supported
        """
        if self.BC == BoundaryCondition.Periodic:
            self.AMat[idx, (idx + 1) % self.N] = -self.exp_imp * alpha
            self.AMat[idx, (idx - 1) % self.N] = -self.exp_imp * alpha

        elif self.BC == BoundaryCondition.HardWall:
            self.AMat[idx, idx] = 1 + 2 * alpha
            if idx == 0:
                self.AMat[idx, idx + 1] = -2 * alpha
            elif idx == self.N - 1:
                self.AMat[idx, idx - 1] = -2 * alpha

        elif self.BC == BoundaryCondition.Open:
            pass

        else:
            raise ValueError(
                f"Invalid boundary condition: {self.BC}, cannot resolve "
                "diffusion matrix A"
            )

    def _matrix_boundary_B(self, alpha: float, idx: int):
        """Determines / sets the parameters of matrix B on the boundaries

        Args:
            alpha (float): coefficient for diffusion matrix terms, see
                documentation for definition
            idx (int): x-array index where

        Raises:
            ValueError: raised when self.BC parameter is invalid / not supported
        """
        if self.BC == BoundaryCondition.Periodic:
            self.BMat[idx, (idx + 1) % self.N] = alpha * (1 - self.exp_imp)
            self.BMat[idx, (idx - 1) % self.N] = alpha * (1 - self.exp_imp)

        elif self.BC == BoundaryCondition.HardWall:
            self.BMat[idx, idx] = 1
            if idx == 0:
                self.BMat[idx, idx + 1] = 0
            elif idx == self.N - 1:
                self.BMat[idx, idx - 1] = 0

        elif self.BC == BoundaryCondition.Open:
            pass

        else:
            raise ValueError(
                f"Invalid boundary condition: {self.BC}, cannot resolve "
                "diffusion matrix A"
            )

    def work_step(
        self, force_params_pre: Tuple, force_params_post: Tuple,
        force_function: Callable, energy_function: Callable
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
            sum(energy_function(self.x_array, force_params_pre) * self.prob) * self.dx
        )
        newEnergy = (
            sum(energy_function(self.x_array, force_params_post) * self.prob) * self.dx
        )

        self.integrate_step(force_params_post, force_function)

        # Work done per step is given by the change in average energy through
        # force function update
        self.work_accumulator += newEnergy - currEnergy
        self.work_tracker.append(self.work_accumulator)
        # Power is the change in work, divided by the time over which the change
        # occurred (which is dt here)
        self.power_tracker.append((newEnergy - currEnergy) / self.dt)

    def flux_step(self, force_params: Tuple, force_function: Callable):
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
            self.D * force_function(self.x_array, force_params) * self.prob
            - self.D * np.gradient(self.prob)
        )
        # Calculate integrated (net) flux over current configuration
        self.flux_tracker = sum(self.flux) * self.dx
        self.integrate_step(force_params, force_function)

    def lax_wendroff(
        self, force_params: Tuple, force_function: Callable, delta_t: float
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
        halfFlux = self._calc_flux_lax_wendroff(
            force_params, force_function, delta_t
        )

        # Update probability with Lax-step using half-time-step flux terms
        for index in range(len(self.prob)):
            new_prob[index] = (
                self.prob[index]
                - (self.D * delta_t / self.dx)
                * (halfFlux[index+1] - halfFlux[index])
            )

        # set probability to be update prob
        self.prob = new_prob

    def _calc_flux_lax_wendroff(
        self, force_params: Tuple, force_function: Callable, delta_t: float
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
                - self._get_flux_diff_lax_wendroff(force_function, force_params, delta_t, i)
            )
            half_flux[i + 1] = (
                force_function(self.x_array[i] + 0.5 * self.dx, force_params) * half_prob[i + 1]
            )

        # Boundary terms
        if self.BC == BoundaryCondition.Periodic:
            half_prob[0] = (
                0.5 * (self.prob[0] + self.prob[-1])
                - self._get_flux_diff_lax_wendroff(
                    force_function, force_params, delta_t, len(self.prob) - 1
                )
            )

            half_prob[-1] = (
                0.5 * (self.prob[-1] + self.prob[0])
                - self._get_flux_diff_lax_wendroff(force_function, force_params, delta_t, 0)
            )
 
            half_flux[0] = (
                force_function(self.x_array[0] - 0.5 * self.dx, force_params) * half_prob[0]
            )
            half_flux[-1] = half_flux[0]

        elif self.BC == BoundaryCondition.Open:

            fluxFw = (
                (self.D * delta_t / (2 * self.dx))
                * force_function(self.x_array[0], force_params)
                * self.prob[0]
            )
            fluxRev = (
                (self.D * delta_t / (2 * self.dx))
                * force_function(self.x_array[-1], force_params)
                * self.prob[-1]
            )
            half_prob[0] = 0.5 * self.prob[0] - fluxFw
            half_prob[-1] = 0.5 * self.prob[-1] + fluxRev
            half_flux[0] = force_function(self.x_array[0] - 0.5 * self.dx, force_params) * half_prob[0]
            half_flux[-1] = force_function(self.x_array[-1] + 0.5 * self.dx, force_params) * half_prob[-1]

        else:
            # Hard wall boundaries
            half_prob[0] = 0.5 * self.prob[0]
            half_prob[-1] = 0.5 * self.prob[-1]
            half_flux[0] = force_function(self.x_array[0] - 0.5 * self.dx, force_params) * half_prob[0]
            half_flux[-1] = force_function(self.x_array[-1] + 0.5 * self.dx, force_params) * half_prob[-1]

        return half_flux

    def _get_flux_diff_lax_wendroff(
        self, force_function: Callable, force_params: Tuple, delta_t: float,
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
            (self.D * delta_t / (2 * self.dx))
            * force_function(self.x_array[(idx + 1) % len(self.x_array)], force_params)
            * self.prob[(idx + 1) % len(self.x_array)]
        )
        fluxRev = (
            (self.D * delta_t / (2 * self.dx))
            * force_function(self.x_array[idx], force_params)
            * self.prob[idx]
        )

        return fluxFw - fluxRev


if __name__ == "__main__":
    D = 1
    dx = 0.01
    dt = 0.01
    x_array = np.arange(-1, 1, dx)
    fpe = FokkerPlanck1D(D, dt, dx, x_array)

    print("-- DONE --")
