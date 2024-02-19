# Base class for FPE integrator

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np
import time
import scipy.sparse

from fokker_planck.types.basetypes import DiffScheme, BoundaryCondition, SplitMethod


class Integrator(ABC):
    """Base class for integrator objects (abstract), all instances of FPE
    integrators must inherit from this base class. Universal functions are
    defined and implemented here, while instance-specific methods are
    relegated to the inherited classes

    Raises:
        ValueError: _description_
        NotImplementedError: _description_
        ValueError: _description_
        ValueError: _description_
    """
    def __init__(
        self, D: float, dt: float, diff_scheme: DiffScheme,
        boundary_cond: BoundaryCondition, split_method: SplitMethod,
        output: bool = False, const_diffusion: bool = True
    ):
        """Constructor for base integrator class

        Args:
            D (float): diffusion coefficient for dynaimcs (assumed to be constant)
            dt (float): Discrete ime step size
            diffScheme (str): Integration scehem used for diffusion step: can
                be one of `explicit`, `implicit` or `crank-nicolson`, default
                is to use `crank-nicolson`
            adScheme (str): INtegration scheme used for advection step, for now
                this is a passthrough method as only the 2-step lax-wendroff
                method is implemented
            boundaryCond (str): Boundary condition to use on the domain, can be
                one of `hard-wall`, `periodic`, or 'open', default is to use
                `hard-wall`
            splitMethod (str): Operator splitting method used to segment the 
                advection and diffusiojn integration steps. Default is symmetric
                Strang splitting
            output (Optional[bool], optional): Whether to show internal output
                messages during execution. Defaults to False.
            constDiff (Optional[bool], optional): Whether or not the diffusion
                coefficient will be constant through the course of simulation.
                Defaults to True.
        """
        # Validate input parameters
        self._validate_input(
            diff_scheme, boundary_cond, split_method
        )

        # Initialize instance variables. For now, this routine supports
        # constant diffusion coefficient integrations (i.e. D is not a tensor)
        self.D = D
        self.dt = dt

        # Condition for satisfying CFL criterion
        self.CFL = None

        # Instantiate the instance directives
        self.diff_scheme = diff_scheme    # Integration scheme for diffusion term
        self.BC = boundary_cond          # Boundary conditions
        self.split_method = split_method  # Integator splitting method
        self.const_diffusion = const_diffusion      # Flag for if diffusion matrix is const

        # Flag for whether or not output will be printed
        self.output = output

        # Default initial argument for sparse method utilization in matrix
        # operations
        self.sparse_calc = False

        # Physics trackers
        self.work_accumulator = 0
        self.work_tracker = []
        self.power_tracker = []

    def _validate_input(
        self, diff_scheme: DiffScheme, boundary_cond: BoundaryCondition,
        split_method: SplitMethod
    ):
        """Validation method to handle input parameters. In this method the
        checks are for whether or not the input diffusion scheme, advection
        scheme and boundary conditions are supported, by seeing if the keys
        are present in the config.py file

        Args:
            diffScheme (str): Diffusion scheme identifier
            adScheme (str): Advection scheme identifier
            boundaryCond (str): Boundary conditions
            splitMethod (str): Operator splitting method

        Raises:
            ValueError: Diffusion scheme not recognized
            NotImplementedError: Advection scheme not recognized
            ValueError: boundary condition not recognized
            ValueError: Operator splitting scheme not recognized
        """
        if not isinstance(diff_scheme, DiffScheme):
            raise ValueError("diffScheme not recognized, see config file")

        if not isinstance(boundary_cond, BoundaryCondition):
            raise ValueError(
                "boundary condition not recognized, see config file"
            )

        if not isinstance(split_method, SplitMethod):
            raise ValueError(
                "Splitting method not recognized, see config file"
            )

    def _setDiffusionScheme(self):
        """Method to set the diffusion scheme in the integrator instance. In
        all cases there are three options: implicit, explicit, and
        crank-nicolson. The latter is the default choice.
        """
        if (self.diff_scheme == DiffScheme.Implicit):
            self.exp_imp = 1.0

        elif (self.diff_scheme == DiffScheme.Explicit):
            self.exp_imp = 0.0

        elif (self.diff_scheme == DiffScheme.CrankNicolson):
            self.exp_imp = 0.5

        else:
            if (self.output):
                print("\t\tInt scheme unknown, using default setting (CN)...")
            self.exp_imp = 0.5

    def test_sparse(self):
        """Routine to check whether sparse matrix calculations are more rapid
        than standard matrix multiplication techniques. This is expected to be
        the case for large matrices, and more impactful when diffisuon is not
        constant. For constant diffusion term, the update due to diffusion is
        only a matrix multiplication (and not a matrix inversion) so the
        scaling is better with N.
        """
        if (self.const_diffusion):
            # NOTE Implement checks here
            self.spase_calc = False

        else:
            sBMat = scipy.sparse.csr_matrix(self.BMat)
            sAMat = scipy.sparse.csr_matrix(self.AMat)
            self._test_sparse(sAMat, sBMat)

    def _test_sparse(
        self, sAMat: scipy.sparse.csr_matrix,
        sBMat: scipy.sparse.csr_matrix
    ):
        """Method to check whether full diffusion update (matrixc inversion) is
        faster when sparse matrix data structures are used, as opposed to
        standard matrix formats.

        Args:
            sAMat (scipy.sparse.csr_matrix): Sparse formatted verison of
                instance variable AMat
            sBMat (scipy.sparse.csr_matrix): Sparse matrix formatted version of
                instance variable BMat
        """
        # Start timer
        start_sparse_full = time.time()
        sbVec = sBMat.dot(self.prob)
        _ = scipy.sparse.linalg.spsolve(sAMat, sbVec)
        # Stop timer
        end_sparse_full = time.time()
        time_sparse = end_sparse_full - start_sparse_full

        # Start timer (non-sparse calculations)
        start_reg_full = time.time()
        bVec = np.matmul(self.BMat, self.prob)
        _ = np.linalg.solve(self.AMat, bVec)
        # stop timer
        end_reg_full = time.time()
        time_reg = end_reg_full - start_reg_full

        # If sparse calculation is faster, use sparse matrix methods
        if (time_sparse < time_reg):
            if (self.output):
                print("\t\tSparse matrix methods preferred...")
            self.sparse_calc = True
            self.BMat = sBMat
            self.AMat = sAMat

        # Otherwise, use standard dense matrix calculation techniques
        else:
            if (self.output):
                print("\t\tDense matrix methods preferred...")
            self.sparse_calc = False

    def integrate_step_advection(
        self, force_params: Tuple, force_function: Callable
    ):
        """Method to indtegrate a step of ONLY the advection term (so no
        diffusion). This cannot be done, simply, by setting D = 0 as terms in
        the advection equation depend on D (because the dynamics of the system
        are governed by D in the FPE).

        Args:
            forceParams (Tuple): Parameters for the force function
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        self.advection_update(force_params, force_function, self.dt)

    def diffusion_update(self):
        """Update to the system probability based on the diffusion dynamics
        """
        if (self.sparse_calc):
            bVec = self.BMat.dot(self.prob)
            self.prob = scipy.sparse.linalg.spsolve(self.AMat, bVec)
        else:
            if(self.const_diffusion):
                self.prob = np.matmul(self.CMat, self.prob)
            else:
                bVec = np.matmul(self.BMat, self.prob)
                self.prob = np.linalg.solve(self.AMat, bVec)

    def advection_update(
        self, force_params: Tuple, force_function: Callable, delta_t: float
    ):
        """Update the advective component of the equation

        Args:
            forceParams (Tuple): Parameters for force function
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
            deltaT (float): Time step for dynamics
        """
        # Currently, this only directs towards the LW routine, but supports
        # the flexibility to add others in the future
        self.lax_wendroff(force_params, force_function, delta_t)

    def integrate_step(self, force_params: Tuple, force_function: Callable):
        """Integrates the equations by one time step. In this package, operator
        splitting methods are used, and this interface routine directs the update
        step towards the correct endpoint: Strang, Lie, ot SWSS

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        if (self.split_method is SplitMethod.Lie):
            self._integrate_step_lie(force_params, force_function)

        elif (self.split_method is SplitMethod.Strang):
            self._integrate_step_strang(force_params, force_function)

        elif (self.split_method is SplitMethod.SymStrang):
            self._integrate_step_SWSS(force_params, force_function)
        else:
            self._integrate_step_strang(force_params, force_function)

        self.normalize_prob()

    def _integrate_step_lie(self, force_params: Tuple, force_function: Callable):
        """Implementation of Lie splitting update of FPE

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        self.advection_update(force_params, force_function, self.dt)
        self.diffusion_update()

    def _integrate_step_strang(
        self, force_params: Tuple, force_function: Callable
    ):
        """Implementation of (symmetric) Strang Splitting update of FPE

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        self.advection_update(force_params, force_function, 0.5 * self.dt)
        self.diffusion_update()
        self.advection_update(force_params, force_function, 0.5 * self.dt)

    def _integrate_step_SWSS(
        self, force_params: Tuple, force_function: Callable
    ):
        """Implementation of the Symmetrically weighted Strang splitting update
        of the FRE.

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        initProb = self.prob
        self.advection_update(force_params, force_function, self.dt)
        self.diffusion_update()
        prob_1 = self.prob
        self.prob = initProb
        self.diffusion_update()
        self.advection_update(force_params, force_function, self.dt)
        prob_2 = self.prob
        self.prob = 0.5 * (prob_1 + prob_2)

    @property
    def get_prob(self) -> np.ndarray:
        return self.prob

    @abstractmethod
    def init_physical_trackers(self):
        pass

    @abstractmethod
    def init_diffusion_matrix(self):
        pass

    @abstractmethod
    def check_CFL(self, force_params: Tuple, force_function: Callable) -> bool:
        pass

    @abstractmethod
    def lax_wendroff(self, *args, **kwargs):
        pass

    @abstractmethod
    def normalize_prob(self):
        pass
