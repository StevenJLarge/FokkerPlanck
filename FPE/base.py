# Base class for FPE integrator

from abc import ABCMeta, abstractclassmethod
from typing import Callable, Optional, Tuple
import numpy as np
import time
import scipy.sparse

from FPE import config


class BaseIntegrator(metaclass=ABCMeta):
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
        self, D: float, dt: float, diffScheme: str, adScheme: str,
        boundaryCond: str, splitMethod: str, output: Optional[bool] = False,
        constDiff: Optional[bool] = True
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
        self._validateInput(diffScheme, adScheme, boundaryCond, splitMethod)

        # Initialize instance variables. For now, this routine supports
        # constant diffusion coefficient integrations (i.e. D is not a tensor)
        self.D = D
        self.dt = dt

        # Condition for satisfying CFL criterion
        self.CFL = None

        # Instantiate the instance directives
        self.diffScheme = diffScheme    # Integration scheme for diffusion term
        self.adScheme = adScheme        # Integration scheme for advection term
        self.BC = boundaryCond          # Boundary conditions
        self.splitMethod = splitMethod  # Integator splitting method
        self.constDiff = constDiff      # Flag for if diffusion matrix is const

        # Flag for whether or not output will be printed
        self.output = output

        # Default initial argument for sparse method utilization in matrix
        # operations
        self.sparseCalc = False

        # Physics trackers
        self.workAccumulator = 0
        self.workTracker = []
        self.powerTracker = []

    def _validateInput(
        self, diffScheme: str, adScheme: str, boundaryCond: str,
        splitMethod: str
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
        if diffScheme not in config.diffSchemes:
            raise ValueError("diffScheme not recognized, see config file")

        if adScheme != "lax-wendroff":
            raise NotImplementedError(
                "Alternate advection methods currently not implemented, "
                "must use `lax-wendroff`"
            )

        if boundaryCond not in config.boundaryConditions:
            raise ValueError(
                "boundary condition not recognized, see config file"
            )

        if splitMethod not in config.splittingMethods:
            raise ValueError(
                "Splitting method not recognized, see config file"
            )

    def _setDiffusionScheme(self):
        """Method to set the diffusion scheme in the integrator instance. In
        all cases there are three options: implicit, explicit, and
        crank-nicolson. The latter is the default choice.
        """
        if(self.diffScheme.lower() == "implicit"):
            if(self.output):
                print("\t\tUsing fully implicit integration scheme...")
            self.expImp = 1.0

        elif(self.diffScheme.lower() == "explicit"):
            if(self.output):
                print("\t\tUsing fully explicit integration scheme...")
            self.expImp = 0.0

        elif(self.diffScheme.lower() == "crank-nicolson"):
            if(self.output):
                print("\t\tUsing Crank-Nicolson integration scheme...")
            self.expImp = 0.5

        else:
            if(self.output):
                print("\t\tInt scheme unknown, using default setting (CN)...")
            self.expImp = 0.5

    def testSparse(self):
        """Routine to check whether sparse matrix calculations are more rapid
        than standard matrix multiplication techniques. This is expected to be
        the case for large matrices, and more impactful when diffisuon is not
        constant. For constant diffusion term, the update due to diffusion is
        only a matrix multiplication (and not a matrix inversion) so the
        scaling is better with N.
        """
        if(self.constDiff):
            # NOTE Implement checks here
            self.spaseCalc = False

        else:
            sBMat = scipy.sparse.csr_matrix(self.BMat)
            sAMat = scipy.sparse.csr_matrix(self.AMat)
            self._testSparse(sAMat, sBMat)

    def _testSparse(
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
        startSparse_full = time.time()
        sbVec = sBMat.dot(self.prob)
        _ = scipy.sparse.linalg.spsolve(sAMat, sbVec)
        # Stop timer
        endSparse_full = time.time()
        timeSparse = endSparse_full - startSparse_full

        # Start timer (non-sparse calculations)
        startReg_full = time.time()
        bVec = np.matmul(self.BMat, self.prob)
        _ = np.linalg.solve(self.AMat, bVec)
        # stop timer
        endReg_full = time.time()
        timeReg = endReg_full - startReg_full

        # If sparse calculation is faster, use sparse matrix methods
        if(timeSparse < timeReg):
            if(self.output):
                print("\t\tSparse matrix methods preferred...")
            self.sparseCalc = True
            self.BMat = sBMat
            self.AMat = sAMat

        # Otherwise, use standard dense matrix calculation techniques
        else:
            if(self.output):
                print("\t\tDense matrix methods preferred...")
            self.sparseCalc = False

    def integrateStepAdvection(
        self, forceParams: Tuple, forceFunction: Callable
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
        self.advectionUpdate(forceParams, forceFunction, self.dt)

    def diffusionUpdate(self):
        """Update to the system probability based on the diffusion dynamics
        """
        if(self.sparseCalc):
            bVec = self.BMat.dot(self.prob)
            self.prob = scipy.sparse.linalg.spsolve(self.AMat, bVec)
        else:
            if(self.constDiff):
                self.prob = np.matmul(self.CMat, self.prob)
            else:
                bVec = np.matmul(self.BMat, self.prob)
                self.prob = np.linalg.solve(self.AMat, bVec)

    def advectionUpdate(
        self, forceParams: Tuple, forceFunction: Callable, deltaT: float
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
        self.laxWendroff(forceParams, forceFunction, deltaT)

    def integrate_step(self, forceParams: Tuple, forceFunction: Callable):
        """Integrates the equations by one time step. In this package, operator
        splitting methods are used, and this interface routine directs the update
        step towards the correct endpoint: Strang, Lie, ot SWSS

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        if(self.splitMethod == 'lie'):
            self._integrateStepLie(forceParams, forceFunction)

        elif(self.splitMethod == 'strang'):
            self._integrateStepStrang(forceParams, forceFunction)

        elif(self.splitMethod == 'swss'):
            self._integrateStepSWSS(forceParams, forceFunction)
        else:
            self._integrateStepStrang(forceParams, forceFunction)

    def _integrateStepLie(self, forceParams: Tuple, forceFunction: Callable):
        """Implementation of Lie splitting update of FPE

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        self.diffusionUpdate()

    def _integrateStepStrang(self, forceParams: Tuple, forceFunction: Callable):
        """Implementation of (symmetric) Strang Splitting update of FPE

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)
        self.diffusionUpdate()
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)

    def _integrateStepSWSS(self, forceParams: Tuple, forceFunction: Callable):
        """Implementation of the Symmetrically weighted Strang splitting update
        of the FRE.

        Args:
            forceParams (Tuple): Tuple representing the current force parameters
            forceFunction (Callable): The function taking arguments of the form
                (x, *args) that gives the force on the system as function of
                its position
        """
        initProb = self.prob
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        self.diffusionUpdate()
        prob_1 = self.prob
        self.prob = initProb
        self.diffusionUpdate()
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        prob_2 = self.prob
        self.prob = 0.5 * (prob_1 + prob_2)

    def check_CFL(self, forceParams: Tuple, forceFunction: Callable) -> bool:
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

        # Find maximum force over the entire domain
        maxForce = np.max(np.abs(forceFunction(self.xArray, forceParams)))

        # set CFL and check if it is less than or greater than Unity
        self.CFL = maxForce * self.dt / self.dx
        if(self.CFL > 1):
            if self.output:
                print("\t\tStability warning, invalid CFL --> " + str(self.CFL) + "\n\n")
            return False

        if self.output:
            print("\t\tCFL criterion satisfied, CFL --> " + str(self.CFL) + "\n\n")
        return True

    @property
    def get_prob(self) -> np.ndarray:
        return self.prob

    @abstractclassmethod
    def initializePhysicalTrackers(self):
        pass

    @abstractclassmethod
    def initDiffusionMatrix(self):
        pass

    # @abstractclassmethod
    # def check_CFL(self):
    #     pass

    @abstractclassmethod
    def laxWendroff(self, *args, **kwargs):
        pass
