# Base class for FPE integrator

from abc import ABCMeta, abstractclassmethod
from typing import Callable, Optional, Tuple, Union
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
        if diffScheme not in config.diffSchemes:
            raise ValueError("diffScheme not recognized, see config file")

        if adScheme != "lax-wendroff":
            raise NotImplementedError(
                "Alternate advection methods currentyl unavailable, must use"
                "`lax-wendroff`"
            )

        if boundaryCond not in config.boundaryConditions:
            raise ValueError(
                "boundary condition not recognized, see config file"
            )

        if splitMethod not in config.splittingMethods:
            raise ValueError(
                "Splitting method not recognized, see config file"
            )

    def initializeGaussianProbability(
        self, mean: Union[float, np.ndarray],
        variance: Union[float, np.ndarray],
        corrMat: Optional[np.ndarray] = None
    ):
        pass

    def _setDiffusionScheme(self):
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
        # If constant diffusion matrix, then run reduced calculation (we dont
        # need to invert every step in this case)
        if(self.constDiff):
            self.spaseCalc = False

        else:
            sBMat = scipy.sparse.csr_matrix(self.BMat)
            sAMat = scipy.sparse.csr_matrix(self.AMat)
            self._testSparse_gen(sAMat, sBMat)

    def _testSparse_gen(
        self, sAMat: scipy.sparse.csr_matrix,
        sBMat: scipy.sparse.csr_matrix
    ):
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
            if(self.output):
                print("\t\tSparse matrix methods preferred...")
            self.sparseCalc = True
            self.BMat = sBMat
            self.AMat = sAMat

        else:
            if(self.output):
                print("\t\tDense matrix methods preferred...")
            self.sparseCalc = False

    def integrateStepAdvection(
        self, forceParams: Tuple, forceFunction: Callable
    ):
        self.advectionUpdate(forceParams, forceFunction, self.dt)

    def diffusionUpdate(self):
        if(self.sparseCalc):
            if(self.constDiff):
                self.prob = self.CMat.dot(self.prob)
            else:
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

    # @classmethod
    def _integrateStepLie(self, forceParams: Tuple, forceFunction: Callable):
        """Implementation of Lie splitting update of FPE

        Args:
            forceParams (Tuple): _description_
            forceFunction (Callable): _description_
        """
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        self.diffusionUpdate()

    # @classmethod
    def _integrateStepStrang(self, forceParams: Tuple, forceFunction: Callable):
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)
        self.diffusionUpdate()
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)

    # @classmethod
    def _integrateStepSWSS(self, forceParams: Tuple, forceFunction: Callable):
        initProb = self.prob
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        self.diffusionUpdate()
        prob_1 = self.prob
        self.prob = initProb
        self.diffusionUpdate()
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        prob_2 = self.prob
        self.prob = 0.5 * (prob_1 + prob_2)

    @property
    def prob(self) -> np.ndarray:
        return self.prob

    @abstractclassmethod
    def initializePhysicalTrackers(self):
        pass

    @abstractclassmethod
    def initDiffusionMatrix(self):
        pass

    @abstractclassmethod
    def check_CFL(self):
        pass

    @abstractclassmethod
    def laxWendroff(self, *args, **kwargs):
        pass

