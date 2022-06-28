# Base class for FPE integrator

from abc import ABC, abstractclassmethod
from typing import Callable, Optional, Tuple, Union
import numpy as np
import time
import scipy.sparse

from FPE import config


class BaseIntegratorResult(ABC):
    # Base class for integrator result containers
    def __init__(self):
        pass


class BaseIntegrator(ABC):
    # Base class for integrator
    def __init__(
        self, D: float, dt: float, diffScheme: str, adScheme: str,
        boundaryCond: str, splitMethod: str, output: Optional[bool] = False,
        constDiff: Optional[bool] = True
    ):
        # Validate input parameters
        self._validateInput(diffScheme, adScheme, boundaryCond, splitMethod)

        # Initialize instance variables. For now, this routine supports
        # constant diffusion coefficient integrations (i.e. D is not a tensor)
        self.D = D
        self.dt = dt

        # Instantiate the instance directives
        self.diffScheme = diffScheme    # Integration scheme for diffusion term
        # self.adScheme = adScheme        # Integration scheme for advection term
        if adScheme != "lax-wendroff":
            raise DeprecationWarning(
                "Alternate advection methods unavailable, defaulting"
                + "to LW, in a future version this will raise an error "
            )
        self.adScheme = "lax-wendroff"
        self.BC = boundaryCond          # Boundary conditions
        self.splitMethod = splitMethod  # Integator splitting method
        self.constDiff = constDiff      # Flag for if diffusion matrix is cost

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

        # if adScheme not in config.adSchemes:
        #     raise ValueError("adScheme not recognized, see config file")

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

    # NOTE this routine is now depriciated
    def _testSparse_CD(
        self, sAMat: scipy.sparse.csr.csr_matrix,
        sBMat: scipy.sparse.csr.csr_matrix, sCMat: scipy.sparse.csr.csr_matrix
    ):
        startSparse_inv = time.time()
        _ = sCMat.dot(self.prob)
        endSparse_inv = time.time()
        timeSparse = endSparse_inv - startSparse_inv

        startReg_inv = time.time()
        _ = np.linalg.solve(self.CMat, self.prob)
        endReg_inv = time.time()
        timeReg = endReg_inv - startReg_inv

        if(timeSparse < timeReg):
            if(self.output):
                print("\t\tSparse matrix methods preferred...")

            self.sparseCalc = True
            self.AMat = sAMat
            self.BMat = sBMat
            self.CMat = sCMat

        else:
            if(self.output):
                print("\t\tDense matrix methods preferred...")
            self.sparseCalc = False

    def _testSparse_gen(
        self, sAMat: scipy.sparse.csr.csr_matrix,
        sBMat: scipy.sparse.csr.csr_matrix
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

    # ANCHOR Think of a better name for this...
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
        # There is now only the one advection method that is stable, so this
        # will be used
        self.laxWendroff(forceParams, forceFunction, deltaT)

    def integrate_step(self, forceParams: Tuple, forceFunction: Callable):
        if(self.splitMethod == 'lie'):
            self._integrateStepLie(forceParams, forceFunction)

        elif(self.splitMethod == 'strang'):
            self._integrateStepStrang(forceParams, forceFunction)

        elif(self.splitMethod == 'swss'):
            self._integrateStepSWSS(forceParams, forceFunction)
        else:
            # NOTE: Potentially change this to raise an error?
            self._integrateStepStrang(forceParams, forceFunction)

    @classmethod
    def _integrateStepLie(self, forceParams: Tuple, forceFunction: Callable):
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        self.diffusionUpdate()

    # @classmethod
    def _integrateStepStrang(self, forceParams, forceFunction):
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)
        self.diffusionUpdate()
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)

    @classmethod
    def _integrateStepSWSS(self, forceParams, forceFunction):
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
    def get_prob(self) -> np.ndarray:
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

    # @abstractclassmethod
    # def lax(self, *args, **kwargs):
    #     pass
