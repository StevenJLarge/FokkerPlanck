# Base class for FPE integrator

from abc import ABC, abstractclassmethod
from typing import Callable, Optional, Tuple, Union
import numpy as np
import time
import scipy.sparse


class BaseIntegratorResult(ABC):
    # Base class for integrator result containers
    def __init__(self):
        pass


class BaseIntegrator(ABC):
    # Base class for integrator
    def __init__(
        self, D: float, diffScheme: str, adScheme: str, boundaryCond: str,
        splitMethod: str, output: Optional[bool] = False,
        constDiff: Optional[bool] = True
    ):
        # Validate input parameters
        self._validateInput(diffScheme, adScheme, boundaryCond, splitMethod)

        # Initialize instance variables. For now, this routine supports
        # constant diffusion coefficient integrations (i.e. D is not a tensor)
        self.D = D

        # Instantiate the instance directives
        self.diffScheme = diffScheme    # Integration scheme for diffusion term
        self.adScheme = adScheme        # Integration scheme for advection term
        self.BC = boundaryCond          # Boundary conditions
        self.splitMethod = splitMethod  # Integator splitting method
        self.constDiff = constDiff      # Flag for if diffusion matrix is cost

        # Flag for whether or not output will be printed
        self.output = output

        # Default initial argument for sparse method utilization in matrix
        # operations
        self.sparTest = False

    def _validateInput(
        self, diffScheme: str, adScheme: str, coundaryCond: str,
        splitMethod: str
    ):
        pass

    def initializeGaussianProbability(
        self, mean: Union[float, np.ndarray],
        variance: Union[float, np.ndarray],
        corrMat: Optional[np.ndarray] = None
    ):
        pass

    def _setDiffusionScheme(self):
        if(self.diffScheme.lower() == "implicit"):
            if(self.output is True):
                print("\t\tUsing fully implicit integration scheme...")
            self.expImp = 1.0

        elif(self.diffScheme.lower() == "explicit"):
            if(self.output is True):
                print("\t\tUsing fully explicit integration scheme...")
            self.expImp = 0.0

        elif(self.diffScheme.lower() == "crank-nicolson"):
            if(self.output is True):
                print("\t\tUsing Crank-Nicolson integration scheme...")
            self.expImp = 0.5

        else:
            if(self.output is True):
                print("\t\tInt scheme unknown, using default setting (CN)...")
            self.expImp = 0.5

    def testSparse(self):
        # If constant diffusion matrix, then run reduced calculation (we dont
        # need to invert every step in this case)
        if(self.constDiff is True):
            # Initialize sparse-matrix representations of A, B, C
            sBMat = scipy.sparse.csr_matrix(self.BMat)
            sAMat = scipy.sparse.csr_matrix(self.AMat)
            sCMat = scipy.sparse.csr_matrix(self.CMat)
            self._testSparse_CD(sAMat, sBMat, sCMat)

        else:
            sBMat = scipy.sparse.csr_matrix(self.BMat)
            sAMat = scipy.sparse.csr_matrix(self.AMat)
            self._testSparse_gen(sAMat, sBMat)

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
            if(self.output is True):
                print("\t\tSparse matrix methods preferred...")
            self.sparTest = True
            self.BMat = sBMat
            self.AMat = sAMat

        else:
            if(self.output is True):
                print("\t\tDense matrix methods preferred...")
            self.sparTest = False

    # ANCHOR Think of a better name for this...
    def integrateStepAdvection(
        self, forceParams: Tuple, forceFunction: Callable
    ):
        self.advectionUpdate(forceParams, forceFunction, self.dt)

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
        self, forceParams: Tuple, forceFunction: Callable, deltaT: float
    ):
        if(self.adScheme == 'lax-wendroff' or self.adScheme == 'lw'):
            self.laxWendroff(forceParams, forceFunction, deltaT)
        if(self.adScheme == 'lax' or self.adScheme == 'l'):
            self.lax(forceParams, forceFunction, deltaT)
        else:
            self.laxWendroff(forceParams, forceFunction, deltaT)

    def integrate_step(self, forceParams: Tuple, forceFunction: Callable):
        if(self.splitMethod == 'lie'):
            self._integrateStepLie(forceParams, forceFunction)

        elif(self.splitMethod == 'strang'):
            self._integrateStepStrang(forceParams, forceFunction)

        elif(self.splitMethod == 'swss'):
            self._integrateStepSWSS(forceParams, forceFunction)
        else:
            self._integrateStepStrang(forceParams, forceFunction)

    def _integrateStepLie(self, forceParams: Tuple, forceFunction: Callable):
        self.advectionUpdate(forceParams, forceFunction, self.dt)
        self.diffusionUpdate()

    def _integrateStepStrang(self, forceParams, forceFunction):
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)
        self.diffusionUpdate()
        self.advectionUpdate(forceParams, forceFunction, 0.5 * self.dt)

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

    @abstractclassmethod
    def lax(self, *args, **kwargs):
        pass
