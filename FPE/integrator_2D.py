from typing import Callable, Optional, Tuple, Dict
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import inv
from FPE.base import BaseIntegrator


# ANCHOR 2D Integrator
class FPE_integrator_2D(BaseIntegrator):

    def __init__(
        self, D: float, dt: float, dx: float, dy: float, xArray: float,
        yArray: float, diffScheme: Optional[str] = 'crank-nicolson',
        adScheme: Optional[str] = 'lax-wendroff',
        boundaryCond: Optional[str] = 'hard-wall',
        splitMethod: Optional[str] = 'strang',
        output: Optional[bool] = False,
        constDiff: Optional[bool] = True
    ):

        super().__init__(
            D, dt, diffScheme, adScheme, boundaryCond, splitMethod, output,
            constDiff
        )
        self.dx = dx
        self.dy = dy
 
        self.Nx = len(xArray)
        self.Ny = len(yArray)
        self.N = self.Nx * self.Ny

        # self.prob = np.ones((self.N,self.N))/(sum(self.Nx*self.dx)*sum(self.Ny*self.dy))
        self.prob = np.ones(self.Nx * self.Ny) / (self.Nx * self.Ny * self.dx * self.dy)
        self.xArray = xArray
        self.yArray = yArray

        self.constDiff = constDiff
        self.sparTest = False

        self.initDiffusionMatrix()

    @property
    def dimension(self) -> int:
        return 2

    @property
    def mean(self) -> Dict:
        prob_grid = self.prob.reshape(self.Nx, self.Ny)
        marginal_x = np.sum(prob_grid, axis=1) * self.dy
        marginal_y = np.sum(prob_grid, axis=0) * self.dx

        mean_x = np.sum(marginal_x * self.dx * self.xArray)
        mean_y = np.sum(marginal_y * self.dy * self.yArray)

        return {'x_mean': mean_x, 'y_mean': mean_y}

    @property
    def covariance(self) -> np.ndarray:
        pass

    def _flatten_probability(self, prob: np.ndarray):
        pass

    def reset(
        self, covariance: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None
    ):
        if covariance is not None and mean is not None:
            self.initializeProbability(mean, covariance)
        else:
            self.prob = np.ones((self.Nx, self.Ny)) / (self.Nx * self.Ny * self.dx * self.dy)
        self.initializePhysicalTrackers()

    def initializePhysicalTrackers(self):
        self.workAccumulator = 0
        self.workTracker = []
        self.powerTracker = []
        self.timeTracker = []

        self.xFlux = np.zeros(self.Nx)
        self.yFlux = np.zeros(self.Ny)
        self.xFluxTracker = 0
        self.yFluxTracker = 0

    def initializeProbability(self, mean: np.ndarray, covariance: np.ndarray):
        pass

    def initializeUserProbability(
        self, func: Callable, params: Optional[Tuple] = None
    ):
        pass

    def initDiffusionMatrix(self):
        if(self.output is True):
            print("\n\nInitializing diffusion term integration matrix...\n")

        self._setDiffusionScheme()

        if(self.output is True):
            print("\t\tInitializing integration matrices for diffusion\n")

        alpha = self.D * self.dt / (self.dx * self.dx)

        self.AMat = sp.lil_matrix((self.N, self.N))
        self.BMat = sp.lil_matrix((self.N, self.N))

        # Bulk term initializations
        self.AMat.setdiag(1 + 4 * alpha * self.expImp)
        self.AMat.setdiag(-1 * self.expImp * alpha, k=1)
        self.AMat.setdiag(-1 * self.expImp * alpha, k=-1)
        self.AMat.setdiag(-1 * self.expImp * alpha, k=self.Nx)
        self.AMat.setdiag(-1 * self.expImp * alpha, k=-self.Nx)

        self.BMat.setdiag(1 - 4 * alpha * (1 - self.expImp))
        self.BMat.setdiag(alpha * (1 - self.expImp), k=1)
        self.BMat.setdiag(alpha * (1 - self.expImp), k=-1)
        self.BMat.setdiag(alpha * (1 - self.expImp), k=self.Nx)
        self.BMat.setdiag(alpha * (1 - self.expImp), k=-self.Nx)

        self._initializeBoundaryTerms(alpha)

        # convert to csc format
        self.AMat = self.AMat.tocsc()
        self.BMat = self.BMat.tocsc()

        self.CMat = inv(self.AMat).dot(self.BMat)

    def _initializeBoundaryTerms(self, alpha: float):
        self._matrixBoundary_A(alpha, 0)
        self._matrixBoundary_B(alpha, 0)

        self._matrixBoundary_A(alpha, self.Nx - 1)
        self._matrixBoundary_B(alpha, self.Nx - 1)

    def _matrixBoundary_A(self, alpha: float, idx: int):
        if self.BC == 'periodic':
            # x-dimension BCs

            # Need to place top right and bottom left elements in each block
            # matrix for X and Y directions
            for image in range(0, self.Ny):
                x_idx = idx + self.Nx * image
                y_idx_fw = (idx + 1) % self.Nx + self.Nx * image
                y_idx_rv = (idx - 1) % self.Nx + self.Nx * image

                self.AMat[x_idx, y_idx_fw] = -self.expImp * alpha
                self.AMat[x_idx, y_idx_rv] = -self.expImp * alpha

            for image in range(1, self.Ny):
                self.AMat[image * self.Nx - 1, image * self.Nx] = 0
                self.AMat[image * self.Nx, image * self.Nx - 1] = 0

            # y-dimension BCs
            self.AMat.setdiag(-self.expImp * alpha, k=self.N - self.Nx)
            self.AMat.setdiag(-self.expImp * alpha, k=-(self.N - self.Nx))

        elif self.BC == "hard-wall":
            # This resolves the HW conditions in the X-dimension, still need
            # to resolve in y
            # NOTE Need to update this, I aded the 8, but I need to figure out
            # where the 4 goes and make sure the 2s are right, as well as
            # resolve the multi-coordinate boundaries,..

            # NOTE THIS IS INCORRECT....
            # X-boundaries only
            for image in range(1, self.Ny - 1):
                _idx = idx + self.Nx * image
                _idx_fwd = idx + 1 + (self.Nx * image)
                _idx_rev = idx - 1 + (self.Nx * image)
                self.AMat[_idx, _idx] = 1 + 4 * alpha
                # self.AMat[_idx, abs(idx - 1) + self.Nx * image] = -1 * alpha
                self.AMat[_idx, _idx_fwd] = -1 * alpha
                self.AMat[_idx, _idx_rev] = -1 * alpha

            # Y-boundaries only
            for image in range(1, self.Nx - 1):
                # We want this to change all x-index values WHEN y = 0, Ny-1.
                # So this is all of the x-values within the top left and bottom
                # right BLOCKS aide from the internal boundary points.
                _idx_fwd = idx - image
                _idx_rev = idx + image

                self.AMat[_idx_fwd, _idx] = -1 * alpha
                self.AMat[_idx_rev, _idx] = -1 * alpha

            # Need to test this / verify that this is the correct way of doing this...
            for diag_idx in range(self.Nx):
                self.AMat[diag_idx, self.Nx + diag_idx] = -2 * alpha
                self.AMat[self.N - 1 - diag_idx, self.N - self.Nx - 1 - diag_idx] = -2 * alpha

            # These are the conditions where X and Y boudnaries are involved
            # Check this AND look into the other terms in these rows that need to be updated
            self.AMat[0, 0] = 1.0
            self.AMat[-1, -1] = 1.0
            self.AMat[self.Nx - 1, self.Ny - 1] = 1.0
            self.AMat[self.N - self.Nx, self.N - self.Ny] = 1.0

        elif self.BC == "open":
            for image in range(1, self.Ny):
                self.AMat[image * self.Nx - 1, image * self.Nx] = 0
                self.AMat[image * self.Nx, image * self.Nx - 1] = 0

        else:
            raise ValueError(
                f"Invalid boundary condition `{self.BC}`, cannot resolve "
                "diffusion matrix A"
            )

    def _matrixBoundary_B(self, alpha: float, idx: int):

        if self.BC == "periodic":
            # x-dimension BCs
            for image in range(0, self.Ny):
                x_idx = idx + self.Nx * image
                y_idx_fw = (idx + 1) % self.Nx + self.Nx * image
                y_idx_rv = (idx - 1) % self.Nx + self.Nx * image

                self.BMat[x_idx, y_idx_fw] = alpha * (1 - self.expImp)
                self.BMat[x_idx, y_idx_rv] = alpha * (1 - self.expImp)

            for image in range(1, self.Ny):
                self.BMat[image * self.Nx - 1, image * self.Nx] = 0
                self.BMat[image * self.Nx, image * self.Nx - 1] = 0

            # y-dimension BCs
            self.BMat.setdiag(alpha * (1 - self.expImp), k=self.N - self.Nx)
            self.BMat.setdiag(alpha * (1 - self.expImp), k=-(self.N - self.Nx))

        elif self.BC == "hard-wall":
            for image in range(0, self.Ny):
                _idx = idx + self.Nx * image
                self.BMat[_idx, _idx] = 1
                self.BMat[_idx, abs(idx - 1) + self.Nx * image] = 0

            for image in range(1, self.Ny):
                self.BMat[image * self.Nx - 1, image * self.Nx] = 0
                self.BMat[image * self.Nx, image * self.Nx - 1] = 0

            for diag_idx in range(self.Nx):
                self.BMat[diag_idx, self.Nx + diag_idx] = 0
                self.BMat[self.N - 1 - diag_idx, self.N - self.Nx - 1 - diag_idx] = 0

        elif self.BC == "open":
            for image in range(1, self.Ny):
                self.BMat[image * self.Nx - 1, image * self.Nx] = 0
                self.BMat[image * self.Nx, image * self.Nx - 1] = 0

        else:
            raise ValueError(
                f"Invalid boundary condition {self.BC}, cannot resolve "
                "diffusion matrix B"
            )

    # ANCHOR need to update function signature
    def work_step(self):
        pass

    def flux_step(self):
        pass

    # TODO Migrate this to the base class
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

    # TODO Keep this in the base class
    def diffusionUpdate(self):
        if (self.constDiff is True):
            self.prob = self.CMat.dot(self.prob)
        else:
            bVec = self.BMat.dot(self.prob)
            self.prob = scipy.sparse.linalg.spsolve(self.AMat, bVec)

    # TODO Also put this in the base class
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
        # self.laxWendroff_lieSplit_leg(forceFunc_x, forceFunc_y, forceParams, deltaT)
        pass

    # Probably just stick with LW for this? Just need to get it working...
    def lax_leg(self, forceParams: Tuple, forceFunction: Callable, deltaT: float):
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

    def lax_lieSplit_leg(
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

    def laxWendroff_lieSplit_leg(
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

    # TODO write out hese routines
    def laxWendroff(self):
        pass

    def _calcFlux_laxWendroff(self):
        pass

    def _getFluxDiff_laxWendroff(self):
        pass


if __name__ == "__main__":
    D = 1.0
    dt = 0.01
    dx = 0.1
    dy = 0.1
    xArray = np.arange(-1, 1, dx)
    yArray = np.arange(-1, 1, dy)

    fpe = FPE_integrator_2D(D, dt, dx, dy, xArray, yArray)


    fpe = FPE_integrator_2D(1.0, dt, dx, dy, xArray, yArray)

