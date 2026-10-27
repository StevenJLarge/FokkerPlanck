from typing import Callable, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from fokker_planck.base import Integrator
from fokker_planck.types.basetypes import (
    DiffScheme, BoundaryCondition, SplitMethod
)

ForcePair = Tuple[Callable, Callable]


class FokkerPlanck2D(Integrator):
    """2-D Fokker-Planck integrator.

    Probability is stored flattened with x the fastest index (Fortran order of
    an ``(Nx, Ny)`` grid). Diffusion uses sparse Crank-Nicolson on the 5-point
    Laplacian. Advection uses dimensionally split Lax-Wendroff
    (Ax(δt/2) Ay(δt) Ax(δt/2)), with the same wall / periodic / open flux
    rules as the 1-D integrator. Advection and diffusion are then combined
    with the parent class operator splitting.
    """

    def __init__(
        self, D: float, dt: float, dx: float, dy: float,
        x_array: np.ndarray, y_array: np.ndarray,
        diff_scheme: DiffScheme = DiffScheme.CrankNicolson,
        boundary_cond: BoundaryCondition = BoundaryCondition.HardWall,
        split_method: SplitMethod = SplitMethod.Strang,
        output: bool = False,
        const_diffusion: bool = True
    ):
        super().__init__(
            D, dt, diff_scheme, boundary_cond, split_method, output,
            const_diffusion
        )
        self.dx = dx
        self.dy = dy
        self.x_array = np.asarray(x_array, dtype=float)
        self.y_array = np.asarray(y_array, dtype=float)
        self.Nx = len(self.x_array)
        self.Ny = len(self.y_array)
        self.N = self.Nx * self.Ny

        self.prob = np.ones(self.N) / (self.N * self.dx * self.dy)
        self.init_physical_trackers()
        self.init_diffusion_matrix()

    @property
    def dimension(self) -> int:
        return 2

    @property
    def prob_grid(self) -> np.ndarray:
        return self.prob.reshape((self.Nx, self.Ny), order="F")

    @prob_grid.setter
    def prob_grid(self, grid: np.ndarray):
        self.prob = np.asarray(grid, dtype=float).reshape(-1, order="F")

    @property
    def mean(self) -> np.ndarray:
        grid = self.prob_grid
        mass_elem = grid * self.dx * self.dy
        mean_x = np.sum(mass_elem * self.x_array[:, None])
        mean_y = np.sum(mass_elem * self.y_array[None, :])
        return np.array([mean_x, mean_y])

    @property
    def covariance(self) -> np.ndarray:
        grid = self.prob_grid
        mx, my = self.mean
        dx, dy = self.dx, self.dy
        xx = np.sum(grid * ((self.x_array[:, None] - mx) ** 2) * dx * dy)
        yy = np.sum(grid * ((self.y_array[None, :] - my) ** 2) * dx * dy)
        xy = np.sum(
            grid
            * (self.x_array[:, None] - mx)
            * (self.y_array[None, :] - my)
            * dx * dy
        )
        return np.array([[xx, xy], [xy, yy]])

    def _cell_volume(self) -> float:
        return self.dx * self.dy

    def normalize_prob(self):
        self.prob = self.prob / (np.sum(self.prob) * self._cell_volume())

    def reset(
        self, covariance: Optional[np.ndarray] = None,
        mean: Optional[np.ndarray] = None
    ):
        if covariance is not None and mean is not None:
            self.initialize_probability(mean, covariance)
        else:
            self.prob = np.ones(self.N) / (self.N * self._cell_volume())
        self.init_physical_trackers()

    def init_physical_trackers(self):
        self.work_accumulator = 0
        self.work_tracker = []
        self.power_tracker = []
        self.time_tracker = []
        self.flux_x = np.zeros((self.Nx, self.Ny))
        self.flux_y = np.zeros((self.Nx, self.Ny))
        self.flux_tracker = np.zeros(2)

    def initialize_probability(
        self, mean: Union[np.ndarray, Tuple[float, float]],
        covariance: Union[np.ndarray, Tuple[float, float]]
    ):
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        if cov.shape == (2,):
            cov = np.diag(cov)
        X, Y = np.meshgrid(self.x_array, self.y_array, indexing="ij")
        delta = np.stack([X - mean[0], Y - mean[1]], axis=-1)
        inv_cov = np.linalg.inv(cov)
        quad = np.einsum("...i,ij,...j", delta, inv_cov, delta)
        self.prob_grid = np.exp(-0.5 * quad)
        self.normalize_prob()

    def initialize_user_probability(
        self, func: Callable, params: Optional[Tuple] = None
    ):
        X, Y = np.meshgrid(self.x_array, self.y_array, indexing="ij")
        grid = func(X, Y, *(params or ()))
        self.prob_grid = np.asarray(grid, dtype=float)
        self.normalize_prob()

    def _unpack_forces(
        self, force_function: Union[Callable, ForcePair],
        force_y: Optional[Callable] = None
    ) -> ForcePair:
        if force_y is not None:
            return force_function, force_y
        if isinstance(force_function, (tuple, list)) and len(force_function) == 2:
            return force_function[0], force_function[1]
        raise TypeError(
            "2D integrator requires force_x and force_y, either as two "
            "arguments or as a length-2 tuple"
        )

    def _eval_force(
        self, force: Callable, x, y, params
    ) -> np.ndarray:
        return np.asarray(force(x, y, params), dtype=float)

    def _force_fields(
        self, force_x: Callable, force_y: Callable, params
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = np.meshgrid(self.x_array, self.y_array, indexing="ij")
        shape = X.shape
        fx = np.broadcast_to(
            np.asarray(self._eval_force(force_x, X, Y, params), dtype=float),
            shape
        ).copy()
        fy = np.broadcast_to(
            np.asarray(self._eval_force(force_y, X, Y, params), dtype=float),
            shape
        ).copy()
        return fx, fy

    def check_CFL(
        self, force_params: Tuple, force_function: Union[Callable, ForcePair],
        force_y: Optional[Callable] = None
    ) -> bool:
        force_x, force_y = self._unpack_forces(force_function, force_y)
        fx, fy = self._force_fields(force_x, force_y, force_params)
        cfl_x = np.max(np.abs(fx)) * self.D * self.dt / self.dx
        cfl_y = np.max(np.abs(fy)) * self.D * self.dt / self.dy
        self.CFL = float(max(cfl_x, cfl_y))
        if self.CFL > 1.0:
            if self.output:
                print(f"\t\tStability warning, invalid CFL --> {self.CFL}\n")
            return False
        return True

    def integrate_step(
        self, force_params: Tuple, force_function: Union[Callable, ForcePair],
        force_y: Optional[Callable] = None
    ):
        forces = self._unpack_forces(force_function, force_y)
        super().integrate_step(force_params, forces)

    def advection_update(
        self, force_params: Tuple, force_function: Union[Callable, ForcePair],
        delta_t: float
    ):
        self.lax_wendroff(force_params, force_function, delta_t)

    def diffusion_update(self):
        rhs = self.BMat.dot(self.prob)
        self.prob = self._A_factor.solve(rhs)

    def init_diffusion_matrix(self):
        self._setDiffusionScheme()
        theta = self.exp_imp
        alpha_x = self.D * self.dt / (self.dx * self.dx)
        alpha_y = self.D * self.dt / (self.dy * self.dy)

        A = sp.lil_matrix((self.N, self.N))
        B = sp.lil_matrix((self.N, self.N))

        for j in range(self.Ny):
            for i in range(self.Nx):
                row = i + j * self.Nx
                A[row, row] += 1.0
                B[row, row] += 1.0
                self._add_axis_terms(
                    A, B, row, i, j, axis="x",
                    alpha=alpha_x, theta=theta
                )
                self._add_axis_terms(
                    A, B, row, i, j, axis="y",
                    alpha=alpha_y, theta=theta
                )

        self.AMat = A.tocsc()
        self.BMat = B.tocsc()
        self._A_factor = splu(self.AMat)
        self.sparse_calc = True

    def _add_axis_terms(
        self, A, B, row: int, i: int, j: int, axis: str,
        alpha: float, theta: float
    ):
        if axis == "x":
            idx, n = i, self.Nx
            minus_col = (i - 1) + j * self.Nx
            plus_col = (i + 1) + j * self.Nx
            wrap_minus = (n - 1) + j * self.Nx
            wrap_plus = 0 + j * self.Nx
        else:
            idx, n = j, self.Ny
            minus_col = i + (j - 1) * self.Nx
            plus_col = i + (j + 1) * self.Nx
            wrap_minus = i + (n - 1) * self.Nx
            wrap_plus = i + 0 * self.Nx

        at_lo = idx == 0
        at_hi = idx == n - 1
        one_m = 1.0 - theta

        if self.BC is BoundaryCondition.Periodic:
            A[row, row] += 2.0 * theta * alpha
            A[row, wrap_minus if at_lo else minus_col] += -theta * alpha
            A[row, wrap_plus if at_hi else plus_col] += -theta * alpha
            B[row, row] += -2.0 * one_m * alpha
            B[row, wrap_minus if at_lo else minus_col] += one_m * alpha
            B[row, wrap_plus if at_hi else plus_col] += one_m * alpha
            return

        if self.BC is BoundaryCondition.HardWall and (at_lo or at_hi):
            # Finite-volume no-flux: only the interior face contributes,
            # L_wall = (p_interior - p_wall) / dz^2
            A[row, row] += theta * alpha
            B[row, row] += -one_m * alpha
            neigh = plus_col if at_lo else minus_col
            A[row, neigh] += -theta * alpha
            B[row, neigh] += one_m * alpha
            return

        # Interior, or Open (missing neighbour treated as p=0)
        A[row, row] += 2.0 * theta * alpha
        B[row, row] += -2.0 * one_m * alpha
        if not at_lo:
            A[row, minus_col] += -theta * alpha
            B[row, minus_col] += one_m * alpha
        if not at_hi:
            A[row, plus_col] += -theta * alpha
            B[row, plus_col] += one_m * alpha

    def lax_wendroff(
        self, force_params: Tuple,
        force_function: Union[Callable, ForcePair],
        delta_t: float
    ):
        force_x, force_y = self._unpack_forces(force_function)
        grid = self.prob_grid
        # Dimensional Strang split so the advection substep stays second order
        grid = self._lax_wendroff_axis(grid, force_x, force_params, 0.5 * delta_t, axis=0)
        grid = self._lax_wendroff_axis(grid, force_y, force_params, delta_t, axis=1)
        grid = self._lax_wendroff_axis(grid, force_x, force_params, 0.5 * delta_t, axis=0)
        self.prob_grid = grid

    def _lax_wendroff_axis(
        self, grid: np.ndarray, force: Callable, params, delta_t: float,
        axis: int
    ) -> np.ndarray:
        if axis == 0:
            dz = self.dx
            Xc, Yc = np.meshgrid(self.x_array, self.y_array, indexing="ij")
            faces = np.empty(self.Nx + 1)
            faces[0] = self.x_array[0] - 0.5 * self.dx
            faces[1:] = self.x_array + 0.5 * self.dx
            Xf, Yf = np.meshgrid(faces, self.y_array, indexing="ij")
        else:
            dz = self.dy
            Xc, Yc = np.meshgrid(self.x_array, self.y_array, indexing="ij")
            faces = np.empty(self.Ny + 1)
            faces[0] = self.y_array[0] - 0.5 * self.dy
            faces[1:] = self.y_array + 0.5 * self.dy
            Xf, Yf = np.meshgrid(self.x_array, faces, indexing="ij")

        F = np.broadcast_to(
            np.asarray(self._eval_force(force, Xc, Yc, params), dtype=float),
            grid.shape
        )
        Fh_shape = (self.Nx + 1, self.Ny) if axis == 0 else (self.Nx, self.Ny + 1)
        Fh = np.broadcast_to(
            np.asarray(self._eval_force(force, Xf, Yf, params), dtype=float),
            Fh_shape
        )

        p = np.moveaxis(grid, axis, 0)
        F = np.moveaxis(np.array(F, dtype=float, copy=True), axis, 0)
        Fh = np.moveaxis(np.array(Fh, dtype=float, copy=True), axis, 0)
        n = p.shape[0]
        coeff = self.D * delta_t / dz

        half_p = np.zeros((n + 1,) + p.shape[1:])
        half_flux = np.zeros_like(half_p)

        half_p[1:-1] = (
            0.5 * (p[1:] + p[:-1])
            - 0.5 * coeff * (F[1:] * p[1:] - F[:-1] * p[:-1])
        )
        half_flux[1:-1] = Fh[1:-1] * half_p[1:-1]

        if self.BC is BoundaryCondition.Periodic:
            wrap = 0.5 * coeff * (F[0] * p[0] - F[-1] * p[-1])
            half_p[0] = 0.5 * (p[0] + p[-1]) - wrap
            half_flux[0] = Fh[0] * half_p[0]
            half_flux[-1] = half_flux[0]
        elif self.BC is BoundaryCondition.Open:
            half_p[0] = 0.5 * p[0] - 0.5 * coeff * F[0] * p[0]
            half_p[-1] = 0.5 * p[-1] + 0.5 * coeff * F[-1] * p[-1]
            half_flux[0] = Fh[0] * half_p[0]
            half_flux[-1] = Fh[-1] * half_p[-1]
        else:
            half_flux[0] = 0.0
            half_flux[-1] = 0.0

        new_p = p - coeff * (half_flux[1:] - half_flux[:-1])
        return np.moveaxis(new_p, 0, axis)

    def work_step(
        self, force_params_pre: Tuple, force_params_post: Tuple,
        force_x: Callable, force_y: Callable, energy_function: Callable
    ):
        X, Y = np.meshgrid(self.x_array, self.y_array, indexing="ij")
        grid = self.prob_grid
        vol = self._cell_volume()
        curr_energy = np.sum(energy_function(X, Y, force_params_pre) * grid) * vol
        new_energy = np.sum(energy_function(X, Y, force_params_post) * grid) * vol

        self.integrate_step(force_params_post, force_x, force_y)

        self.work_accumulator += new_energy - curr_energy
        self.work_tracker.append(self.work_accumulator)
        self.power_tracker.append((new_energy - curr_energy) / self.dt)

    def flux_step(
        self, force_params: Tuple, force_x: Callable, force_y: Callable
    ):
        fx, fy = self._force_fields(force_x, force_y, force_params)
        grid = self.prob_grid
        dp_dx, dp_dy = np.gradient(grid, self.dx, self.dy)
        self.flux_x = self.D * fx * grid - self.D * dp_dx
        self.flux_y = self.D * fy * grid - self.D * dp_dy
        vol = self._cell_volume()
        self.flux_tracker = np.array([
            np.sum(self.flux_x) * vol,
            np.sum(self.flux_y) * vol,
        ])
        self.integrate_step(force_params, force_x, force_y)


FPE_integrator_2D = FokkerPlanck2D
