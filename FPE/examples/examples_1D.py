# Samples of the FPE Integrator in action

import os
from typing import Tuple, Optional, List, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.integrate as si
from FPE import Integrator
import FPE.forceFunctions as ff

from pathlib import Path


class ProbabilityTracker():
    def __init__(self, xArray: Iterable, prob: np.ndarray, time: float):
        self.X = xArray
        self.normalization = [calc_normalization(prob, xArray)]
        self.density = [prob]
        self.time = [time]

    def update(self, prob: np.ndarray, time: float):
        self.normalization.append(calc_normalization(prob, self.X))
        self.density.append(prob)
        self.time.append(time)

    def report(self) -> Tuple[Iterable]:
        return self.density, self.time, self.normalization, self.X


class ProbabilityWorkTracker(ProbabilityTracker):
    def __init__(self, xArray: Iterable, prob: np.ndarray, time: float):
        super().__init__(xArray, prob, time)
        self.time_full = [time]

    def update_time(self, time: float):
        self.time_full.append(time)

    def report(self, obj: Integrator.FPE_Integrator_1D) -> Tuple[Iterable]:
        # NOTE Need to propagate time_full through the logic here
        return obj.workTracker, obj.powerTracker, self.time_full, *super().report()


def calc_normalization(prob: np.ndarray, xVals: np.ndarray) -> float:
    return si.trapz(prob, xVals)


def calcDiffusion(BC: Optional[str] = "hard-wall") -> ProbabilityTracker:
    dt = 0.005
    dx = 0.001
    D = 1.0
    xArray = np.arange(-1.5, 1.5, dx)

    obj = Integrator.FPE_Integrator_1D(D, dt, dx, xArray, boundaryCond=BC)
    obj.initializeProbability(0, 0.125)

    elapsed_time = 0.0
    counter = 0

    probRes = ProbabilityTracker(xArray, obj.getprob, elapsed_time)

    while elapsed_time <= 1.0:
        # obj.integrate_step((0,), ff.noForce)
        obj.diffusionUpdate()
        if counter % 5 == 0:
            probRes.update(obj.get_prob, elapsed_time)

        elapsed_time += dt
        counter += 1

    return probRes.report()


def calcAdvection(BC: Optional[str] = "periodic") -> ProbabilityTracker:
    dt = 0.00125
    dx = 0.001
    D = 1.0
    kVal = 1.0

    xArray = np.arange(-1.5, 1.5, dx)
    obj = Integrator.FPE_Integrator_1D(D, dt, dx, xArray, boundaryCond=BC)
    obj.initializeProbability(0, 0.125)

    elapsed_time = 0.0
    counter = 0

    probRes = ProbabilityTracker(xArray, obj.get_prob, elapsed_time)

    while elapsed_time <= 2.0:
        obj.advectionUpdate([kVal], ff.constantForce, dt)
        if counter % 25 == 0:
            probRes.update(obj.get_prob, elapsed_time)
        elapsed_time += dt
        counter += 1

    return probRes.report()


def calcHarmonicRelaxation(BC: Optional[str] = "periodic") -> ProbabilityTracker:
    dt = 0.001
    dx = 0.0075
    D = 1.0
    k_trap = 2.0
    trap_center = 0.5

    xArray = np.arange(-2.0, 2.0, dx)
    obj = Integrator.FPE_Integrator_1D(D, dt, dx, xArray, boundaryCond=BC)
    obj.initializeProbability(-0.5, 0.125)

    elapsed_time = 0.0
    counter = 0

    probRes = ProbabilityTracker(xArray, obj.get_prob, elapsed_time)

    while elapsed_time <= 1.0:
        obj.integrate_step([k_trap, trap_center], ff.harmonicForce)
        if counter % 25 == 0:
            probRes.update(obj.get_prob, elapsed_time)
        counter += 1
        elapsed_time += dt

    return probRes.report()


def calcHarmonicConstVel(trap_vel: float, trap_strength: float, BC: Optional[str] = "periodic") -> ProbabilityTracker:
    dt = 0.001
    dx = 0.0075
    D = 1.0
    k_trap = trap_strength
    trap_velocity = trap_vel
    beta = 1.0

    force_parameters = [k_trap, trap_velocity, D, beta, 0]
    force_params_fwd = [k_trap, trap_velocity, D, beta, trap_velocity * dt]

    xArray = np.arange(-2.0, 2.0, dx)
    obj = Integrator.FPE_Integrator_1D(D, dt, dx, xArray, boundaryCond=BC)
    # Initialize system to Eq dist for zero-velocity
    obj.initializeProbability(0.0, 0.25)

    elapsed_time = 0.0
    counter = 0
    # probRes = ProbabilityTracker(xArray, obj.get_prob, elapsed_time)
    probRes = ProbabilityWorkTracker(xArray, obj.get_prob, elapsed_time)

    while elapsed_time <= 1.0:
        # obj.integrate_step(force_parameters, ff.harmonicForce_constVel)
        obj.work_step(force_parameters, force_params_fwd, ff.harmonicForce_constVel, ff.harmonicEnergy_constVel)
        if counter % 25 == 0:
            probRes.update(obj.get_prob, elapsed_time)
        else:
            probRes.updateTime(elapsed_time)
        counter += 1
        elapsed_time += dt

    return probRes.report(obj)


def genTrackingPlot(
    density_tracker: List, time_tracker: List, norm_tracker: List,
    xArray: np.ndarray, write_name: str, write_path: str,
    write_format: Optional[str] = "pdf"
):
    sns.set(style="darkgrid")
    _, ax = plt.subplots(2, 1, figsize=(6.3, 4.5))
    Pal = sns.color_palette("Spectral", len(density_tracker))

    for i, prob in enumerate(density_tracker):
        ax[0].plot(xArray, prob, linewidth=2.5, color=Pal[i])

    ax[1].plot(time_tracker, norm_tracker, linewidth=2.5, color='k')

    ax[0].set_xlabel(r"Position $x$", fontsize=15)
    ax[0].set_ylabel(r"$p(x)$", fontsize=15)

    ax[1].set_xlabel(r"Time $t$", fontsize=15)
    ax[1].set_ylabel(r"$\sum p(x_i)\Delta x_i$", fontsize=15)

    ax[1].set_ylim([0, 1.1])

    plt.tight_layout()
    # plt.savefig(os.path.join(write_path, write_name), format=write_format)
    plt.show()
    plt.close()


def genHarmonicWorkPlot(
    work_arr: List, power_arr: List, time: List, vel_arr: List,
    vel_labels: List, trap_strength: float, write_dir: str, write_name: str
):
    sns.set(style="darkgrid")
    _, ax = plt.subplots(1, 2, figsize=(6.3, 3.5))
    Pal = sns.color_palette("husl", len(vel_arr))

    theory_arr_work = [calcHarmonicWork_theory(v, trap_strength, time) for v in vel_arr]
    theory_arr_power = [calcHarmonicPower_theory(v, trap_strength, time) for v in vel_arr]

    # Generate work plots
    for i, (w, w_t) in enumerate(zip(work_arr, theory_arr_work)):
        ax[0].plot(time, w, linewidth=2.0, color=Pal[i], label=vel_labels[i])
        ax[0].plot(
            time, w_t, '--', linewidth=1.0, color='k', alpha=0.6,
            label=(lambda x: "Theory" if x == 0 else None)(i)
        )

    # Generate power plots
    for i, (p, p_t) in enumerate(zip(power_arr, theory_arr_power)):
        ax[1].plot(time, p, linewidth=2.0, color=Pal[i])
        ax[1].plot(time, p_t, '--', linewidth=1.0, color='k', alpha=0.6)

    ax[0].legend(fontsize=12)
    ax[0].set_xlabel(r"Elapsed time $\tau$", fontsize=15)
    ax[0].set_ylabel(r"Mean work $\langle W\rangle_{[0, \tau]}$", fontsize=15)
    ax[1].set_xlabel(r"Elapsed time $\tau$", fontsize=15)
    ax[1].set_ylabel(r"Mean power input $\langle P(\tau)\rangle$", fontsize=15)

    plt.tightlayout()
    plt.savefig(os.path.join(write_dir, write_name), format="pdf")
    plt.show()
    plt.close()



def calcHarmonicPower_theory(
    velocity: float, trap_strength: float, time_tracker: Iterable,
    beta: Optional[float] = 1, D: Optional[float] = 1
) -> List:
    return (
        ((velocity ** 2) / beta * D)
        * (1 - np.exp(-beta * D * trap_strength * np.array(time_tracker)))
    )


def calcHarmonicWork_theory(
    velocity: float, trap_strength: float, time_tracker: Iterable,
    beta: Optional[float] = 1, D: Optional[float] = 1 
) -> List:
    return (
        ((velocity ** 2) / (beta * D))
        * (np.array(time_tracker)
        - (1 - np.exp(-beta * D * trap_strength) * np.array(time_tracker)) / (beta * D * trap_strength))
    )


def runDiffusionTests(write_dir: str):
    write_name_hw = "diffusion_example_HW.pdf"
    write_name_periodic = "diffusion_example_P.pdf"
    write_name_open = "diffusion_example_O.pdf"

    print("Working on hard-wall...")
    density_tracker, time_tracker, norm_tracker, xVals = calcDiffusion()
    genTrackingPlot(
        density_tracker, time_tracker, norm_tracker, xVals,
        write_name_hw, write_dir
    )

    print("Working on periodic...")
    density_tracker, time_tracker, norm_tracker, xVals = calcDiffusion(BC="periodic")
    genTrackingPlot(
        density_tracker, time_tracker, norm_tracker, xVals,
        write_name_periodic, write_dir
    )

    print("Working on open boundary...")
    density_tracker, time_tracker, norm_tracker, xVals = calcDiffusion(BC="open")
    genTrackingPlot(
        density_tracker, time_tracker, norm_tracker, xVals,
        write_name_open, write_dir
    )


def runAdvectionTests(write_dir: str):
    write_name = "advection_test_constForce.pdf"
    print("working on advection test (PBCs)")
    density_tracker, time_tracker, norm_tracker, xVals = calcAdvection()
    genTrackingPlot(
        density_tracker, time_tracker, norm_tracker, xVals, write_name,
        write_dir
    )


def runAdvectionDiffusionTests(write_dir: str):
    write_name_relax = "advection_diffusion_harmonicRelax.pdf"
    write_name_constVel = "advection_diffusion_harmonicConstVel.pdf"
    write_name_work = "harmonic_work_theoryCompare.pdf"

    print("Working on harmonic relaxation test (PBCs)")
    density_tracker, time_tracker, norm_tracker, xVals = calcHarmonicRelaxation()
    genTrackingPlot(
        density_tracker, time_tracker, norm_tracker, xVals, write_name_relax,
        write_dir
    )

    print("Working on harmonic constant-velosity (PBCs)")

    trap_strength = 2.0
    vel_arr = [1/2, 1, 2]
    vel_labels = ["1/2", "1", "2"]

    print(f"Velocity --> {vel_arr[0]}...")
    w_track_0, p_track_0, _, _, _, _ = calcHarmonicConstVel(vel_arr[0], trap_strength)

    print(f"Velocity --> {vel_arr[1]}...")
    w_track_1, p_track_1, dens_track_1, t_track_1, n_track_1, x_1 = calcHarmonicConstVel(vel_arr[1], trap_strength)

    print(f"Velocity --> {vel_arr[2]}...")
    w_track_2, p_track_2, _, _, _, _ = calcHarmonicConstVel(vel_arr[2], trap_strength)

    genTrackingPlot(
        dens_track_1, t_track_1, n_track_1, x_1, write_name_constVel,
        write_dir
    )

    genHarmonicWorkPlot(
        [w_track_0, w_track_1, w_track_2],
        [p_track_0, p_track_1, p_track_2],
        t_track_1, vel_arr, vel_labels, trap_strength, write_dir,
        write_name_work
    )


if __name__ == "__main__":
    proj_dir = Path().resolve().parents[1]
    write_dir = os.path.join(proj_dir, "results/visualizations/examples")

    # Diffusion test scenarios
    # runDiffusionTests(write_dir)

    # Advection test scenarios
    # runAdvectionTests(write_dir)

    # Relaxation in a harmonic potential
    runAdvectionDiffusionTests(write_dir)
