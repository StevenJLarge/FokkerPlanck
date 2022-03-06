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

def calcAdvectionDiffusion(BC: Optional[str] = "periodic") -> ProbabilityTracker:
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
    plt.savefig(os.path.join(write_path, write_name), format=write_format)
    plt.show()
    plt.close()


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
    write_name = "acvection_diffusion_constForce.pdf"
    print("working on advection diffusion test (PCSs)")
    density_tracker, time_tracker, norm_tracker, xVals = calcAdvectionDiffusion()
    genTrackingPlot(
        density_tracker, time_tracker, norm_tracker, xVals, write_name,
        write_dir
    )


if __name__ == "__main__":
    proj_dir = Path().resolve().parents[1]
    write_dir = os.path.join(proj_dir, "results/visualizations/examples")

    # Diffusion test scenarios
    # runDiffusionTests(write_dir)

    # Advection test scenarios
    # runAdvectionTests(write_dir)

    #Relaxation in a harmonic potential
    runAdvectionDiffusionTests(write_dir)
