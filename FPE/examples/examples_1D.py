# Samples of the FPE Integrator in action

import os
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.integrate as si
from FPE import Integrator
# import FPE.forceFunctions as ff

from pathlib import Path


def calc_normalization(prob: np.ndarray, xVals: np.ndarray) -> float:
    return si.trapz(prob, xVals)


def calcDiffusion(BC: Optional[str] = "hard-wall") -> Tuple[List, List]:
    dt = 0.005
    dx = 0.001
    D = 1.0
    xArray = np.arange(-1.5, 1.5, dx)

    obj = Integrator.FPE_Integrator_1D(D, dt, dx, xArray, boundaryCond=BC)
    obj.initializeProbability(0, 0.125)

    elapsed_time = 0.0
    counter = 0
    density_tracker = [obj.get_prob]
    time_tracker = [elapsed_time]
    norm_tracker = [calc_normalization(obj.get_prob, xArray)]

    while elapsed_time <= 1.0:
        # obj.integrate_step((0,), ff.noForce)
        obj.diffusionUpdate()
        if counter % 5 == 0:
            density_tracker.append(obj.get_prob)
            time_tracker.append(elapsed_time)
            norm_tracker.append(calc_normalization(obj.get_prob, xArray))
        elapsed_time += dt
        counter += 1

    return density_tracker, time_tracker, norm_tracker, xArray


def genDiffusionOnlyPlot(
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


if __name__ == "__main__":
    proj_dir = Path().resolve().parents[1]
    write_dir = os.path.join(proj_dir, "results/visualizations/examples")
    write_name_hw = "diffusion_example_HW.pdf"
    write_name_periodic = "diffusion_example_P.pdf"
    write_name_open = "diffusion_example_O.pdf"

    print("Working on hard-wall...")
    density_tracker, time_tracker, norm_tracker, xVals = calcDiffusion()
    genDiffusionOnlyPlot(
        density_tracker, time_tracker, norm_tracker, xVals,
        write_name_hw, write_dir
    )

    # NOTE Issue with PBC here
    print("Working on periodic...")
    density_tracker, time_tracker, norm_tracker, xVals = calcDiffusion(BC="periodic")
    genDiffusionOnlyPlot(
        density_tracker, time_tracker, norm_tracker, xVals,
        write_name_periodic, write_dir
    )

    print("Working on open boundary...")
    density_tracker, time_tracker, norm_tracker, xVals = calcDiffusion(BC="open")
    genDiffusionOnlyPlot(
        density_tracker, time_tracker, norm_tracker, xVals,
        write_name_open, write_dir
    )
